#include <torch/torch.h>
#include "ATen/CPUApplyUtils.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Generator.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Parallel.h"

using namespace at;

inline Tensor sort_strides(Tensor& tensor_) {
  IntList strides = tensor_.strides();
  std::vector<int64_t> indices;
  indices.reserve(tensor_.ndimension());
  for (int64_t i = 0; i < tensor_.ndimension(); i++) {
    indices.push_back(i);
  }
  std::sort(indices.begin(), indices.end(), [&strides](int64_t i1, int64_t i2) {
    return strides[i1] > strides[i2];
  });
  Tensor tensor = tensor_.permute(indices);
  return tensor;
}

template <typename Arg>
inline void _setup_arrays(Tensor& tensor, Arg* iter) {
  int64_t max_dim = tensor.ndimension();
  iter->dim_ = 0;
  for (int64_t i = 0; i < max_dim; i++) {
    int64_t size = tensor.size(i);
    int64_t stride = tensor.stride(i);
    while (i + 1 < max_dim &&
           (tensor.size(i + 1) == 1 ||
            tensor.stride(i) == tensor.size(i + 1) * tensor.stride(i + 1))) {
      size = size * tensor.size(i + 1);
      if (tensor.size(i + 1) != 1)
        stride = tensor.stride(i + 1);
      i++;
    }
    iter->sizes_[iter->dim_] = size;
    iter->strides_[iter->dim_] = stride;
    iter->dim_++;
  }
}

template <typename T, int N>
struct strided_tensor_iter_fixed {
 public:
  T* data_ = NULL;
  int64_t dim_;

  int64_t counter_[N];
  int64_t sizes_[N];
  int64_t strides_[N];

  strided_tensor_iter_fixed(strided_tensor_iter_fixed const&) = delete;
  void operator=(strided_tensor_iter_fixed const& x) = delete;
  strided_tensor_iter_fixed(strided_tensor_iter_fixed&&) = default;
  strided_tensor_iter_fixed(Tensor& tensor, bool sort_strides = false)
      : data_(tensor.data<T>()) {
    memset(counter_, 0, sizeof(int64_t) * N);
    _setup_arrays(tensor, this);
  }
};

template <typename T>
struct strided_tensor_iter {
 private:
 public:
  T* data_ = NULL;
  int64_t dim_;

  std::vector<int64_t> counter_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;

  strided_tensor_iter(strided_tensor_iter const&) = delete;
  void operator=(strided_tensor_iter const& x) = delete;
  strided_tensor_iter(strided_tensor_iter&&) = default;
  strided_tensor_iter(Tensor& tensor)
      : data_(tensor.data<T>()),
        dim_(tensor.ndimension()),
        counter_(dim_, 0),
        sizes_(tensor.sizes()),
        strides_(tensor.strides()) {
    _setup_arrays(tensor, this);
  }
};

inline bool _all_equal_numel(at::ArrayRef<Tensor> tensors) {
  if (tensors.size() == 0)
    return true;
  int64_t all_numel = tensors[0].numel();
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].numel() != all_numel)
      return false;
  }
  return true;
}

inline std::string _all_equal_numel_error(at::ArrayRef<Tensor> tensors) {
  std::ostringstream oss;
  oss << "inconsistent tensor size, expected ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].sizes() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1]
      << " to have the same number of elements, but got ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].numel() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1].numel()
      << " elements respectively";
  return oss.str();
}

inline bool _apply_preamble(ArrayRef<Tensor> tensors) {
  checkBackend("CPU_tensor_apply", tensors, Backend::CPU);
  if (!_all_equal_numel(tensors))
    throw std::runtime_error(_all_equal_numel_error(tensors));
  // An empty tensor has no elements
  for (auto& t : tensors)
    if (t.sizes().equals({0}))
      return false;
  internal::init_tbb_num_threads();
  return true;
}

inline int64_t _max_dim_tensors(ArrayRef<Tensor> tensors) {
  int64_t dim = 0;
  for (auto& t : tensors)
    dim = std::max(dim, t.ndimension());
  return dim;
}

inline void iterate(){};

template <typename Arg, typename... Args>
inline void iterate(Arg& iter, Args&... iter_tail) {
  iter.counter_[iter.dim_ - 1]++;
  iter.data_ += iter.strides_[iter.dim_ - 1];
  iterate(iter_tail...);
}

inline bool iterate_continue() {
  return true;
};

template <typename Arg, typename... Args>
inline bool iterate_continue(Arg& iter, Args&... iter_tail) {
  return iter.counter_[iter.dim_ - 1] < iter.sizes_[iter.dim_ - 1] &&
      iterate_continue(iter_tail...);
}

inline void iterate_overflow(){};

template <typename Arg, typename... Args>
inline void iterate_overflow(Arg& iter, Args&... iter_tail) {
  if (iter.counter_[iter.dim_ - 1] == iter.sizes_[iter.dim_ - 1]) {
    for (int64_t i = iter.dim_ - 1; i > 0; i--) {
      if (iter.counter_[i] == iter.sizes_[i]) {
        iter.counter_[i] = 0;
        iter.counter_[i - 1]++;
        iter.data_ = iter.data_ - (iter.sizes_[i] * iter.strides_[i]) +
            iter.strides_[i - 1];
      }
    }
  }
  iterate_overflow(iter_tail...);
}

inline void forward(int64_t offset){};

template <typename Arg, typename... Args>
inline void forward(int64_t offset, Arg& iter, Args&... iter_tail) {
  int64_t multi = offset;
  for (int64_t i = iter.dim_ - 1; i >= 0; i--) {
    int64_t inc = multi % iter.sizes_[i];
    multi = multi / iter.sizes_[i];
    iter.data_ = iter.data_ + inc * iter.strides_[i];
    iter.counter_[i] += inc;
  }
  forward(offset, iter_tail...);
}

inline int64_t max_dim() {
  return 0;
}

template <typename Arg, typename... Args>
inline int64_t max_dim(Arg& iter, Args&... iter_tail) {
  return std::max(iter.dim_, max_dim(iter_tail...));
}

inline void apply_op(){};

template <typename Op, typename... Args>
inline void
apply_op(int64_t numel, int64_t offset, const Op& op, Args... iters) {
  // For 0-dim tensors
  if (numel == 1 && max_dim(iters...) == 0) {
    op(*iters.data_...);
    return;
  }
  if (offset > 0)
    forward(offset, iters...);
  // Splitting this into chunks helps the compiler create faster assembly
  for (int64_t i = 0; i < numel;) {
    for (; iterate_continue(iters...) && i < numel;) {
      op(*iters.data_...);
      iterate(iters...);
      i++;
    }
    iterate_overflow(iters...);
  }
}

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename scalar5,
    typename scalar6,
    typename Op>
inline void CPU_tensor_apply6(
    Tensor tensor1,
    Tensor tensor2,
    Tensor tensor3,
    Tensor tensor4,
    Tensor tensor5,
    Tensor tensor6,
    const Op op) {
  if (!_apply_preamble({tensor1, tensor2, tensor3, tensor4, tensor5, tensor6}))
    return;
  if (_max_dim_tensors({tensor1, tensor2, tensor3, tensor4, tensor5, tensor6}) <= 8) {
    apply_op(
        tensor1.numel(),
        0,
        op,
        strided_tensor_iter_fixed<scalar1, 8>(tensor1),
        strided_tensor_iter_fixed<scalar2, 8>(tensor2),
        strided_tensor_iter_fixed<scalar3, 8>(tensor3),
        strided_tensor_iter_fixed<scalar4, 8>(tensor4),
        strided_tensor_iter_fixed<scalar5, 8>(tensor5),
        strided_tensor_iter_fixed<scalar6, 8>(tensor6));
  } else {
    apply_op(
        tensor1.numel(),
        0,
        op,
        strided_tensor_iter<scalar1>(tensor1),
        strided_tensor_iter<scalar2>(tensor2),
        strided_tensor_iter<scalar3>(tensor3),
        strided_tensor_iter<scalar4>(tensor4),
        strided_tensor_iter<scalar5>(tensor5),
        strided_tensor_iter<scalar6>(tensor6));
  }
}
