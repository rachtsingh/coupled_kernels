#include <torch/torch.h>
#include "ATen/CPUApplyUtils.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Generator.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Parallel.h"

#include "expanded_apply.h"

using namespace at;

THGenerator* get_generator(at::Generator* gen) {
  auto default_gen = &at::globalContext().defaultGenerator(at::Backend::CPU);
  auto gen_ = at::check_generator<at::CPUGenerator>(gen, default_gen);
  return gen_->generator;
}

inline double normal_density(double mean, double std, double x) {
	return std::exp(-(x - mean) * (x - mean)/(2 * (std * std))) * (1/std::sqrt(2 * M_PI * std * std));
}

Tensor normal_normal_couple_sample(const Tensor& p_mean, const Tensor& p_std, const Tensor &q_mean, const Tensor &q_std, const Tensor &x)
{
	Tensor y = q_mean.type().zeros(q_mean.sizes());
	AT_DISPATCH_FLOATING_TYPES(y.type(), "", [&]{
		// set up random generator
		THGenerator *generator = get_generator(nullptr);

		CPU_tensor_apply6<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>(p_mean, p_std, q_mean, q_std, x, y, 
			[generator](const scalar_t& p_mean, const scalar_t& p_std, const scalar_t& q_mean, const scalar_t& q_std, const scalar_t& x, scalar_t& y) {
				double prob = normal_density(p_mean, p_std, x);
				double qprob = normal_density(q_mean, q_std, x);
				double u = THRandom_uniform(generator, 0.0, prob);
				if (u < qprob) {
					y = x;
					return;
				}
				do {
					y = THRandom_normal(generator, q_mean, q_std);
					prob = normal_density(p_mean, p_std, y);
					qprob = normal_density(q_mean, q_std, y);
					u = THRandom_uniform(generator, 0.0, qprob);
				} while (u <= prob);
			}
		);
	});
	return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) // compile using extension name passed from Python
{
    m.def("normal_normal_couple_sample", &normal_normal_couple_sample, "Sample from maximal coupling of normal distributions");
}
