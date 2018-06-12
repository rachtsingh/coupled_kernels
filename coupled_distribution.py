"""
Work to do today:
1. debug the implementation of the code as written, and get 5.1 working
2. write the C++ implementation, delving into the ATen hooks necessary (malmaud + examples)
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torch import distributions as dist
import pdb
import operator
from functools import reduce
from torch.utils.cpp_extension import load

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

my_extension = load(
        "my_extension", 
        sources=["extension.cpp"],
        extra_include_paths=["tbb/include/"],
        verbose=True)

def maximal_coupling(p, q):
    lookup = {
        (dist.Normal, dist.Normal): my_extension.normal_normal_couple_sample
    }

    class CoupledDistribution(dist.Distribution):
        def __init__(self, p, q):
            super(CoupledDistribution, self).__init__()
            self.p = p
            self.q = q

        def sample(self, sample_shape=torch.Size()):
            X = self.p.sample(sample_shape)

            # short-circuit into C++
            if (type(p), type(q)) in lookup.keys():
                # print("short-circuiting into C++")
                func = lookup[(type(p), type(q))]
                return (X, func(self.p.loc, self.p.scale, self.q.loc, self.q.scale, X))

            Y = X.clone()
            W = dist.Uniform(0, self.p.log_prob(X).exp()).sample()

            threshold = self.q.log_prob(Y).exp()
            msk = (W <= threshold) # which samples have succeeded

            # basically, ignore the values where (msk == 1), because they're finalized
            # this is inefficient because:
            # 1 - this creates many extra samples - because threads don't exit
            # 2 - this computes the log prob many more times
            # we can speed this up using a custom implementation of a coupled kernel
            while not msk.all():
                Yp = self.q.sample()
                threshold = self.p.log_prob(Yp).exp()
                W = dist.Uniform(0, self.q.log_prob(Yp).exp()).sample()
                add = ((1 - msk) & (W > threshold))
                Y = Y * (1 - add).type(Y.type()) + Yp * add.type(Y.type())
                msk += add
            return (X, Y)
    return CoupledDistribution(p, q)

def main_test_couple():
    vals = torch.zeros(1000, 2)
    coupling = maximal_coupling(dist.Normal(0.5, 0.8), dist.Normal(-0.5, 0.2))
    for i in range(1000):
        vals[i] = torch.stack(coupling.sample())
    plt.scatter(vals[:, 0], vals[:, 1], s=1)
    plt.show()
    sns.distplot(vals[:, 0])
    plt.show()
    sns.distplot(vals[:, 1])
    plt.show()

def make_gaussian_kernel(scale):
    def GaussianKernel(loc):
        return dist.Normal(loc, scale)
    return GaussianKernel

# test to make sure that this gives a valid MCMC step
# this is as efficient as can be
def CoupledMHKernel(X, Y, q, pi):
    """
    q is a Kernel, pi is the target distribution - needs only to have a 'log_prob' function
    """
    X = X.clone()
    Y = Y.clone()
    Xp, Yp = maximal_coupling(q(X), q(Y)).sample()
    
    # sample U once!
    U = dist.Uniform(0, 1).sample(Xp.size())
    msk = U.log() <= pi.log_prob(Xp) + q(Xp).log_prob(X) - pi.log_prob(X) - q(X).log_prob(Xp)
    X = (1 - msk).float() * X + msk.float() * Xp
    msk = U.log() <= pi.log_prob(Yp) + q(Yp).log_prob(Y) - pi.log_prob(Y) - q(Y).log_prob(Yp)
    Y = (1 - msk).float() * Y + msk.float() * Yp
    return (X, Y)

class TargetDistribution(object):
    def __init__(self, log_prob):
        self.lp = log_prob

    def log_prob(self, X):
        return self.lp(X)

def test_coupled_kernel(target, kernel, X, Y, T=100):
    vals_X = torch.zeros(torch.Size([T]) + X.size())
    vals_Y = torch.zeros(torch.Size([T]) + Y.size())

    for i in range(T):
        vals_X[i] = X.clone()
        vals_Y[i] = Y.clone()
        X, Y = CoupledMHKernel(X, Y, kernel, target)

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    # check that the marginals are correct
    sns.distplot(vals_X.view(-1).numpy(), ax=axes[0], label='X')
    sns.distplot(vals_Y.view(-1).numpy(), ax=axes[1], label='Y')
    plt.legend()
    plt.show()

    # the chains seem to match after a while
    pdb.set_trace()

def mult(size):
    if size == torch.Size([]):
        return 1
    else:
        return reduce(operator.mul, list(size))

def MHKernel(X, q, pi):
    X = X.clone()
    Xp = q(X).sample()
    U = dist.Uniform(0, 1).sample(X.size())
    msk = (U.log() <= pi.log_prob(Xp) + q(Xp).log_prob(X) - (pi.log_prob(X) + q(X).log_prob(Xp)))
    X = (1 - msk).float() * X + msk.float() * Xp
    return X

def test_mh(target, kernel, X, T=100):
    vals = torch.zeros(torch.Size([T]) + X.size())
    for i in range(T):
        vals[i] = X.clone()
        X = MHKernel(X, kernel, target)

    sns.distplot(vals[500:].view(-1).numpy())
    xes = np.linspace(-20, 20, 250).astype(np.float32)
    plt.show()

def main_parallel():
    a = dist.Normal(-4, 1)
    b = dist.Normal(4, 1)

    def lp(x):
        # compute log_probability of mixture of Gaussians
        la = a.log_prob(x)
        lb = b.log_prob(x)
        m = torch.max(la, lb)
        ret = m + ((la - m).exp() + (lb - m).exp()).log() - np.log(2)
        return ret

    target = TargetDistribution(lp)

    kernel = make_gaussian_kernel(torch.ones(25) * 3)
    X = dist.Normal(torch.zeros(25), 4).sample()
    Y = dist.Normal(torch.zeros(25), 4).sample()

    test_mh(target, kernel, X, T=1000)

def main_coupled():
    a = dist.Normal(-4, 1)
    b = dist.Normal(4, 1)

    def lp(x):
        la = a.log_prob(x)
        lb = b.log_prob(x)
        m = torch.max(la, lb)
        ret = m + ((la - m).exp() + (lb - m).exp()).log() - np.log(2)
        return ret
    target = TargetDistribution(lp)

    kernel = make_gaussian_kernel(3)
    X = dist.Normal(10, 10).sample(torch.Size([1000]))
    Y = dist.Normal(10, 10).sample(torch.Size([1000]))
    test_coupled_kernel(target, kernel, X, Y, T=200)

def main():
    a = dist.Normal(-4, 1)
    b = dist.Normal(4, 1)

    def lp(x):
        la = a.log_prob(x)
        lb = b.log_prob(x)
        m = torch.max(la, lb)
        ret = m + ((la - m).exp() + (lb - m).exp()).log() - np.log(2)
        return ret
    target = TargetDistribution(lp)

    kernel = make_gaussian_kernel(3)
    X = dist.Normal(0, 10).sample()
    Y = dist.Normal(0, 10).sample()

    test_mh(target, kernel, X, T=1000)

if __name__ == '__main__':
    main_coupled()
