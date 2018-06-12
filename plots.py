import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torch import distributions as dist
import pdb
import operator
from functools import reduce
from torch.utils.cpp_extension import load

from coupled_distribution import CoupledMHKernel, MHKernel, TargetDistribution, make_gaussian_kernel, flip, maximal_coupling

def test_coupled_kernel(target, kernel, X, Y, T=100):
    vals_X = torch.zeros(torch.Size([T]) + X.size())
    vals_Y = torch.zeros(torch.Size([T]) + Y.size())

    vals_X[0] = X.clone()
    vals_Y[0] = Y.clone()
    X = MHKernel(X, kernel, target)
    vals_X[1] = X.clone()
    vals_Y[1] = Y.clone() # duplicate the value because the chains are offset

    for i in range(2, T):
        vals_X[i] = X.clone()
        vals_Y[i] = Y.clone()
        X, Y = CoupledMHKernel(X, Y, kernel, target)

    return vals_X, vals_Y

def before_table_one():
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
    
    vals_X, vals_Y = test_coupled_kernel(target, kernel, X, Y, T=2000)

    # the chains match after a while
    diff = ((vals_X - vals_Y) != 0).sum(0)
    print("tau: (mean, variance, 99th \%\ile): ({:.3f}, {:.3f}, {:.3f})".format(
        diff.float().mean(),
        diff.float().var(),
        np.percentile(diff.numpy(), 99)
    ))

def hkm(h, k, m, vals_X, vals_Y):
    hX = h(vals_X).float()
    hY = h(vals_Y).float()
    diff = (hX - hY).sum(0) - (hX - hY).cumsum(0)
    hl = diff + hX
    return hl[k:m + 1].sum(0)/(m - k + 1)

def table_one():
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
    
    vals_X, vals_Y = test_coupled_kernel(target, kernel, X, Y, T=4000)

    print("k\tm\tCost\tVariance")
    for k in [1, 100, 200]:
        for m_fac in [1, 10, 20]:
            m = k * m_fac
            hk = hkm(lambda x: x > 3, k, m, vals_X, vals_Y)
            tau = ((vals_X - vals_Y) != 0).sum(0).float()
            cost = (2 * tau + torch.max(torch.ones(tau.size()), m - tau + 1)).mean()
            print("{}\t{} x k\t{:.1f}\t{:.1E}".format(k, m_fac, cost, hk.var()))

def coupled_gibbs_kernel(lamb1, lamb2, beta1, beta2, s, t, alpha, gamma, delta):
    lamb1, lamb2 = maximal_coupling(dist.Gamma(alpha + s, beta1 + t),
                                    dist.Gamma(alpha + s, beta2 + t)).sample()
    beta1, beta2 = maximal_coupling(dist.Gamma(gamma + 10 * alpha, delta + lamb1.sum()),
                                    dist.Gamma(gamma + 10 * alpha, delta + lamb2.sum())).sample()
    return lamb1, lamb2, beta1, beta2

def nuclear_pump():
    # NOTE: not quite working because of a bug in the Gamma sampler (might need to bootstrap with numpy)
    # set up data, set up sampling via MCMC chains, etc.
    s = torch.Tensor([5, 1, 5, 14, 3, 19, 1, 1, 4, 22]).double()
    t = torch.Tensor([94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5]).double()
    alpha = 1.802
    gamma = 0.01
    delta = 1.
    beta1 = dist.Gamma(torch.ones(1000).double() * 0.01, torch.ones(1000).double()).sample().view(1000, 1)
    lamb1 = dist.Gamma(alpha + s, beta1 + t).sample()
    beta2 = dist.Gamma(torch.ones(1000).double() * 0.01, torch.ones(1000).double()).sample().view(1000, 1)
    lamb2 = dist.Gamma(alpha + s, beta2 + t).sample()

    # draw chains for 2000 time steps
    vals_lamb1 = torch.zeros(2000, 1000, 10).double()
    vals_beta1 = torch.zeros(2000, 1000, 1).double()
    vals_lamb2 = torch.zeros(2000, 1000, 10).double()
    vals_beta2 = torch.zeros(2000, 1000, 1).double()

    # set up the X_1 for each, and copy Y_0 (offset chains)
    lamb1 = dist.Gamma(alpha + s, beta1 + t).sample()
    beta1 = dist.Gamma(gamma + 10 * alpha, delta + lamb1.sum()).sample()
    vals_lamb1[1] = lamb1
    vals_lamb2[1] = lamb2
    vals_beta1[1] = beta1
    vals_beta2[1] = beta2

    for t in range(2, 2000):
        lamb1, lamb2, beta1, beta2 = coupled_gibbs_kernel(lamb1, lamb2, beta1, beta2, 
                                                          s, t, alpha, gamma, delta)
        vals_lamb1[t] = lamb1.clone()
        vals_lamb2[t] = lamb2.clone()
        vals_beta1[t] = beta1.clone()
        vals_beta2[t] = beta2.clone()
    
    tau = ((vals_lamb1 - vals_lamb2) != 0).any() | ((vals_beta1 - vals_beta2) != 0)
    pdb.set_trace()
    sns.distplot(tau.sum(0).numpy())
    plt.show()
    pdb.set_trace()

    fig, axes = plt.subplots(1, 3, sharex=True)
    sns.distplot()

if __name__ == '__main__':
    table_one()