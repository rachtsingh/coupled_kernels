Currently reproducing the plots in https://arxiv.org/pdf/1708.03625.pdf.

See `maximal_coupling` in `coupled_distributions.py` for a generic construction of a maximal coupling between
any two PyTorch-style distributions. However, because this relies on extra synchronization, it maybe slower 
than desynchronizing the MCMC chains:

- Customized implementations (including CUDA) forthcoming in extensions.cpp - only the CPU normal-normal coupling
is implemented there at the moment.
