#!/usr/bin/env python
"""Plot likelihood versus residual for a single datum.

The plots are for ordinary least squares and conservative formulation.
The point is two show that the conservative formulation has less skewing effect
in case of an outlier (corresponds to residual going to plus-minus infinity).

USAGE:
    python plot_datum_likelihood.py 1

where 1 is an optional flag to save figures on disk instead of displaying them.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_SIZE
from common import savefig, setup_graphics


def main():
    """Plot the likelihood functions versus residual."""
    R = np.linspace(-8, 8, num=201)
    R2 = R**2
    R2[np.abs(R2) < 1e-12] = 1e-12
    p_ols = np.exp(-R2/2.0)
    p_lscf = (1 - np.exp(-R2/2.0)) / R2

    setup_graphics()
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    ax.semilogy(R, p_ols, '-', label='LS, ordinary')
    ax.semilogy(R, p_lscf, '--', label='LS, extension 1')
    ax.set_ylim((1e-2, 1e0))
    ax.set_xlabel(r'Residual $R$')
    ax.set_ylabel(r'Log-likelihood for a datum')
    ax.legend(loc='best')

    fig.tight_layout(pad=0.1)
    savefig(fig, 'likelihood-vs-residual.pdf')


if __name__ == '__main__':
    main()
