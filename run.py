#!/usr/bin/env python
"""Conduct linear regression with three different least-squares algorithms."""
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from common import FIG_SIZE, savefig, setup_graphics
from leastsquares import OLS, LSConservativeFormulation, BadAndGoodLS


def main():
    """Main function of the script."""
    np.random.seed(42)
    setup_graphics()

    m = 10.0
    b = 350.0
    x = np.linspace(-5, 10, num=31)

    noise_level = 10

    data = m*x + b
    data = data + noise_level * np.random.normal(0, 1, size=len(data))
    print('Ideal data: m = {}, b = {}'.format(m, b))
    print('Noise variance: {}'.format(noise_level))

    print()
    print('Good data')
    print('---------')
    conduct(x, data, noise_level, 'good-data.pdf')

    data[2] *= 3
    data[5] *= 4
    data[6] *= 1.25
    data[9] *= 2.5
    data[13] *= 2.4
    data[15] *= 1.7
    data[21] *= 1.3

    print()
    print('Bad data')
    print('--------')
    conduct(x, data, noise_level, 'bad-data.pdf')

    print()
    print('BadAndGood: sensitivity')
    conduct_bad_and_good(x, data, 'fig-model-bad-and-good.pdf')


def conduct(x, data, noise, filename):
    """Conduct one experiment with data."""

    # Analyze the behavior of the non-normalized residuals first.
    lscf_def_noise = LSConservativeFormulation(x, data)
    lscf_def_noise.fit()

    fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE)
    ax[0].bar(x, lscf_def_noise.residuals)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel('Non-normalized residuals')

    kde = stats.gaussian_kde(lscf_def_noise.residuals)
    __, bins, __ = ax[1].hist(lscf_def_noise.residuals, density=True)
    xx = np.linspace(bins.min(), bins.max(), num=101)
    ax[1].plot(xx, kde(xx))
    ax[1].set_xlabel(r'$R$')
    ax[1].set_ylabel(r'Residuals PDF')

    fig.tight_layout(pad=0.1)
    savefig(fig, 'residuals-' + filename)

    # Now fit and plot the fitting results.
    ols = OLS(x, data, noise)
    ols.fit()

    lscf = LSConservativeFormulation(x, data, noise)
    lscf.fit()

    lsbag = BadAndGoodLS(x, data, noise=noise, beta=6/len(x), gamma=50)
    lsbag.fit()

    ols.print_result()
    lscf.print_result()
    lsbag.print_result()

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    ax.plot(x, data, 'o', label='Data')
    ax.plot(x, ols.result, '-', label='LS, ordinary')
    ax.plot(x, lscf.result, '--', label='LS, extension 1')
    ax.plot(x, lsbag.result, '-.', label='LS, extension 2')
    ax.legend(loc='best')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Observations and fit')
    fig.tight_layout(pad=0.1)
    savefig(fig, filename)


def conduct_bad_and_good(x, data, filename):
    """Study BadAndGoodLS algorithm."""
    r_1 = BadAndGoodLS(x, data, beta=0.01, gamma=40)
    r_1.fit()

    r_2 = BadAndGoodLS(x, data, beta=0.01, gamma=100)
    r_2.fit()

    r_3 = BadAndGoodLS(x, data, beta=0.01, gamma=200)
    r_3.fit()

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    ax.plot(x, data, 'o', label='Data')
    ax.plot(x, r_1.result, '-', label=r_1.name)
    ax.plot(x, r_2.result, '--', label=r_2.name)
    ax.plot(x, r_3.result, '-.', label=r_3.name)
    ax.legend(loc='best')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Observations and fit')

    fig.tight_layout(pad=0.1)

    savefig(fig, filename)

    r_1.print_result()
    r_2.print_result()
    r_3.print_result()


if __name__ == '__main__':
    main()
