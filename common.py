"""Library module for common functions and constants."""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt


# Figure size suitable for Beamer presentations.
FIG_SIZE = (4.0, 2.47)


def savefig(fig, filename):
    """Save `fig` to file `filename`."""
    filename = os.path.join('figures', filename)
    if len(sys.argv) > 1:
        fig.savefig(filename)
    else:
        plt.show()


def setup_graphics():
    """Configure Matplotlib settings for nice plots in Beamer."""
    nice_fonts = {
        # Use LaTex to write all text
        'text.usetex': True,
        #'font.family': 'serif',
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',
        'font.serif': 'Stix Regular',
        # Use 10pt font in plots, to match 10pt font in document
        'axes.labelsize': 10,
        'font.size': 10,
        # Make the legend/label fonts a little smaller
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    }

    mpl.rcParams.update(nice_fonts)
    plt.style.use('seaborn-colorblind')
