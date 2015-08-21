import functools

import numpy as np
import matplotlib.pyplot as plt


def subplot(plot_function):
    """Wrapper for functions that plot on a matplotlib axes instance
    that autocreates a figure and axes instance if ax is not
    supplied.
    """
    @functools.wraps(plot_function)
    def f(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            return plot_function(self, ax, **kwargs)
        else:
            return plot_function(self, ax, **kwargs)
    return f


class SparsePlots(object):
    def __init__(self, sparsedmd):
        self.dmd = sparsedmd.dmd
        self.sparsedmd = sparsedmd

        self.xdmd = self.dmd.xdmd
        self.Ydmd = self.dmd.Ydmd
        self.Edmd = self.dmd.Edmd
        self.Fdmd = self.dmd.Fdmd

        self.sparse = sparsedmd.sparse

    @subplot
    def xdmd_frequency(self, ax):
        """|xdmd| vs frequency"""
        ax.plot(np.log(self.Edmd).imag, np.abs(self.xdmd), 'ko')
        ax.set_xlabel('frequency')
        ax.set_ylabel('amplitude')

    @subplot
    def xdmd_real(self, ax):
        """|xdmd| vs real part"""
        ax.plot(np.log(self.Edmd).real, np.abs(self.xdmd), 'ko')
        ax.set_xlabel('real')
        ax.set_ylabel('amplitude')

    @subplot
    def performance_loss_gamma(self, ax):
        """Performance loss for the polished vector of amplitudes vs gamma"""
        ax.semilogx(self.sparse.gamma, self.sparse.Ploss, 'ko',
                    linewidth=1, markersize=7)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel('performance loss (%)')
        ax.axis([self.sparse.gamma[0], self.sparse.gamma[-1],
                 0, 1.05 * self.sparse.Ploss[-1]])

    @subplot
    def nonzero_gamma(self, ax):
        """Number of non-zero amplitudes vs gamma"""
        ax.semilogx(self.sparse.gamma, self.sparse.Nz,
                    'ko', linewidth=1, markersize=7)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$N_z$')
        ax.axis([self.sparse.gamma[0], self.sparse.gamma[-1],
                 0, 1.05 * self.sparse.Nz[0]])

    @subplot
    def spectrum_gamma(self, ax, m=20):
        """Spectrum of DT system for a certain value of gamma"""
        ival = np.where(self.sparse.xsp[:, m])  # non zero amplitudes
        ax.plot(self.Edmd.real, self.Edmd.imag, 'ko',
                self.Edmd[ival].real, self.Edmd[ival].imag, 'r+',
                linewidth=1, markersize=7)
        ax.set_xlabel(r'$Re(\mu_i)$')
        ax.set_ylabel(r'$Im(\mu_i)$')

        # plot a unit circle
        theta = np.linspace(0, 2 * np.pi, 100)  # create vector theta
        x = np.cos(theta)                       # generate x-coordinate
        y = np.sin(theta)                       # generate y-coordinate
        ax.plot(x, y, 'b--', linewidth=1)       # plot unit circle
        ax.axis('equal')

    @subplot
    def xdmd_xpol_frequency(self, ax, m=20):
        """|xdmd| and |xpol| vs frequency for a certain value of
        gamma amplitudes in log scale
        """
        ival = np.where(self.sparse.xsp[:, m])  # non zero amplitudes
        ax.semilogy(np.log(self.Edmd).imag, np.abs(self.xdmd),
                    'ko', linewidth=1, markersize=7)
        ax.semilogy(np.log(self.Edmd[ival]).imag,
                    np.abs(self.sparse.xpol[ival, m].squeeze()),
                    'r+', linewidth=1, markersize=7)
        ax.set_xlabel('frequency')
        ax.set_ylabel('amplitude')

    @subplot
    def performance_loss_nmodes(self, ax):
        """Performance loss vs number of dmd modes
        """
        # we only care about the performance loss
        # when the number of modes changes
        n_modes = self.sparse.Nz
        n_change = np.where(np.diff(n_modes))

        Nz = self.sparse.Nz[n_change]
        Ploss = self.sparse.Ploss[n_change]

        ax.plot(Nz, Ploss, 'ko')
        ax.set_xlabel('number of dmd modes')
        ax.set_ylabel('performance loss (%)')
        ax.axis([Nz[-1], Nz[0], 0, 1.05 * Ploss[-1]])
