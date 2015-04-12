# The entire dmdsp method for the channel example
# translated line by line from matlab source

# Start of run_dmdsp
from collections import namedtuple
import functools

import numpy as np
import scipy.linalg as linalg
import scipy.io
import matplotlib.pyplot as plt


def to_snaps(data, decomp_axis=-1):
    """Transform data into snapshots, using decomp_axis as
    the decomposition axis.
    """
    sd = data.shape[decomp_axis]   # size of the decomp axis
    shape_idx = range(len(data.shape))
    # we reset the value of decomp_axis to allow indexing with -1
    decomp_axis = shape_idx.pop(decomp_axis)
    # put the decomp axis last and reshape to 2d array
    return data.transpose(shape_idx + [decomp_axis]).reshape((-1, sd))


def to_data(snapshots, shape, decomp_axis=-1):
    """Reshape snapshots or modes to match data that
    had a `shape` and that was decomposed along `decomp_axis`.
    """
    shape = list(shape)
    shape.pop(decomp_axis)  # remove decomp axis (it is implicitly last)
    reshaped = np.asarray(snapshots).reshape(shape + [-1])

    rshape = list(reshaped.shape)
    irshape = range(len(rshape))  # indices of array shape
    id = irshape.pop(-1)  # pull out index of decomp axis
    irshape.insert(decomp_axis, id)  # and insert where it came from
    return reshaped.transpose(irshape)


def run_dmdsp(UstarX1, S, V, gammaval):
    """Emulates behaviour of run_dmdsp in the matlab source.

    Inputs: matrices U'*X1, S, and V (for a specified flow type)

    Outputs:

    Fdmd - optimal matrix on the subspace spanned by the POD modes U of X0
    Edmd - eigenvalues of Fdmd
    Ydmd - eigenvectors of Fdmd
    xdmd - optimal vector of DMD amplitudes
    answer - gamma-parameterized structure containing output of dmdsp
    """
    spdmd = SparseDMD()
    spdmd.dmd.init(UstarX1, S, V)

    Fdmd = spdmd.dmd.Fdmd
    Ydmd = spdmd.dmd.Ydmd
    Edmd = spdmd.dmd.Edmd
    xdmd = spdmd.dmd.xdmd

    answer = spdmd.dmdsp(gammaval)

    return Fdmd, Edmd, Ydmd, xdmd, answer


class DMD(object):
    def __init__(self, snapshots=None, dt=1):
        """Dynamic Mode Decomposition with optimal amplitudes.

        Arguments
            snapshots - the matrix of snapshots
            dt - time interval between snapshots

        Methods
            compute - perform the decomposition on the snapshots and
                      calculate the modes, frequencies and amplitudes
            dmd_reduction - compute the matrices U*X1, S and V that
                            represent the input snapshots
            init - perform the dmd computation given a reduction

        Attributes
            modes - the dmd modes
            amplitudes - corresponding optimal amplitudes
            ritz_values - corresponding ritz values
            frequencies - corresponding frequencies
        """
        self.snapshots = snapshots
        self.dt = dt
        self.computed = False

    def compute(self):
        reduction = self.dmd_reduction(self.snapshots)
        self.init(reduction.UstarX1, reduction.S, reduction.V)

        self.modes = np.dot(reduction.U, self.Ydmd)
        self.ritz_values = self.Edmd
        self.frequencies = np.log(self.ritz_values) / self.dt
        # the optimal amplitudes
        self.amplitudes = self.xdmd
        self.computed = True

    @property
    def reduction(self):
        """Compute the reduced form of the data snapshots"""
        return self.dmd_reduction(self.snapshots)

    @staticmethod
    def dmd_reduction(snapshots):
        """Takes a series of snapshots and splits into two subsequent
        series, X0, X1, where

            snapshots = [X0, X1[-1]]

        then computes the (economy) single value decomposition

            X0 = U S V*

        returns an object with attributes

            UstarX1 - U* X1, the left singular vectors (POD modes)
                      projected onto the data
            U       - U, the left singular vectors
            S       - S, the singular values
            V       - V, the right singular vectors
            X0      - X0, snapshots[:, :-1]
            X1      - X1, snapshots[:. 1:]
        """
        X0 = snapshots[:, :-1]
        X1 = snapshots[:, 1:]

        # economy size SVD of X0
        U, S, Vh = linalg.svd(X0, full_matrices=False)
        ## n.b. differences with matlab svd:
        ## 1. S is a 1d array of diagonal elements
        ## 2. Vh == V': the matlab version returns V for X = U S V',
        ##    whereas python returns V'
        S = np.diag(S)
        V = Vh.T.conj()

        # truncate zero values from svd
        r = np.linalg.matrix_rank(S)
        U = U[:, :r]
        S = S[:r, :r]
        V = V[:, :r]

        # Determine matrix UstarX1
        UstarX1 = np.dot(U.T.conj(), X1)   # conjugate transpose (U' in matlab)

        reduction_keys = ('UstarX1', 'S', 'V', 'U', 'X0', 'X1')
        Reduction = namedtuple('DMDReduction', reduction_keys)

        return Reduction(UstarX1=UstarX1, S=S, V=V, U=U, X0=X0, X1=X1)

    def init(self, UstarX1=None, S=None, V=None):
        """Calculate the DMD matrix from the data reduction."""
        if UstarX1 is not None:
            self.UstarX1 = UstarX1
        if S is not None:
            self.S = S
        if V is not None:
            self.V = V

        self.UstarX1 = UstarX1
        self.S = S
        self.V = V

        Vstar = self.V.T.conj()

        # The number of snapshots
        N = Vstar.shape[1]

        # Optimal DMD matrix resulting from Schmid's 2010 algorithm
        # Fdmd = U'*X1*V*inv(S)
        self.Fdmd = np.dot(np.dot(self.UstarX1, self.V),
                                  linalg.inv(self.S))

        # eigenvalue decomposition of Fdmd
        self.Edmd, self.Ydmd = linalg.eig(self.Fdmd)

        # Form Vandermonde matrix
        # Vand = Edmd ** np.arange(N)[None].T
        self.Vand = np.vander(self.Edmd, N).T[::-1].T

        # Determine optimal vector of amplitudes xdmd
        # Objective: minimize the least-squares deviation between
        # the matrix of snapshots X0 and the linear combination of
        # the dmd modes
        # Can be formulated as:
        # minimize || G - L*diag(xdmd)*R ||_F^2
        L = self.Ydmd
        R = self.Vand
        G = np.dot(self.S, Vstar)

        # Form matrix P, vector q, and scalar s, where
        # J = x'*P*x - q'*x - x'*q + s
        # with x as the optimization variable (i.e., the unknown
        # vector of amplitudes). We seek to minimize J to find the
        # optimal amplitudes.
        self.P = np.dot(L.T.conj(), L) * np.dot(R, R.T.conj()).conj()
        self.q = np.diagonal(np.dot(np.dot(R, G.T.conj()), L)).conj()
        self.s = np.trace(np.dot(G.T.conj(), G))

        # Optimal vector of amplitudes xdmd
        # computed by cholesky factorization of P (more efficient
        # way of computing x = P^(-1) q, as P is hermitian positive
        # definite matrix) === solve(Pl.T.conj(), solve(Pl, self.q))
        self.xdmd = linalg.cho_solve(linalg.cho_factor(self.P), self.q)


class SparseDMD(object):
    def __init__(self, snapshots=None, dmd=None, rho=1, maxiter=10000,
                 eps_abs=1e-6, eps_rel=1e-4):
        # TODO: allow data, decomp_axis as an argument instead of snapshots
        """Sparse Dynamic Mode Decomposition, using ADMM to find a
        sparse set of optimal dynamic mode amplitudes

        Inputs:
            snapshots - the matrix of data snapshots, shape (d, N)
                        where N is the number of snapshots and d is
                        the number of data points in a snapshot.
            rho     - augmented Lagrangian parameter
            maxiter - maximum number of ADMM iterations
            eps_abs - absolute tolerance for ADMM
            eps_rel - relative tolerance for ADMM

        Defaults:
            If snapshots is not supplied and you have precomputed
            the dmd reduction [U^*X1, S, V], you can initialise the
            dmd with SparseDMD.dmd.init(U^*X1, S, V).

            rho = 1
            maxiter = 10000
            eps_abs = 1.e-6
            eps_rel = 1.e-4
        """
        self.rho = rho
        self.max_admm_iter = maxiter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        if snapshots is not None:
            self.dmd = DMD(snapshots)
            self.dmd.compute()
        elif not snapshots and dmd is not None:
            self.dmd = dmd
        elif not snapshots and not dmd:
            self.dmd = DMD()

    def compute_sparse(self, gammaval):
        """Compute the sparse dmd structure and set as attribute."""
        self.gammaval = gammaval
        self.sparse = self.dmdsp(gammaval)

    def dmdsp(self, gammaval):
        """Inputs:
            gammaval - vector of gamma to perform sparse optimisation over

        Returns:
            answer - gamma-parameterized structure containing
                answer.gamma - sparsity-promoting parameter gamma
                answer.xsp   - vector of amplitudes resulting from (SP)
                answer.xpol  - vector of amplitudes resulting from (POL)
                answer.Jsp   - J resulting from (SP)
                answer.Jpol  - J resulting from (POL)
                answer.Nz    - number of nonzero elements of x
                answer.Ploss - optimal performance loss 100*sqrt(J(xpol)/J(0))

        Additional information:

        http://www.umn.edu/~mihailo/software/dmdsp/
        """
        # Number of optimization variables
        self.n = len(self.dmd.q)
        # length of parameter vector
        ng = len(gammaval)

        self.Prho = self.dmd.P + (self.rho / 2.) * np.identity(self.n)

        answer = SparseAnswer(self.n, ng)
        answer.gamma = gammaval

        for i, gamma in enumerate(gammaval):
            ret = self.optimize_gamma(gamma)

            answer.xsp[:, i] = ret['xsp']
            answer.xpol[:, i] = ret['xpol']

            answer.Nz[i] = ret['Nz']
            answer.Jsp[i] = ret['Jsp']
            answer.Jpol[i] = ret['Jpol']
            answer.Ploss[i] = ret['Ploss']

        answer.nonzero[:] = answer.xsp != 0

        return answer

    def optimize_gamma(self, gamma):
        """Minimise
            J(a)
        subject to
            E^T a = 0

        This amounts to finding the optimal amplitudes for a given
        sparsity. Sparsity is encoded in the structure of E.

        The first step is solved using ADMM.

        The second constraint is satisfied using KKT_solve.
        """
        # Use ADMM to solve the gamma-parameterized problem,
        # minimising J, with initial conditions z0, y0
        y0 = np.zeros(self.n)  # Lagrange multiplier
        z0 = np.zeros(self.n)  # initial amplitudes
        z = self.admm(z0, y0, gamma)

        # Now use the minimised amplitudes as the input to the
        # sparsity contraint to create a vector of polished
        # (optimal) amplitudes
        xpol = self.KKT_solve(z)[:self.n]

        # outputs that we care about...
        # vector of amplitudes
        sparse_amplitudes = z
        # number of non-zero amplitudes
        num_nonzero = (z != 0).sum()
        # least squares residual
        residuals = self.residuals(z)

        # Vector of polished (optimal) amplitudes
        polished_amplitudes = xpol
        # Polished (optimal) least-squares residual
        polished_residual = self.residuals(xpol)
        # Polished (optimal) performance loss
        polished_performance_loss = 100 * \
            np.sqrt(polished_residual / self.dmd.s)

        return {'xsp':   sparse_amplitudes,
                'Nz':    num_nonzero,
                'Jsp':   residuals,
                'xpol':  polished_amplitudes,
                'Jpol':  polished_residual,
                'Ploss': polished_performance_loss,
                }

    def admm(self, z, y, gamma):
        """Alternating direction method of multipliers."""
        # Optimization:
        # This has been reasonably optimized already and performs ~3x
        # faster than a naive translation of the matlab version.

        # Two major changes are a custom function for calculating
        # the norm of a 1d vector and accessing the lapack solver
        # directly.

        # However it still isn't as fast as matlab (~1/3rd the
        # speed).

        # There are two complexity sources:
        # 1. the matrix solver. I can't see how this can get any
        #    faster (tested with Intel MKL on Canopy).
        # 2. the test for convergence. This is the dominant source
        #    now (~3x the time of the solver)

        # One simple speedup (~2x faster) is to only test
        # convergence every n iterations (n~10). However this breaks
        # output comparison with the matlab code. This might not
        # actually be a problem.

        # Further avenues for optimization:
        # - write in cython and import as compiled module, e.g.
        #   http://docs.cython.org/src/userguide/numpy_tutorial.html
        # - use two cores, with one core performing the admm and
        #   the other watching for convergence.

        a = (gamma / self.rho)
        q = self.dmd.q

        # precompute cholesky decomposition
        C = linalg.cholesky(self.Prho, lower=False)
        # link directly to LAPACK fortran solver for positive
        # definite symmetric system with precomputed cholesky decomp:
        potrs, = linalg.get_lapack_funcs(('potrs',), arrays=(C, q))

        # simple norm of a 1d vector, called directly from BLAS
        norm, = linalg.get_blas_funcs(('nrm2',), arrays=(q,))

        # square root outside of the loop
        root_n = np.sqrt(self.n)

        for ADMMstep in xrange(self.max_admm_iter):
            ### x-minimization step (alpha minimisation)
            u = z - (1. / self.rho) * y
            qs = q + (self.rho / 2.) * u
            # Solve P x = qs, using fact that P is hermitian and
            # positive definite and assuming P is well behaved (no
            # inf or nan).
            xnew = potrs(C, qs, lower=False, overwrite_b=False)[0]
            ###

            ### z-minimization step (beta minimisation)
            v = xnew + (1 / self.rho) * y
            # Soft-thresholding of v
            # zero for |v| < a
            # v - a for v > a
            # v + a for v < -a
            # n.b. This doesn't actually do this because v is
            # complex. This is the same as the matlab source. You might
            # want to use np.sign, but this won't work because v is complex.
            abs_v = np.abs(v)
            znew = ((1 - a / abs_v) * v) * (abs_v > a)
            ###

            ### Lagrange multiplier update step
            y = y + self.rho * (xnew - znew)
            ###

            ### Test convergence of admm
            # Primal and dual residuals
            res_prim = norm(xnew - znew)
            res_dual = self.rho * norm(znew - z)

            # Stopping criteria
            eps_prim = root_n * self.eps_abs \
                        + self.eps_rel * max(norm(xnew), norm(znew))
            eps_dual = root_n * self.eps_abs + self.eps_rel * norm(y)

            if (res_prim < eps_prim) & (res_dual < eps_dual):
                return z
            else:
                z = znew

        return z

    def KKT_solve(self, z):
        """Polishing of the sparse vector z. Seeks solution to
        E^T z = 0
        """
        # indices of zero elements of z (i.e. amplitudes that
        # we are ignoring)
        ind_zero = np.where(abs(z) < 1E-12)

        # number of zero elements
        m = len(ind_zero[0])

        # Polishing of the nonzero amplitudes
        # Form the constraint matrix E for E^T x = 0
        E = np.identity(self.n)[:, ind_zero].squeeze()
        # n.b. we don't form the sparse matrix as the original
        # matlab does as it doesn't seem to affect the
        # computation speed or the output.
        # If you want to use a sparse matrix, use the
        # scipy.sparse.linalg.spsolve solver with a csc matrix
        # and stack using scipy.sparse.{hstack, vstack}

        # Form KKT system for the optimality conditions
        KKT = np.vstack((np.hstack((self.dmd.P, E)),
                         np.hstack((E.T.conj(), np.zeros((m, m))))
                         ))
        rhs = np.hstack((self.dmd.q, np.zeros(m)))

        # Solve KKT system
        return linalg.solve(KKT, rhs)

    def residuals(self, x):
        """Calculate the residuals from a minimised
        vector of amplitudes x.
        """
        # conjugate transpose
        x_ = x.T.conj()
        q_ = self.dmd.q.T.conj()

        x_P = np.dot(x_, self.dmd.P)
        x_Px = np.dot(x_P, x)
        q_x = np.dot(q_, x)

        return x_Px.real - 2 * q_x.real + self.dmd.s

    def compute_sparse_reconstruction(self, Ni, shape=None, decomp_axis=1):
        """Compute a reconstruction of the input data based on a sparse
        selection of modes.

        Ni - the index that selects the desired number of
             modes in self.sparse.Nz

        shape - the shape of the original input data. If not supplied,
                the original snapshots will be reconstructed.

        Once computed, the reconstruction is available as
        self.reconstruction and has the following attributes:

        r.nmodes  # number of modes (3)
        r.data    # the original data
        r.rdata   # the reconstructed data (or snapshots)
        r.modes   # the modes (3 of them)
        r.freqs   # corresponding complex frequencies
        r.amplitudes  # corresponding amplitudes
        r.ploss   # performance loss
        """
        self.reconstruction = SparseReconstruction(self, Ni, data, decomp_axis)


class SparseAnswer(object):
    """A set of results from sparse dmd optimisation.

    Attributes:
    gamma     the parameter vector
    nz        number of non-zero amplitudes
    nonzero   where modes are nonzero
    jsp       square of frobenius norm (before polishing)
    jpol      square of frobenius norm (after polishing)
    ploss     optimal performance loss (after polishing)
    xsp       vector of sparse amplitudes (before polishing)
    xpol      vector of amplitudes (after polishing)
    """
    def __init__(self, n, ng):
        """Create an empty sparse dmd answer.

        n - number of optimization variables
        ng - length of parameter vector
        """
        # the parameter vector
        self.gamma = np.zeros(ng)
        # number of non-zero amplitudes
        self.Nz = np.zeros(ng)
        # square of frobenius norm (before polishing)
        self.Jsp = np.zeros(ng, dtype=np.complex)
        # square of frobenius norm (after polishing)
        self.Jpol = np.zeros(ng, dtype=np.complex)
        # optimal performance loss (after polishing)
        self.Ploss = np.zeros(ng, dtype=np.complex)
        # vector of amplitudes (before polishing)
        self.xsp = np.zeros((n, ng), dtype=np.complex)
        # vector of amplitudes (after polishing)
        self.xpol = np.zeros((n, ng), dtype=np.complex)

    @property
    def nonzero(self):
        """where amplitudes are nonzero"""
        return self.xsp != 0


class SparseReconstruction(object):
    """Reconstruction of the input data based on a
    desired number of modes.

    Returns an object with the following attributes:

        r = dmd.make_sparse_reconstruction(nmodes=3)

        r.nmodes  # number of modes (3)
        r.data    # the reconstructed data
        r.modes   # the modes (3 of them)
        r.freqs   # corresponding complex frequencies
        r.amplitudes  # corresponding amplitudes
        r.ploss   # performance loss

    Returns error if the given number of modes cannot be found
    over the gamma we've looked at.

    TODO: think about a gamma search function?
    """
    def __init__(self, sparse_dmd, number_index, shape=None, decomp_axis=1):
        """
        sparse_dmd - a SparseDMD instance with the sparse solution computed

        number_index - the index that selects the desired number of
                       modes in sparse_dmd.sparse.Nz

        data - the original input data. If not supplied, the original
               snapshots will be reconstructed.
        """
        self.dmd = sparse_dmd.dmd
        self.sparse_dmd = sparse_dmd.sparse

        self.nmodes = self.sparse_dmd.Nz[number_index]
        self.Ni = number_index

        self.data_shape = shape
        self.decomp_axis = decomp_axis

        self.rdata = self.sparse_reconstruction()

        nonzero = self.sparse_dmd.nonzero[:, number_index]

        self.modes = self.dmd.modes[:, nonzero]
        self.freqs = self.dmd.Edmd[nonzero]
        self.amplitudes = self.sparse_dmd.xpol[nonzero, number_index]
        self.ploss = self.sparse_dmd.Ploss[number_index]

    def sparse_reconstruction(self):
        """Reconstruct the snapshots using a given number of modes.

        Ni is the index that gives the desired number of
        modes in `self.sparse.Nz`.

        shape is the shape of the original data. If None (default),
        the reconstructed snapshots will be returned; otherwise the
        snapshots will be reshaped to the original data dimensions,
        assuming that they were decomposed along axis `decomp_axis`.
        """
        amplitudes = np.diag(self.sparse_dmd.xpol[:, self.Ni])
        modes = self.dmd.modes
        time_series = self.dmd.Vand

        # we take the real part because for real data the modes are
        # in conjugate pairs and should cancel out. They don't
        # exactly because this is an optimal fit, not an exact
        # match.
        snapshot_reconstruction = np.dot(modes,
                                         np.dot(amplitudes,
                                                time_series)).real

        if self.data_shape is not None:
            data_reconstruction = to_data(snapshot_reconstruction,
                                          self.data_shape,
                                          self.decomp_axis)
            return data_reconstruction
        else:
            return snapshot_reconstruction


def subplot(plot_function):
    """Wrapper for functions that plot on a matplotlib axes instance
    that autocreates a figure and axes instance if ax is not
    supplied.
    """
    @functools.wraps(plot_function)
    def f(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            plot_function(self, ax, **kwargs)
            return fig
        else:
            plot_function(self, ax, **kwargs)
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


if __name__ == '__main__':
    import time
    channel_mat = 'matlab/codes/channel/channel.mat'

    mat_dict = scipy.io.loadmat(channel_mat)

    UstarX1 = mat_dict['UstarX1']
    S = mat_dict['S']
    V = mat_dict['V']

    # Sparsity-promoting parameter gamma
    # Lower and upper bounds relevant for this flow type
    gamma_grd = 200
    gammaval = np.logspace(np.log10(0.15), np.log10(160), gamma_grd)

    tic = time.time()
    Fdmd, Edmd, Ydmd, xdmd, py_answer = run_dmdsp(UstarX1, S, V, gammaval)
    toc = time.time()
    print "time elapsed: ", toc - tic
