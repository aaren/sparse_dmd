import numpy as np
import scipy.linalg as linalg

from .dmd import DMD
from .util import to_data


class SparseDMD(object):
    def __init__(self, snapshots=None, dmd=None, axis=-1, dt=1,
                 rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):
        # TODO: allow data, axis as an argument instead of snapshots
        """Sparse Dynamic Mode Decomposition, using ADMM to find a
        sparse set of optimal dynamic mode amplitudes

        Inputs:
            snapshots - the matrix of data snapshots, shape (d, N)
                        where N is the number of snapshots and d is
                        the number of data points in a snapshot.

                        Alternately, multi-dimensional data can be
                        given here and it will be reshaped into the
                        snapshot matrix along the given `axis`.

            dmd     - optional precomputed DMD instance

            axis    - decomposition axis, default -1

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
            self.dmd = DMD(snapshots, axis=axis, dt=dt)
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
            # ## x-minimization step (alpha minimisation)
            u = z - (1. / self.rho) * y
            qs = q + (self.rho / 2.) * u
            # Solve P x = qs, using fact that P is hermitian and
            # positive definite and assuming P is well behaved (no
            # inf or nan).
            xnew = potrs(C, qs, lower=False, overwrite_b=False)[0]
            # ##

            # ## z-minimization step (beta minimisation)
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
            # ##

            # ## Lagrange multiplier update step
            y = y + self.rho * (xnew - znew)
            # ##

            # ## Test convergence of admm
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
        ind_zero = abs(z) < 1E-12

        # number of zero elements
        m = ind_zero.sum()

        # Polishing of the nonzero amplitudes
        # Form the constraint matrix E for E^T x = 0
        E = np.identity(self.n)[:, ind_zero]
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

    def reconstruction(self, Ni):
        """Compute a reconstruction of the input data based on a sparse
        selection of modes.

        Ni - the index that selects the desired number of
             modes in self.sparse.Nz

        shape - the shape of the original input data. If not supplied,
                the original snapshots will be reconstructed.

        Returns a SparseReconstruction with the following attributes:

        r.nmodes  # number of modes (3)
        r.data    # the original data
        r.rdata   # the reconstructed data (or snapshots)
        r.modes   # the modes (3 of them)
        r.freqs   # corresponding complex frequencies
        r.amplitudes  # corresponding amplitudes
        r.ploss   # performance loss
        """
        return SparseReconstruction(self,
                                    number_index=Ni,
                                    shape=self.dmd.data_shape,
                                    axis=self.dmd.axis)


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
    def __init__(self, sparse_dmd, number_index, shape=None, axis=-1):
        """
        sparse_dmd - a SparseDMD instance with the sparse solution computed

        number_index - the index that selects the desired number of
                       modes in sparse_dmd.sparse.Nz

        shape - the original input data shape. Used for reshaping the
                reconstructed snapshots.

        axis - the decomposition axis in the input data. Defaults to
               -1, i.e. will work with matrix of snapshots.
        """
        self.dmd = sparse_dmd.dmd
        self.sparse_dmd = sparse_dmd.sparse

        self.nmodes = self.sparse_dmd.Nz[number_index]
        self.Ni = number_index

        self.data_shape = shape
        self.axis = axis

        self.rmodes = self.sparse_reconstruction()

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
        assuming that they were decomposed along axis `axis`.
        """
        amplitudes = np.diag(self.sparse_dmd.xpol[:, self.Ni])
        modes = self.dmd.modes
        time_series = self.dmd.Vand

        # we take the real part because for real data the modes are
        # in conjugate pairs and should cancel out. They don't
        # exactly because this is an optimal fit, not an exact
        # match.
        reconstruction = np.dot(modes, np.dot(amplitudes, time_series))
        return reconstruction.real

    @property
    def rdata(self):
        """Convenience function to return reduced modes reshaped
        into original data shape.
        """
        if self.data_shape is not None:
            data_reconstruction = to_data(self.rmodes,
                                          self.data_shape,
                                          self.axis)
            return data_reconstruction

    @property
    def dmodes(self):
        """Convenience function to return modes reshaped into original
        data shape.
        """
        return to_data(snapshots=self.modes,
                       shape=self.data_shape,
                       axis=self.axis)
