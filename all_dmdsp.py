# The entire dmdsp method for the channel example
# translated line by line from matlab source

# Start of run_dmdsp

import numpy as np

import scipy.linalg as linalg
import scipy.io


def run_dmdsp():
    channel_mat = 'matlab/codes/channel/channel.mat'

    mat_dict = scipy.io.loadmat(channel_mat)

    UstarX1 = mat_dict['UstarX1']
    S = mat_dict['S']
    V = mat_dict['V']

    # Sparsity-promoting parameter gamma
    # Lower and upper bounds relevant for this flow type
    gamma_grd = 200
    gammaval = np.logspace(np.log10(0.15), np.log10(160), gamma_grd)

    Vstar = V.T.conj()

    # The number of snapshots
    N = Vstar.shape[1]

    # Optimal DMD matrix resulting from Schmid's 2010 algorithm
    # Fdmd = U'*X1*V*inv(S)
    Fdmd = np.dot(np.dot(UstarX1, V), linalg.inv(S))
    # Determine the rank of Fdmd
    r = np.diag(Fdmd).size

    # E-value decomposition of Fdmd
    Edmd, Ydmd = linalg.eig(Fdmd)
    Ddmd = np.diag(Edmd)

    # Form Vandermonde matrix
    # Vand = Edmd ** np.arange(N)[None].T
    Vand = np.vander(Edmd, N).T[::-1].T

    # Determine optimal vector of amplitudes xdmd
    # Objective: minimize the least-squares deviation between
    # the matrix of snapshots X0 and the linear combination of the dmd modes
    # Can be formulated as:
    # minimize || G - L*diag(xdmd)*R ||_F^2
    L = Ydmd
    R = Vand
    G = np.dot(S, Vstar)

    # Form matrix P, vector q, and scalar s
    # J = x'*P*x - q'*x - x'*q + s
    # x - optimization variable (i.e., the unknown vector of amplitudes)
    P = np.dot(L.T.conj(), L) * np.dot(R, R.T.conj()).conj()
    q = np.diagonal(np.dot(np.dot(R, G.T.conj()), L)).conj()
    s = np.trace(np.dot(G.T.conj(), G))

    # Cholesky factorization of P
    Pl = linalg.cholesky(P, lower=True)

    # Optimal vector of amplitudes xdmd
    xdmd = linalg.solve(Pl.T.conj(), linalg.solve(Pl, q))

    options = {'rho': 1,
               'maxiter': 10000,
               'eps_abs': 1E-6,
               'eps_rel': 1E-4}

    answer = dmdsp(P, q, s, gammaval, options)

    return answer


def dmdsp(P, q, s, gammaval, options=None):
    """Inputs:  matrix P
                vector q
                scalar s
                sparsity promoting parameter gamma

            (2) options

                options.rho     - augmented Lagrangian parameter rho
                options.maxiter - maximum number of ADMM iterations
                options.eps_abs - absolute tolerance
                options.eps_rel - relative tolerance

                If options argument is omitted, the default values are set to

                options.rho = 1
                options.maxiter = 10000
                options.eps_abs = 1.e-6
                options.eps_rel = 1.e-4
    """
    if not options:
        options = {'rho':     1,
                   'maxiter': 10000,
                   'eps_abs': 1E-6,
                   'eps_rel': 1E-4}

    # Data preprocessing
    rho = options['rho']
    Max_ADMM_Iter = options['maxiter']
    eps_abs = options['eps_abs']
    eps_rel = options['eps_rel']

    # Number of optimization variables
    n = len(q)
    # Identity matrix
    I = np.identity(n)

    # % Allocate memory for gamma-dependent output variables
    ng = len(gammaval)
    answer = {
        'gamma': gammaval,
        'Nz':    np.zeros((1, ng)),  # number of non-zero amplitudes
        'Jsp':   np.zeros((1, ng), dtype=np.complex),  # square of Frobenius norm (before polishing)
        'Jpol':  np.zeros((1, ng), dtype=np.complex),  # square of Frobenius norm (after polishing)
        'Ploss': np.zeros((1, ng), dtype=np.complex),  # optimal performance loss (after polishing)
        'xsp':   np.zeros((n, ng), dtype=np.complex),  # vector of amplitudes (before polishing)
        'xpol':  np.zeros((n, ng), dtype=np.complex),  # vector of amplitudes (after polishing)
    }

    # Cholesky factorization of matrix P + (rho/2)*I
    Prho = P + (rho / 2.) * I
    Plow = linalg.cholesky(Prho, lower=True)
    Plow_star = Plow.T.conj()

    for i, gamma in enumerate(gammaval):
        # Initial conditions
        y = np.zeros((n, 1))  # Lagrange multiplier
        z = np.zeros((n, 1))

        # Use ADMM to solve the gamma-parameterized problem
        for ADMMstep in range(Max_ADMM_Iter):
            # x-minimization step
            u = z - (1 / rho) * y
            # TODO: solve or lstsq?
            xnew = linalg.solve(Plow_star,
                                linalg.solve(Plow,
                                             q[:, None] + (rho / 2.) * u))

            # z-minimization step
            a = (gamma / rho) * np.ones((n, 1))
            v = xnew + (1 / rho) * y
            # Soft-thresholding of v
            znew = ((1 - a / abs(v)) * v) * (abs(v) > a)

            # Primal and dual residuals
            res_prim = linalg.norm(xnew - znew)
            res_dual = rho * linalg.norm(znew - z)

            # Lagrange multiplier update step
            y = y + rho * (xnew - znew)

            # Stopping criteria
            eps_prim = np.sqrt(n) * eps_abs + eps_rel * max([linalg.norm(xnew),
                                                             linalg.norm(znew)])
            eps_dual = np.sqrt(n) * eps_abs + eps_rel * linalg.norm(y)

            if (res_prim < eps_prim) & (res_dual < eps_dual):
                break
            else:
                z = znew

        # Record output data
        answer['xsp'][:, i] = z.squeeze()  # vector of amplitudes
        answer['Nz'][:, i] = (z != 0).sum()  # number of non-zero amplitudes
        # Frobenius norm (before polishing)
        answer['Jsp'][:, i] = np.dot(np.dot(z.T.conj(), P), z).real \
                              - 2 * np.dot(q.T.conj(), z).real \
                              + s

        # Polishing of the nonzero amplitudes
        # Form the constraint matrix E for E^T x = 0
        ind_zero = np.where(abs(z.squeeze()) < 1E-12)  # indices of zero elements of z
        m = len(ind_zero[0])  # number of zero elements
        E = I[:, ind_zero].squeeze()
        # TODO: how do we do sparse?
        # why do we even bother??
        # import scipy.sparse
        # E = scipy.sparse.coo_matrix(E)

        # Form KKT system for the optimality conditions
        KKT = np.vstack((np.hstack((P, E)),
                         np.hstack((E.T.conj(), np.zeros((m, m))))
                         ))
        rhs = np.vstack((q[:, None], np.zeros((m, 1))))

        # Solve KKT system
        sol = linalg.solve(KKT, rhs)

        # Vector of polished (optimal) amplitudes
        xpol = sol[:n]

        # Record output data
        answer['xpol'][:, i] = xpol.squeeze()
        # Polished (optimal) least-squares residual
        answer['Jpol'][:, i] = np.dot(np.dot(xpol.T.conj(), P), xpol).real \
                               - 2 * np.dot(q.T.conj(), xpol).real \
                               + s
        # Polished (optimal) performance loss
        answer['Ploss'][:, i] = 100 * np.sqrt(answer['Jpol'][:, i] / s)

    return answer
