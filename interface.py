"""Create a pathway between our lab data (in hdf5),
and .mat file containing UstarX1, S, V, dT"""

import numpy as np
import scipy.linalg as lin
import scipy.io

import gc_turbulence as g

r = g.ProcessedRun(g.default_processed + 'r13_12_16a.hdf5')


# create the matrix of snapshots
X = to_snaps(r.Uf_)  # nb. must be MATRIX, not array

X0 = X[:, :-1]
X1 = X[:, 1:]

# economy size SVD of X0
U, S, V = lin.svd(X0, full_matrices=False)

# rank of S (ends up as the number of modes)
# should be equal to X.shape[0]
r = np.rank(S)

# Truncated versions of U, S, and V
# TODO is this actually necessary? is S not already rxr?
U = U[:, :r]
S = S[:r, :r]
V = V[:, :r]

# Determine matrix UstarX1
UstarX1 = U.H * X1   # conjugate transpose of U  (U' in matlab)

data = {'UStarX1': UstarX1,
        'S':       S,
        'V':       V,
        'dT':      0.015}  # TODO: get this from run

scipy.io.savemat('data.mat', data)
