"""Create a pathway between our lab data (in hdf5),
and .mat file containing UstarX1, S, V, dT"""

import numpy as np
import scipy.linalg as lin
import scipy.io
import scipy.ndimage as ndi

import gc_turbulence as g

run = g.ProcessedRun(g.default_processed + 'r13_12_16a.hdf5')


def to_snaps(data):
    """Create the matrix of snapshots by flattening the non decomp
    axes so we have a 2d array where we index the decomp axis like
    snapshots[:,i]

    i.e. for M data points per snapshot and N snapshots, the
    snapshot matrix is shape (M, N)

    The decomposition axis is the x dimension of the front relative
    data.
    """
    iz, ix, it = data.shape
    snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))
    return snapshots


def find_nan_slice(data):
    """Find the slice that contains nans in the data, assuming that
    they are contiguous.
    """
    return ndi.find_objects(np.isnan(data[...]))


def complement(nan_slices):
    """Compute the slice that complements a slice that contains
    nans, i.e. return the slice that will not include any nans.
    """
    sz, sx, st = nan_slices[0]
    return (slice(None), slice(None), slice(st.stop, None))


# create the matrix of snapshots
# nb. array, not matrix
X = to_snaps(run.Uf_[complement(find_nan_slice(run.Uf_[:]))])

X0 = X[:, :-1]
X1 = X[:, 1:]

# economy size SVD of X0
U, S, V = lin.svd(X0, full_matrices=False)
## n.b. differences with matlab svd:
## 1. S is a 1d array of diagonal elements
## 2.

# rank of S (np.rank doesn't get it... the rank is the number of
# independent vectors, which has to be the length of a diagonal
# matrix) should be equal to X.shape[0]
r = S.size
# now actually form the diagonal matrix
S = np.diag(S)

# Truncated versions of U, S, and V
# TODO is this actually necessary? is S not already rxr?
# maybe necessary if S isn't nonzero all the way along the diagonal
# U = U[:, :r]
# S = S[:r, :r]
# V = V[:, :r]

# Determine matrix UstarX1
UstarX1 = np.dot(U.T.conj(), X1)   # conjugate transpose of U  (U' in matlab)

data = {'UstarX1': UstarX1,
        'S':       S,
        'V':       V,
        'dT':      0.015}  # TODO: get this from run

scipy.io.savemat('data.mat', data)
