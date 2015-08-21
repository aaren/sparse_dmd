from .dmd import DMD
from .util import to_data, to_snaps
from .sparse import SparseDMD
from .plots import SparsePlots


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
