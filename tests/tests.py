import numpy.testing as nt

import scipy.io

import all_dmdsp


def test_compare_outputs():
    """Compare the python output with the matlab output.
    They should be identical."""
    py_answer = all_dmdsp.run_dmdsp()

    mat_answer = scipy.io.loadmat('tests/answer.mat')

    for k in py_answer:
        nt.assert_array_almost_equal(py_answer[k].squeeze(),
                                     mat_answer[k].squeeze())


def test_compare_inputs():
    """Compare UstarX1, V, S generated with my method with those
    calculated with a matlab method.

    Do this with my lab data as don't want to re generate original
    channel.mat data.

    Requires using h5read in matlab to parse my lab data.
    """
