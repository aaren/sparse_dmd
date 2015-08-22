import numpy as np


def to_snaps(data, axis=-1):
    """Transform data into snapshots, using axis as
    the decomposition axis.
    """
    sd = data.shape[axis]   # size of the decomp axis
    shape_idx = range(len(data.shape))
    # we reset the value of axis to allow indexing with -1
    axis = shape_idx.pop(axis)
    # put the decomp axis last and reshape to 2d array
    return data.transpose(shape_idx + [axis]).reshape((-1, sd))


def to_data(snapshots, shape, axis=-1):
    """Reshape snapshots or modes to match data that
    had a `shape` and that was decomposed along `axis`.
    """
    shape = list(shape)
    shape.pop(axis)  # remove decomp axis (it is implicitly last)
    reshaped = np.asarray(snapshots).reshape(shape + [-1])

    rshape = list(reshaped.shape)
    irshape = range(len(rshape))  # indices of array shape
    id = irshape.pop(-1)  # pull out index of decomp axis
    irshape.insert(axis, id)  # and insert where it came from
    return reshaped.transpose(irshape)
