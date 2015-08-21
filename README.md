*This tool is in development. The API may change! If you raise an issue I will most likely respond :)*

## Sparse DMD in Python

The Dynamic Mode Decomposition is a tool for analysing spatially
distributed time-series, motivated by seeking recurring patterns in
2D velocity data from experiments in fluids.

The DMD finds the best fit to the data with a number of 'dynamic'
modes, each having a distinct frequency of oscillation.

A drawback of the standard DMD is that the number of modes is the
same as the number of fields in the decomposition axis of the data
and that there is no clear way to select which of these modes best
represent the data.

The sparse DMD (Jovanovic et al, 2014) aims to find a reduced number
of dynamic modes that best represent the data.

This is a Python version of the reference [matlab source][matlab_source].

[matlab_source]: http://www.ece.umn.edu/users/mihailo//software/dmdsp/download.html


### Usage

Assuming that `u` is some 3d array containing 2d velocity data
through time, we use a convenience method to create the matrix of
snapshots and compute the standard DMD:

```python
import sparse_dmd

# load data
u = load_some_data()

# create snapshots, using last array axis for decomposition
snapshots = sparse_dmd.to_snaps(u, decomp_axis=-1)

# create and compute the standard dmd
dmd = sparse_dmd.DMD(snapshots)
dmd.compute()
```

We can access the dynamic modes, ritz values and (optimal)
amplitudes as attributes on this object:

```python
# the dynamic modes
dmd.modes
# the optimal amplitudes
dmd.amplitudes
# the ritz values
dmd.ritz_values
```

Now we can compute the sparse dmd, given some parameterisation
range:

```python
# initialise the sparse dmd (can also do directly from snapshots)
spdmd = sparse_dmd.SparseDMD(dmd=dmd)

# create range of sparsity parameterisation
gamma = np.logspace(-2, 6, 200)

# compute the sparse dmd using this gamma range
spdmd.compute_sparse(gamma)
```

You may have to tweak the range of `gamma` manually, until it nicely
covers your data.

You can now access the results of the sparsity computation:

```python
# polished optimal mode amplitudes
optimal_amplitudes = dmd.sparse.xpol

# number of non-zero amplitudes
dmd.sparse.Nz
```

#### Plotting

The plotting routines from the matlab source have been copied over
and can easily be performed:

```python
import matplotlib.pyplot as plt

plotter = sparse_dmd.SparsePlots(dmd)

fig = plotter.performance_loss_gamma()
plt.show()

fig, ax = plt.subplots()
plotter.nonzero_gamma(ax)
```

#### Reconstruction

Given a sparse computation and a desired number of modes we can
attempt to reconstruct the original data.

Currently you have to perform the computation with given gamma and
look the array of the number of nonzero amplitudes, `dmd.sparse.Nz`,
for the index into `gamma` corresponding to the number of modes that
you want (e.g. `Ni=30` here).

```python
# compute the reconstruction for gamma[30]
dmd.compute_sparse_reconstruction(Ni=30, shape=u.shape, decomp_axis=-1)

reconstructed_data = dmd.reconstruction.rdata

reduced_set_of_modes = dmd.reconstruction.modes
reduced_set_of_ritz_values = dmd.reconstruction.freqs
reduced_set_of_amplitudes = dmd.reconstruction.amplitudes
```


### Performance

Performance is on par with the matlab version when using accelerated
Anaconda.

On a 6 * 2.6GHz Opteron, 32GB machine, using the `channel.mat`
example data with a gamma parameterisation of `logspace(log10(0.15),
log10(160), 200)`, the original Matlab (R2012a) (with printing
supressed) takes 1.60s and this Python version takes 1.55s (using
accelerated Anaconda).


### Contributing

Very welcome, especially for performance! Just create an issue /
open a pull request.


### How did I translate this?

1. Take the matlab source and form a single `.m` file that
   contains the entire program in imperative form.

2. Go through this file line by line, writing the equivalent
   Python and checking that the intermediate results are
   the same in both Python and MATLAB.

3. Write a simple test, comparing the output of both matlab and
   python with the same input data.

4. Refactor the Python program, using the test to avoid mistakes.
