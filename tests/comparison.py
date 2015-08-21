import time

import numpy as np
import scipy.io

from sparse_dmd import run_dmdsp

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
