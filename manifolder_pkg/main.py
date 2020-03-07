# Python version of MATLAB code
#  for more information, see the README.md

### Original MATLAB Headar
# MATLAB code implementation of the toy example from:
# R. Talmon and R. R. Coifman, "Empirical intrinsic geometry
# for nonlinear modeling and time series filtering",
# in PNAS Vol. 110 No. 31 pp.12535?12540.
# ***************************************************************@
# This implementation generates the underlying diffusion processes
# and the corresponding measurements under the Poisson modality and
# recovers the underlying processes using the proposed EIG method.
# Author: Ronen Talmon.
# Created:  1/6/13.
# ***************************************************************@
# Altered for AVLab Time Series Data by Alex Cloninger, Srinjoy Das

## Python
# To make the code as similar to MATLAB as possible, run all the files inline.
# Files can be run individually, and probably broken into proper functions later.
#

import time
import numpy as np
import os

# print(os.getcwd())

start_time = time.time()

# always flush print outputs
import functools

print = functools.partial(print, flush=True)

###
### load data (python)
####
print('\n*** executing load_data.py')
print()
exec(open('manifolder_pkg/load_data.py').read())

### Modeling

###
### compute_histograms (python)
###
print('\n*** executing histograms_overlap.py')
exec(open('manifolder_pkg/histograms_overlap.py').read())

###
### covariances
###
print('\n*** executing covariances.py')
exec(open('manifolder_pkg/covariances.py').read())

###
### embedding
###
print('\n*** executing embedding.py')
exec(open('manifolder_pkg/embedding.py').read())

###
### clustering
###
print('*** executing clustering.py')
exec(open('manifolder_pkg/clustering.py').read())

###
### end
###
elapsed_time = time.time() - start_time
print('\n\t Program Executed in', str(np.round(elapsed_time, 2)), 'seconds')   # about 215 seconds (four minutes)

# --------
