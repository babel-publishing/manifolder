###
### load data
###

# from original Matlab header:
## Configuration
# mat file contains:
# - Dim: dimension to reduce pseudoinverse of covariance matrices to.
# - z: original time series with each time point as a column
# - x_interp: useful statistics to similarly compress for comparison
#           purposes

# note, not using x_interp in this code

data_location = '../../data/simple_data.csv'

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pandas as pd

print('loading data from' + data_location + '... ', end='')

# the .cvs version of the data was created directly from the .mat file with
#   >>> load solar_wind_data.mat
#   >>> csvwrite('solar_wind_data.csv',z)

# Dim would normally be loaded from the .mat file, just set it here
Dim = 3

# the easiest way to load data from csv is to use pandas, which brings in a dataframe
df = pd.read_csv(data_location, header=None)

# rather than use a dataframe, work on the data directly, as a numpy array
# keep the 'observations as columns' orientation of the original MATLAB code
# NOTE - Python code typically orients the matrices the other way ... careful!
# NOTE - indexing in this code is pythonic, starting at zero ... careful!

z = df.values

print('z.shape', z.shape)

# shift and scale the rows to [0,1]
# z = z - np.min(z, axis=1).reshape(-1, 1)
# z = z / np.max(z, axis=1).reshape(-1, 1)
#
# print('z.shape,', z.shape)
#
N = z.shape[0]     # will be 8, the number of features

print('done!')
