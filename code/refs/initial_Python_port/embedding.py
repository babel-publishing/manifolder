# Compute the embedding
# ***************************************************************@

import numpy as np

from manifolder_helper import eigs_like_matlab

###
### Part I
###

## Configuration
m = 4000           # starting point for sequantial processing/extension
data = z_mean.T    # set the means as the input set
M = data.shape[0]

# Choose subset of examples as reference
# this is 'take m (4000) random values from z_mean, and sort them
# subidx = sort(randperm(size(z_mean, 2), m))
# Choose first m examples as reference (commented out, don't do this
# subidx = 1:m;
subidx = np.arange(z_mean.shape[1])
np.random.shuffle(subidx)    # shuffle is inplace in python
subidx = subidx[:m]          # take a portion of the data
subidx.sort()                # sort is also in place ...

# dataref = data(subidx,:)
dataref = data[subidx, :]

##
# Affinity matrix computation

print('computing Dis matrix ', end='', flush=True)

waitbar_increments = m // 10
Dis = np.zeros((M, m))

for j in range(m):
    if j % waitbar_increments == 0:
        print('.', end='')
    # waitbar(j / m, h) # printing in stead of waitbar

    # tmp1 = inv_c(:,:,subidx(j)) * dataref(j,:)'  # is 40 x 1 in MATLAB
    tmp1 = inv_c[:, :, subidx[j]] @ dataref[j, :].T   # 40, in Python

    a2 = np.dot(dataref[j, :], tmp1)   # a2 is a scalar
    b2 = np.sum(data * (inv_c[:, :, subidx[j]] @ data.T).T, 1)
    ab = data @ tmp1                   # only @ works here

    # this tiles the matrix ... repmat is like np.tile
    # Dis[:,j] = repmat[a2, M, 1] + b2 - 2*ab
    Dis[:, j] = (np.tile(a2, [M, 1])).flatten() + b2 - 2*ab

print('done!')

## Anisotropic kernel

print('aniostropic kernel ... ', end='')

ep = np.median(np.median(Dis, 0))      # default scale - should be adjusted for each new realizations

A = np.exp(-Dis / (4*ep))    # is numpy okay with exponential of matrices?  okay, calculates them individually
W_sml = A.T @ A
d1 = np.sum(W_sml, 0)
A1 = A / np.tile(np.sqrt(d1), [M, 1])
W1 = A1.T @ A1

d2 = np.sum(W1, 0)
A2 = A1 / np.tile(np.sqrt(d2), [M, 1])
W2 = A2.T @ A2

D = np.diag(np.sqrt(1 / d2))

###
### Part II
###

# Compute eigenvectors

# in numpy,
# from numpy import linalg as LA
# w, v = LA.eig(np.diag((1, 2, 3)))
#  v are the values, diagonal in a matrix, and w are the eigenvectors

# [V, E] = eigs(W2, 10) Matlab
V, E = eigs_like_matlab(W2, 10)   # think this is correct now ...

#print('V.shape', V.shape)
#print('E.shape', E.shape)

# python np.sum(A,0) <=> matlab sum(A)
# in matlab, srted are the values of sum(E) sorted (in descending order)
# and IE are the indices that sorted them
# [srtdE, IE] = sort(sum(E), 'descend')

# this is python eqivalent ... note that IE will have values one less than the MATLAB, because zero indexing
# TODO - is this sorted right?
IE = np.sum(E, 0).argsort()[::-1]      # find the indices to sort, and reverse them
srtdE = np.sum(E, 0)[IE]

# Phi = D @ V(:, IE(1, 2:10))
Phi = D @ V[:, IE[1:]]

print('done')

###
### Part III
###

# TODO - not necessary?  (Independent coordinates?)

# Extend reference embedding to the entire set
print('extending embedding (building Psi) ... ', end='', flush=True)

Psi_list = []      # holds all the psi_i values

omega = np.sum(A2, 1)
A2_nrm = A2 / np.tile(omega.reshape([-1, 1]), [1, m])      # omega needed to be shaped as a column

# for i=1:size(Phi,2)
for i in range(Phi.shape[1]):
    # this line is strange ... order of operations for @?, what is the offset?
    psi_i = A2_nrm @ Phi[:, i] / np.sqrt((srtdE[i + 1]))
    # [Psi, psi_i]
    Psi_list.append(psi_i)

# convert Psi_list back into an array, shaped like MATLAB version
Psi = np.array(Psi_list).T

# psi have have very small imaginary values ...
# cast to real here, but need to check
Psi = np.real(Psi)

# print('Psi.shape', Psi.shape)

print('done')

# Since close to a degenerate case - try to rotate according to:
# A. Singer and R. R. Coifman, "Spectral ICA", ACHA 2007.
#
