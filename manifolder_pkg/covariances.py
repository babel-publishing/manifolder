### covariances

print('computing local covariances ', end='')

# from scipy.linalg import svd
import manifolder_helper as mh
from numpy.linalg import inv

# import numpy

# Estimate local covariance matrices
# ***************************************************************@

## Configuration
# ncov = 10    # (previous value) size of neighborhood for covariance
ncov = 40     # size of neighborhood for covariance

## Covariance estimation

z_mean = np.zeros_like(z_hist)    # Store the mean histogram in each local neighborhood

# NOTE, original matlab call should have used N * nbins ... length(hist_bins) works fine in MATLAB,
# but in python hist_bins has one more element than nbins, since it defines the boundaries ...

# inv_c = zeros(N*length(hist_bins), N*length(hist_bins), length(z_hist)) # Store the inverse covariance matrix of histograms in each local neighborhood
inv_c = np.zeros((N * nbins, N * nbins, z_hist.shape[1]))

# precalculate the values over which i will range ...
# this is like 40 to 17485 (inclusive) in python
# 41 to 17488 in MATLAB ... (check?)
irange = range(ncov, z_hist.shape[1] - ncov - 1)

# instead of waitbar, print .......... to the screen during processing
waitbar_increments = int(irange[-1] / 10)

for i in irange:
    if i % waitbar_increments == 0:
        print('.', end='')
    # not sure of the final number boundary for the loop ...
    # win = z_hist(:, i-ncov:i+ncov-1)
    # TODO - Alex, is this the right range in MATLAB?
    win = z_hist[:, i - ncov:i + ncov]      # python, brackets do not include end, in MATLAB () includes end

    ###
    ### IMPORTANT - the input to the cov() call in MATLAB is TRANSPOSED compared to numpy
    ###    cov(win.T) <=> np.cov(win)
    ###
    #
    # # Python example
    # A = np.array([[0, 1 ,2],[3, 4, 5]])
    # print(A)
    # print(np.cov(A.T))
    #
    # % MATLAB example
    # >> A = [[0 1 2];[3 4 5]]
    # >> cov(A)
    #
    # TODO - lol, don't use 40x40, use a different number of bins, etc.
    c = np.cov(win)

    #     # Denoise via projection on "known" # of dimensions
    #    [U S V] = svd(c); # matlab
    # python SVD looks very similar to MATLAB:
    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    #    factors a such that a == U @ S @ Vh
    U, S, V = mh.svd_like_matlab(c)

    # inverse also works the same in Python as MATLAB ...
    # matlab:
    # >> X = [1 0 2; -1 5 0; 0 3 -9]
    # >> Y = inv(X)
    #
    #     0.8824   -0.1176    0.1961
    #     0.1765    0.1765    0.0392
    #     0.0588    0.0588   -0.0980
    #
    # Python:
    # X = np.array([[1, 0, 2],[-1, 5, 0],[0, 3, -9]])
    # Y = inv(X)
    #
    # [[ 0.8824 -0.1176  0.1961]
    #  [ 0.1765  0.1765  0.0392]
    #  [ 0.0588  0.0588 -0.098 ]]

    # inv_c(:,:,i) = U(:,1:Dim) * inv(S(1:Dim,1:Dim)) * V(:,1:Dim)'  # matlab
    inv_c[:, :, i] = U[:, :Dim] @ inv(S[:Dim, :Dim]) @ V[:, :Dim].T  # NICE!

    # z_mean(:, i) = mean(win, 2); # matlab
    z_mean[:, i] = np.mean(win, 1)

print(' done')
