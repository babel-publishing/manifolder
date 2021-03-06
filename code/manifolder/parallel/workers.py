#this file contains the worker functions needed for parallel processing. These must be
#in a separate file, otherwise it will throw errors on Windows and/or hang forever in
#Jupyter Notebook
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

from manifolder import helper as mh
from multiprocessing import Lock, shared_memory

def parallel_init(l):
    global lock #should be shared among worker processes
    lock = l

def dis(inv_c, subidx, dataref, data, M, j):
    tmp1 = inv_c[:, :, subidx[j]] @ dataref[j, :].T  # 40, in Python

    a2 = np.dot(dataref[j, :], tmp1)  # a2 is a scalar
    b2 = np.sum(data * (inv_c[:, :, subidx[j]] @ data.T).T, 1)
    ab = data @ tmp1  # only @ works here

    # this tiles the matrix ... repmat is like np.tile
    # Dis[:,j] = repmat[a2, M, 1] + b2 - 2*ab
    return (np.tile(a2, [M, 1])).flatten() + b2 - 2 * ab
    
#shared memory version of dis, must have Python >= 3.8 to use
def dis_shm(inv_c_shape, inv_c_type,
            subidx_shape, subidx_type,
            dataref_shape, dataref_type,
            data_shape, data_type, 
            dis_shape, dis_type, 
            M, count, start):
    #connect to shared memory and create numpy array objects backed by it
    shm_inv_c = shared_memory.SharedMemory(name='inv_c')
    shm_subidx = shared_memory.SharedMemory(name='subidx')
    shm_dataref = shared_memory.SharedMemory(name='dataref')
    shm_data = shared_memory.SharedMemory(name='data')
    shm_Dis = shared_memory.SharedMemory(name='dis')
    inv_c = np.ndarray(inv_c_shape, inv_c_type, buffer=shm_inv_c.buf)
    subidx = np.ndarray(subidx_shape, subidx_type, buffer=shm_subidx.buf)
    dataref = np.ndarray(dataref_shape, dataref_type, buffer=shm_dataref.buf)
    data = np.ndarray(data_shape, data_type, buffer=shm_data.buf)
    Dis = np.ndarray(dis_shape, dis_type, buffer=shm_Dis.buf)
    #connected to shared memory, perform calculation as usual
    for j in range(start, start + count):
        tmp1 = inv_c[:, :, subidx[j]] @ dataref[j, :].T  # 40, in Python

        a2 = np.dot(dataref[j, :], tmp1)  # a2 is a scalar
        b2 = np.sum(data * (inv_c[:, :, subidx[j]] @ data.T).T, 1)
        ab = data @ tmp1  # only @ works here

        # this tiles the matrix ... repmat is like np.tile
        # Dis[:,j] = repmat[a2, M, 1] + b2 - 2*ab
        ret = (np.tile(a2, [M, 1])).flatten() + b2 - 2 * ab

        lock.acquire()
        Dis[:, j] = ret
        lock.release()
    del inv_c
    del subidx
    del dataref
    del data
    del Dis
    #close handles to shared memory
    shm_inv_c.close()
    shm_subidx.close()
    shm_dataref.close()
    shm_data.close()
    shm_Dis.close()

def covars(z_hist_arr, ncov, nbins, N, Dim, snip):

    z_hist = z_hist_arr[snip]

    z_mean = np.zeros_like(z_hist)  # Store the mean histogram in each local neighborhood

    # NOTE, original matlab call should have used N * nbins ... length(hist_bins) works fine in MATLAB,
    # but in python hist_bins has one more element than nbins, since it defines the boundaries ...

    # inv_c = zeros(N*length(hist_bins), N*length(hist_bins), length(z_hist))
    # Store the inverse covariance matrix of histograms in each local neighborhood
    inv_c = np.zeros((N * nbins, N * nbins, z_hist.shape[1]))

    # precalculate the values over which i will range ...
    # this is like 40 to 17485 (inclusive) in python
    # 41 to 17488 in MATLAB ... (check?)
    irange = range(ncov, z_hist.shape[1] - ncov - 1)

    for i in irange:
        # not sure of the final number boundary for the loop ...
        # win = z_hist(:, i-ncov:i+ncov-1)
        # TODO - Alex, is this the right range in MATLAB?
        win = z_hist[:,
              i - ncov:i + ncov]  # python, brackets do not include end, in MATLAB () includes end

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

        #  De-noise via projection on "known" # of dimensions
        #    [U S V] = svd(c); # matlab
        # python SVD looks very similar to MATLAB:
        #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
        #    factors a such that a == U @ S @ Vh
        
        # Compute full svd
        # U, S, V = mh.svd_like_matlab(c)

        # Compute largest singular vectors only
        U, S, V = mh.svds_like_matlab(c, Dim)

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
        inv_c[:, :, i] = U[:, :Dim] @ pinv(S[:Dim, :Dim]) @ V[:, :Dim].T  # NICE!

        # z_mean(:, i) = mean(win, 2); # matlab
        z_mean[:, i] = np.mean(win, 1)

    return (z_mean, inv_c)

def histograms(self_z, H, stepSize, N, hist_bins, snip):
    ## Concatenate 1D histograms (marginals) of each sensor in short windows
    z_hist_list = []  # in Python, lists are sometimes easier than concatinate

    z = self_z[snip]

    #print('calculating histograms for snip ', snip, ' of ', n, ' (dim ', self.N ,' timeseries) ', end='')

    # for dim=1:N
    for dim in range(N):  # loop run standard Python indexing, starting at dim = 0
        series = z[dim, :]  # grab a row of data

        # NOTE, MATLAB and python calculate histograms differently
        # MATLAB uses nbins values, as bins centerpoints, and
        # Python uses nbins+1 values, to specify the bin endpoints

        # note, hist_bins will always be [0 .25 .5 .75 1], in MATLAB
        # equivalent for python hist is
        #   [-0.12   0.128  0.376  0.624  0.872  1.12 ]
        # hist_bins = mh.histogram_bins_centered(series, self.nbins)

        z_hist_dim_list = []

        # for i=1:floor((size(z,2)-H)/stepSize)
        i_range = int(np.floor(z.shape[1] - H) / stepSize)
        for i in range(i_range):
            # interval = z(dim, 1 + (i - 1) * stepSize: (i - 1) * stepSize + H);
            interval = series[i * stepSize:i * stepSize + H]

            # take the histogram here, and append it ... should be nbins values
            # first value returned by np.histogram the actual histogram
            #
            #  NOTE!!! these bins to not overlap completely with the MATLAB version,
            #   but are roughly correct ... probably exact boundaries are not the same,
            #   would need to look into this ...
            #
            hist = np.histogram(interval, hist_bins[dim])[0]
            z_hist_dim_list.append(hist)

        # convert from a list, to array [nbins x (series.size/stepSize?)]
        z_hist_dim = np.array(z_hist_dim_list).T

        # z_hist = [z_hist; z_hist_dim];
        z_hist_list.append(z_hist_dim)

    return z_hist_list
