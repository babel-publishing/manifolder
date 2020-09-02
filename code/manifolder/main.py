__all__ = (
    'Manifolder',
)

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

from sklearn.cluster import KMeans

from manifolder import helper as mh

from manifolder.parallel import workers
from multiprocessing import Pool, TimeoutError, Lock#, Process, Manager

import functools
from functools import partial

import time
import sys
import os
import math

#import tslearn
#from tslearn.metrics import dtw
#from tslearn.metrics import cdist_dtw

import sklearn_extra
from sklearn_extra.cluster import KMedoids

import dtw

#from pyclustering.utils import calculate_distance_matrix
from pyclustering.cluster.kmedoids import kmedoids

import random
from random import sample

print = functools.partial(print, flush=True)


def test():
    print('test function called')


# class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
class Manifolder():
    """
    Implementation of Emperical Intrinsic Geometry (EIG) for time-series.

    Parameters
    ----------
    dim : int, optional, default 3
        The dimension of the underlying manifold.
        This will typically be somewhat smaller than the dimension of the data

    H: int, optional, default 40
        Non-overlapping window length for histogram/empirical densities estimation

    step_size: int, optional, default 5
        Stride between histograms

    nbins: int, optional, default 5
        Number of bins to use when creating histogram

    See Also
    --------

    Notes
    -----

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> manifolder = Manifolder().fit(data)
    >>> clusters() = manifolder.clusters()
    """

    def __init__(self, dim=3, H=40, step_size=5, nbins=5, distance_measure=None, ncov=40, n_jobs=None):

        self.Dim = dim
        self.H = H
        self.stepSize = step_size
        self.nbins = nbins

        self.distance_measure = distance_measure

        self.ncov = ncov

    def fit_transform(self, X, parallel=False, use_dtw=False, dtw_downsample_factor=1, dtw_stack=False, dtw_stack_dims=None):
        """
        Fit (find the underlying manifold).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self.
        """

        ### IMPORTANT - sklearn, and Python / data science in general use the convention where
        ###
        ###  data = [samples, features]
        ###
        ### manifolder takes the data in this semi-standard format, but internally uses the
        ### 'observations as columns' format from the original MATLAB
        ###
        # print('fit was called, not yet implemented')
        self._load_data(X)

        if parallel:
            l = Lock()
            pool = Pool(initializer=workers.parallel_init, initargs=(l,))#, maxtasksperchild=1)
            self._histograms_parallel(process_pool=pool)
            if use_dtw:
                self.dtw_matrix_parallel(self.get_snippets(downsample_factor=dtw_downsample_factor, 
                    stack=dtw_stack, stack_dimensions=dtw_stack_dims), process_pool=pool)
            else:
                self._covariances_parallel(process_pool=pool)
                self._embedding_parallel(process_pool=pool)
            pool.close()
            pool.join()
            if use_dtw:
                return
        else:
            self._histograms_overlap()
            if use_dtw:
                self.dtw_matrix(self.get_snippets(downsample_factor=dtw_downsample_factor, 
                    stack=dtw_stack, stack_dimensions=dtw_stack_dims))
                return
            else:
                self._covariances()
                self._embedding()

        return self.Psi  # the final clustering is in Psi
        # self._clustering()

        # sklearn fit() tends to return self
        return self

    def _load_data(self, data):
        """ loads the data, in [samples, nfeatures]
            NOTE - internally, data is stored in the
            format used in the original code """
        if not isinstance(data, list):
            self.z = [data.T]  # creates a list of length 1
        else:
            n = len(data)
            for snip in range(n):
                if snip == 0:
                    self.z = [data[snip].T]
                else:
                    self.z.append(data[snip].T)  # time is a function of the columns, internally

        self.N = self.z[0].shape[0]  # will be 8, the number of features

    #Returns a numpy array of all windows for dtw
    def get_windows(self, downsample_factor=1):
        i_range = int(np.floor(self.z[0].shape[1] - self.H) / self.stepSize)
        n = len(self.z) * int(np.floor(self.z[0].shape[1] - self.H) / self.stepSize)
        windows = np.zeros((n, self.H//downsample_factor))
        for snip in range(len(self.z)):
            z = self.z[snip]
            #currently only using one dimension, can add dimensions through stacking
            #for dim in range(self.N):
            #    series = z[dim, :] grab a row of data
            series = z[0, :]
            for i in range(i_range):
                windows[snip*len(self.z)+i, :] = self.downsample(series[i * self.stepSize:i * self.stepSize + self.H], downsample_factor)
        return windows


    #returns a 2d numpy array of all snippets. If stack is left false, only the first 
    # dimension of the data will be used. If true, it will stack the dimensions
    # specified in the iterable stack_dimensions, or all dimensions if stack_dimensions
    # is left as None, by interleaving the data points from each dimension
    def get_snippets(self, downsample_factor=1, stack=False, stack_dimensions=None):
        data_len = self.z[0].shape[1]
        if not stack:
            num_dims = 1
            stack_dimensions = (0,)
        if stack_dimensions == None:
            num_dims = self.z[0].shape[0]
            stack_dimensions = range(num_dims)
        else:
            num_dims = len(stack_dimensions)
        all_snippets = np.zeros((len(self.z), (data_len // downsample_factor) * num_dims))
        if stack:
            print("stacking " + str(num_dims) + " dimensions")
            for snip in range(len(self.z)):
                z = self.z[snip]
                dims = np.zeros((num_dims, data_len // downsample_factor))
                for d in range(num_dims):
                    dims[d,:] = self.downsample(z[d,:], downsample_factor)[:]
                all_snippets[snip,:] = self.stack(dims)[:]
            print("done stacking")
        else:
            for snip in range(len(self.z)):
                z = self.z[snip]
                all_snippets[snip,:] = self.downsample(z[0, :], downsample_factor)
        print(all_snippets.shape)
        return all_snippets


    def downsample(self, x, skip):
        if isinstance(x, list):
            length = len(x)
        elif isinstance(x, np.ndarray):
            length = x.shape[0]
        y = np.zeros(length//skip)
        j=0
        for i in range(0, length, skip):
            y[j] = x[i]
            j += 1
        return y

    def stack(self, dims):
        #transposing results in the data points from each dimension being interwoven.
        # to connect each dimension end to end, simply remove the call to np.transpose
        return np.transpose(dims).flatten()

    def dtw_matrix(self, data):
        start_time = time.time()
        self.dtw_matrix = np.zeros((data.shape[0], data.shape[0]))
        print(self.dtw_matrix.shape)
        start_time = time.time()
        for i in range(data.shape[0]):
            for j in range(i):
                dtw_result = dtw.dtw(data[i,:], data[j,:])#, window_type="sakoechiba", window_args={"window_size":2})
                self.dtw_matrix[i,j] = dtw_result.distance
                self.dtw_matrix[j,i] = dtw_result.distance
        elapsed_time = time.time() - start_time
        print('DTW done in ', str(np.round(elapsed_time, 2)), 'seconds!')
        print(self.dtw_matrix)
        return self.dtw_matrix

    #data must be passed as numpy array of snippets or windows
    def dtw_matrix_parallel(self, data, process_pool=None):
        if not (sys.version_info >= (3,8,0)):
            print('Python version is < 3.8, cannot use shared memory. Aborting')
            assert False

        print('computing dtw matrix in parallel')
        start_time = time.time()
        pool = process_pool
        if process_pool == None:
            l = Lock()
            pool = Pool(initializer=workers.parallel_init, initargs=(l,))
        
        #process_list = []
        #m = Manager()
        #lock = m.Lock()
        self.dtw_distmat = np.zeros((data.shape[0], data.shape[0]))
        
        print('Python version is >= 3.8, using shared memory')
        from multiprocessing import shared_memory
        #create shared memory for numpy arrays
        #print(data.nbytes)
        #print(self.dtw_distmat.nbytes)
        shm_data = shared_memory.SharedMemory(name='dtw_data',
                                              create=True, size=data.nbytes)
        shm_result = shared_memory.SharedMemory(name='dtw_result', 
                                                create=True, size=self.dtw_distmat.nbytes)
        #copy arrays into shared memory
        data_copy = np.ndarray(data.shape, data.dtype, buffer=shm_data.buf)
        np.copyto(data_copy, data, casting='no')
        self.dtw_distmat = np.zeros((data.shape[0], data.shape[0]))
        result = np.ndarray(self.dtw_distmat.shape, self.dtw_distmat.dtype, buffer=shm_result.buf)
        #use pool to run function in parallel
        func = partial(workers.dtw_shm, data_copy.shape, data_copy.dtype, 
                      result.shape, result.dtype)

        #build starmap iterable
        cpu_count = os.cpu_count()
        n = self.dtw_distmat.shape[0]
        each = n*n/float(os.cpu_count())
        arr = [(0, int(math.sqrt(each)))]
        for i in range(2, os.cpu_count()):
            arr.append((arr[-1][1], int(math.sqrt(each*i))))
        arr.append((arr[-1][1], n))
        #run function in parallel
        #for args in arr:
        #    p = Process(target=func, args=args)
        #    process_list.append(p)
        #    print("starting process from ", args[0], " to ", args[1])
        #    p.start()
        #for p in process_list:
        #    print("joining process")
        #    p.join()
        pool.starmap(func, arr)
        print("done processing dtw")
        if process_pool == None:
            pool.close()
            pool.join()
        #copy results out of shared memory
        np.copyto(self.dtw_distmat, result, casting='no')
        del data_copy
        del result

        #close and cleanup shared memory
        shm_data.close()
        shm_data.unlink()
        shm_result.close()
        shm_result.unlink()
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')
        return self.dtw_distmat


    def dtw_call(self, x, y):
        #here is where you can change dtw params for KMedoids clustering
        dtw_result = dtw.dtw(x,y)#, window_type="sakoechiba", window_args={"window_size":2})
        return dtw_result.distance

    def dtw_clustering_skl(self, num_clusters=7):
        all_windows = []
        all_snippets = []
        for snip in range(10):#len(self.z)):
            z = self.z[snip]
            #for dim in range(self.N):
            series = z[1, :]  # z[dim, :] grab a row of data
            all_snippets.append(series)
            i_range = int(np.floor(z.shape[1] - self.H) / self.stepSize)
            for i in range(i_range):
                #you can add in the downsampling here
                all_windows.append(self.downsample(series[i * self.stepSize:i * self.stepSize + self.H], 10))
        
        func = self.dtw_call

        print("dtw clustering on windows ... ", end="")
        start_time = time.time()
        kmedoids = KMedoids(n_clusters = num_clusters, metric=func, init="random").fit(all_windows)
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')
        print("dtw cluster centers:")
        print(kmedoids.cluster_centers_)
        print("dtw cluster labels:")
        print(kmedoids.labels_)
        self.kmedoids_windows = kmedoids
        
        
        print("dtw clustering on full snippets ... ", end="")
        start_time = time.time()
        kmedoids = KMedoids(n_clusters = num_clusters, metric=func, init="random").fit(all_snippets)
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')
        print("dtw cluster centers:")
        print(kmedoids.cluster_centers_)
        print("dtw cluster labels:")
        print(kmedoids.labels_)
        self.kmedoids_snippets = kmedoids
        return kmedoids



    def _histograms_overlap(self):

        n = len(self.z)

        hist_bins = mh.histogram_bins_all_snips(self.z, self.nbins)

        # JD
        # z_hist = []  # will build up  list of histograms, one for per snippet
        # for z in self.z:
        # z is a single snippet here, and self.z is the full list of all snippets

        for snip in range(n):
            ## Concatenate 1D histograms (marginals) of each sensor in short windows
            z_hist_list = []  # in Python, lists are sometimes easier than concatinate

            z = self.z[snip]

            print('calculating histograms for snip ', snip, ' of ', n, ' (dim ', self.N ,' timeseries) ', end='')

            # for dim=1:N
            for dim in range(self.N):  # loop run standard Python indexing, starting at dim = 0
                print('.', end='')
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
                i_range = int(np.floor(z.shape[1] - self.H) / self.stepSize)
                for i in range(i_range):
                    # interval = z(dim, 1 + (i - 1) * stepSize: (i - 1) * stepSize + H);
                    interval = series[i * self.stepSize:i * self.stepSize + self.H]

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

            # JD
            # z_hist.append(np.concatinate(z_hist_list))

            # convert from list back to numpy array
            if snip == 0:
                self.z_hist = [np.concatenate(z_hist_list)]
                self.snip_number = snip*np.ones(self.z_hist[snip].shape[1])
            else:
                self.z_hist.append(np.concatenate(z_hist_list))
                self.snip_number = np.append(self.snip_number,snip*np.ones(self.z_hist[snip].shape[1]))

            print(' done')  # prints 'done' after each snip

    def _histograms_parallel(self, process_pool=None):

        n = len(self.z)

        hist_bins = mh.histogram_bins_all_snips(self.z, self.nbins)

        # JD
        # z_hist = []  # will build up  list of histograms, one for per snippet
        # for z in self.z:
        # z is a single snippet here, and self.z is the full list of all snippets
        print("Calculating histograms in parallel ... ", end='')
        start_time = time.time()
        histfunc = partial(workers.histograms, self.z, self.H, self.stepSize, self.N, hist_bins)
        pool = process_pool
        if process_pool == None:
            l = Lock()
            pool = Pool(initializer=workers.parallel_init, initargs=(l,))
        results = pool.map(histfunc, range(n), chunksize=5)
        if process_pool == None:
            pool.close()
            pool.join()
        for snip in range(n):
            # convert from list back to numpy array
            if snip == 0:
                self.z_hist = [np.concatenate(results[snip])]
                self.snip_number = snip*np.ones(self.z_hist[snip].shape[1])
            else:
                self.z_hist.append(np.concatenate(results[snip]))
                self.snip_number = np.append(self.snip_number,snip*np.ones(self.z_hist[snip].shape[1]))
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')

    def _covariances(self):
        #
        #
        ## Configuration
        # ncov = 10    # (previous value) size of neighborhood for covariance
        # ncov = 40  # size of neighborhood for covariance
        # ncov is passed in, above

        n = len(self.z_hist)

        for snip in range(n):
            print('computing local covariances for snip ', snip, ' of ', n, end='')

            z_hist = self.z_hist[snip]

            z_mean = np.zeros_like(z_hist)  # Store the mean histogram in each local neighborhood

            # NOTE, original matlab call should have used N * nbins ... length(hist_bins) works fine in MATLAB,
            # but in python hist_bins has one more element than nbins, since it defines the boundaries ...

            # inv_c = zeros(N*length(hist_bins), N*length(hist_bins), length(z_hist))
            # Store the inverse covariance matrix of histograms in each local neighborhood
            inv_c = np.zeros((self.N * self.nbins, self.N * self.nbins, z_hist.shape[1]))

            # precalculate the values over which i will range ...
            # this is like 40 to 17485 (inclusive) in python
            # 41 to 17488 in MATLAB ... (check?)
            irange = range(self.ncov, z_hist.shape[1] - self.ncov - 1)

            # instead of waitbar, print .......... to the screen during processing
            waitbar_increments = int(irange[-1] / 10)

            for i in irange:
                if i % waitbar_increments == 0:
                    print('.', end='')
                # not sure of the final number boundary for the loop ...
                # win = z_hist(:, i-ncov:i+ncov-1)
                # TODO - Alex, is this the right range in MATLAB?
                win = z_hist[:,
                      i - self.ncov:i + self.ncov]  # python, brackets do not include end, in MATLAB () includes end

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
                U, S, V = mh.svds_like_matlab(c, self.Dim)

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
                inv_c[:, :, i] = U[:, :self.Dim] @ pinv(S[:self.Dim, :self.Dim]) @ V[:, :self.Dim].T  # NICE!

                # z_mean(:, i) = mean(win, 2); # matlab
                z_mean[:, i] = np.mean(win, 1)

            # append z_mean and inv_c as next rows of mat
            if snip == 0:
                self.z_mean = z_mean
                self.inv_c = inv_c
            else:
                self.z_mean = np.append(self.z_mean, z_mean, axis=1)
                self.inv_c = np.append(self.inv_c, inv_c, axis=2)

            print(' done')  # prints done at the end of each snip


    def _covariances_parallel(self, process_pool=None):
        #
        #
        ## Configuration
        # ncov = 10    # (previous value) size of neighborhood for covariance
        # ncov = 40  # size of neighborhood for covariance
        # ncov is passed in, above

        n = len(self.z_hist)
        print("Computing local covariances in parallel ... ", end='')
        start_time = time.time()
        covfunc = partial(workers.covars, self.z_hist, self.ncov, self.nbins, self.N, self.Dim)
        pool = process_pool
        if process_pool == None:
            l = Lock()
            pool = Pool(initializer=workers.parallel_init, initargs=(l,))
        results = pool.map(covfunc, range(n), chunksize=5)
        if process_pool == None:
            pool.close()
            pool.join()
        for snip in range(n): 
            z_mean, inv_c = results[snip]
            # append z_mean and inv_c as next rows of mat
            if snip == 0:
                self.z_mean = z_mean
                self.inv_c = inv_c
            else:
                self.z_mean = np.append(self.z_mean, z_mean, axis=1)
                self.inv_c = np.append(self.inv_c, inv_c, axis=2)
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')

    def _embedding(self):
        ###
        ### Part I
        ###

        ## Configuration

        # the variable m defines some subset of the data, to make computation faster;
        # this could be various values (10% of the data, all the data, etc.), as long
        # as it is not GREATER than the length of data.
        #   For the smallest change, setting to min 4000 or the data size

        # m = 4000                  # starting point for sequential processing/extension
        #
        # TODO - m allows you to sample various sections of the manifold, ratheer than looking at
        # all points to all points
        # the random points can come from the different chunks as well?
        #   ... for ease of coding, the datastructure could be back to 2D data
        m = np.min((4000, self.z_mean.shape[1]))
        print('using', m, 'for variable m')

        data = self.z_mean.T  # set the means as the input set
        M = data.shape[0]

        # Choose subset of examples as reference
        # this is 'take m (4000) random values from z_mean, and sort them
        # subidx = sort(randperm(size(z_mean, 2), m))
        # Choose first m examples as reference (commented out, don't do this
        # subidx = 1:m;
        subidx = np.arange(self.z_mean.shape[1])
        np.random.shuffle(subidx)  # shuffle is inplace in python
        subidx = subidx[:m]  # take a portion of the data
        subidx.sort()  # sort is also in place ...

        # dataref = data(subidx,:)
        dataref = data[subidx, :]

        ##
        # Affinity matrix computation

        print('computing Dis matrix ', end='')

        waitbar_increments = m // 10
        Dis = np.zeros((M, m))

        for j in range(m):
            if j % waitbar_increments == 0:
                print('.', end='')

            tmp1 = self.inv_c[:, :, subidx[j]] @ dataref[j, :].T  # 40, in Python

            a2 = np.dot(dataref[j, :], tmp1)  # a2 is a scalar
            b2 = np.sum(data * (self.inv_c[:, :, subidx[j]] @ data.T).T, 1)
            ab = data @ tmp1  # only @ works here

            # this tiles the matrix ... repmat is like np.tile
            # Dis[:,j] = repmat[a2, M, 1] + b2 - 2*ab
            Dis[:, j] = (np.tile(a2, [M, 1])).flatten() + b2 - 2 * ab

        print('done!')

        ## Anisotropic kernel

        print('aniostropic kernel ... ', end='')

        ep = np.median(np.median(Dis, 0))  # default scale - should be adjusted for each new realizations

        A = np.exp(-Dis / (4 * ep))
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

        # Compute all eigenvectors and select 10
        # [V, E] = eigs(W2, 10) Matlab
        # V, E = mh.eig_like_matlab(W2, 10)  # think this is correct now ...

        # Compute only 10 eigenvectors, must have symmetric matrix
        V, E = mh.eigs_like_matlab(W2,10)

        # print('V.shape', V.shape)
        # print('E.shape', E.shape)

        # python np.sum(A,0) <=> matlab sum(A)
        # in matlab, srted are the values of sum(E) sorted (in descending order)
        # and IE are the indices that sorted them
        # [srtdE, IE] = sort(sum(E), 'descend')

        # this is python eqivalent ... note that IE will have values one less than the MATLAB, because zero indexing
        # TODO - is this sorted right?
        IE = np.sum(E, 0).argsort()[::-1]  # find the indices to sort, and reverse them
        srtdE = np.sum(E, 0)[IE]

        # Phi = D @ V(:, IE(1, 2:10))
        Phi = D @ V[:, IE[1:]]

        print('done')

        ###
        ### Part III
        ###

        # TODO - not necessary?  (Independent coordinates?)

        # Extend reference embedding to the entire set
        print('extending embedding (building Psi) ... ', end='')

        Psi_list = []  # holds all the psi_i values

        omega = np.sum(A2, 1)
        A2_nrm = A2 / np.tile(omega.reshape([-1, 1]), [1, m])  # omega needed to be shaped as a column

        # for i=1:size(Phi,2)
        for i in range(Phi.shape[1]):
            # this line is strange ... order of operations for @?, what is the offset?
            psi_i = A2_nrm @ Phi[:, i] / np.sqrt((srtdE[i + 1]))
            # [Psi, psi_i]
            Psi_list.append(psi_i)

        # convert Psi_list back into an array, shaped like MATLAB version
        self.Psi = np.array(Psi_list).T

        # psi have have very small imaginary values ...
        # cast to real here, but need to check
        self.Psi = np.real(self.Psi)

        # print('Psi.shape', Psi.shape)

        print('done')

        # Since close to a degenerate case - try to rotate according to:
        # A. Singer and R. R. Coifman, "Spectral ICA", ACHA 2007.
        #

    def _embedding_parallel(self, process_pool=None):
        ###
        ### Part I
        ###

        ## Configuration

        # the variable m defines some subset of the data, to make computation faster;
        # this could be various values (10% of the data, all the data, etc.), as long
        # as it is not GREATER than the length of data.
        #   For the smallest change, setting to min 4000 or the data size

        # m = 4000                  # starting point for sequential processing/extension
        #
        # TODO - m allows you to sample various sections of the manifold, ratheer than looking at
        # all points to all points
        # the random points can come from the different chunks as well?
        #   ... for ease of coding, the datastructure could be back to 2D data
        m = np.min((4000, self.z_mean.shape[1]))
        print('using', m, 'for variable m')

        data = self.z_mean.T  # set the means as the input set
        M = data.shape[0]

        # Choose subset of examples as reference
        # this is 'take m (4000) random values from z_mean, and sort them
        # subidx = sort(randperm(size(z_mean, 2), m))
        # Choose first m examples as reference (commented out, don't do this
        # subidx = 1:m;
        subidx = np.arange(self.z_mean.shape[1])
        np.random.shuffle(subidx)  # shuffle is inplace in python
        subidx = subidx[:m]  # take a portion of the data
        subidx.sort()  # sort is also in place ...

        # dataref = data(subidx,:)
        dataref = data[subidx, :]

        ##
        # Affinity matrix computation

        Dis = np.zeros((M, m))
        print('computing Dis matrix in parallel')
        start_time = time.time()
        pool = process_pool
        if process_pool == None:
            l = Lock()
            pool = Pool(initializer=workers.parallel_init, initargs=(l,))
        if sys.version_info >= (3,8,0):
            print('Python version is >= 3.8, using shared memory')
            from multiprocessing import shared_memory
            #create shared memory for numpy arrays
            shm_inv_c = shared_memory.SharedMemory(name='inv_c',
                                                   create=True, size=self.inv_c.nbytes)
            shm_subidx = shared_memory.SharedMemory(name='subidx',
                                                    create=True, size=subidx.nbytes)
            shm_dataref = shared_memory.SharedMemory(name='dataref',
                                                     create=True, size=dataref.nbytes)
            shm_data = shared_memory.SharedMemory(name='data',
                                                  create=True, size=data.nbytes)
            
            shm_result = shared_memory.SharedMemory(name='dis', 
                                                    create=True, size=Dis.nbytes)
            #copy arrays into shared memory
            inv_c_copy = np.ndarray(self.inv_c.shape, self.inv_c.dtype, buffer=shm_inv_c.buf)
            np.copyto(inv_c_copy, self.inv_c, casting='no')
            subidx_copy = np.ndarray(subidx.shape, subidx.dtype, buffer=shm_subidx.buf)
            np.copyto(subidx_copy, subidx, casting='no')
            dataref_copy = np.ndarray(dataref.shape, dataref.dtype, buffer=shm_dataref.buf)
            np.copyto(dataref_copy, dataref, casting='no')
            data_copy = np.ndarray(data.shape, data.dtype, buffer=shm_data.buf)
            np.copyto(data_copy, data, casting='no')

            #use pool to run function in parallel
            func = partial(workers.dis_shm, inv_c_copy.shape, inv_c_copy.dtype,
                          subidx_copy.shape, subidx_copy.dtype,
                          dataref_copy.shape, dataref_copy.dtype,
                          data_copy.shape, data_copy.dtype, 
                          Dis.shape, Dis.dtype, M)
            
            #build starmap iterable
            arr = []
            cpu_count = os.cpu_count()
            step = m//cpu_count
            start = 0
            for i in range(cpu_count-1):
                arr.append((step, start))
                start = start + step
            arr.append((m - start, start))
            #run function in parallel
            pool.starmap(func, arr)
            
            if process_pool == None:
                pool.close()
                pool.join()
            #copy results out of shared memory
            Dis_copy = np.ndarray(Dis.shape, Dis.dtype, buffer=shm_result.buf)
            np.copyto(Dis, Dis_copy, casting='no')
            del inv_c_copy
            del subidx_copy
            del dataref_copy
            del data_copy
            del Dis_copy
            
            #close and cleanup shared memory
            shm_inv_c.close()
            shm_inv_c.unlink()
            shm_subidx.close()
            shm_subidx.unlink()
            shm_dataref.close()
            shm_dataref.unlink()
            shm_data.close()
            shm_data.unlink()
            shm_result.close()
            shm_result.unlink()
        else:
            # without shared memory, each worker process will use ~700MB of RAM.
            # with it, they will use ~100MB each
            print('Python version is < 3.8, cannot use shared memory. Beware of high memory usage')
            dis = partial(workers.dis, self.inv_c, subidx, dataref, data, M)
            results = pool.map(dis, range(m), chunksize=m//os.cpu_count())
            if process_pool == None:
                pool.close()
                pool.join()
            for j in range(m):
                Dis[:, j] = results[j]
        elapsed_time = time.time() - start_time
        print('done in ', str(np.round(elapsed_time, 2)), 'seconds!')

        ## Anisotropic kernel

        print('aniostropic kernel ... ', end='')

        ep = np.median(np.median(Dis, 0))  # default scale - should be adjusted for each new realizations

        A = np.exp(-Dis / (4 * ep))
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

        # Compute all eigenvectors and select 10
        # [V, E] = eigs(W2, 10) Matlab
        # V, E = mh.eig_like_matlab(W2, 10)  # think this is correct now ...

        # Compute only 10 eigenvectors, must have symmetric matrix
        V, E = mh.eigs_like_matlab(W2,10)

        # print('V.shape', V.shape)
        # print('E.shape', E.shape)

        # python np.sum(A,0) <=> matlab sum(A)
        # in matlab, srted are the values of sum(E) sorted (in descending order)
        # and IE are the indices that sorted them
        # [srtdE, IE] = sort(sum(E), 'descend')

        # this is python eqivalent ... note that IE will have values one less than the MATLAB, because zero indexing
        # TODO - is this sorted right?
        IE = np.sum(E, 0).argsort()[::-1]  # find the indices to sort, and reverse them
        srtdE = np.sum(E, 0)[IE]

        # Phi = D @ V(:, IE(1, 2:10))
        Phi = D @ V[:, IE[1:]]

        print('done')

        ###
        ### Part III
        ###

        # TODO - not necessary?  (Independent coordinates?)

        # Extend reference embedding to the entire set
        print('extending embedding (building Psi) ... ', end='')

        Psi_list = []  # holds all the psi_i values

        omega = np.sum(A2, 1)
        A2_nrm = A2 / np.tile(omega.reshape([-1, 1]), [1, m])  # omega needed to be shaped as a column

        # for i=1:size(Phi,2)
        for i in range(Phi.shape[1]):
            # this line is strange ... order of operations for @?, what is the offset?
            psi_i = A2_nrm @ Phi[:, i] / np.sqrt((srtdE[i + 1]))
            # [Psi, psi_i]
            Psi_list.append(psi_i)

        # convert Psi_list back into an array, shaped like MATLAB version
        self.Psi = np.array(Psi_list).T

        # psi have have very small imaginary values ...
        # cast to real here, but need to check
        self.Psi = np.real(self.Psi)

        # print('Psi.shape', Psi.shape)

        print('done')

        # Since close to a degenerate case - try to rotate according to:
        # A. Singer and R. R. Coifman, "Spectral ICA", ACHA 2007.
        #

    def _clustering(self, numClusters=7, kmns=True, distance_measure=None, nrep=1):

        # Cluster embedding and generate figures and output files
        # ***************************************************************@

        import matplotlib.pyplot as plt

        # Configuration
        intrinsicDim = self.Dim  # can be varied slightly but shouldn't be much larger than Dim

        ## Clusters
        # IDX = kmeans(Psi(:, 1:intrinsicDim), numClusters)

        # Python kmeans see
        # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.vq.kmeans.html
        # scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-05)
        #
        #  note, python expects each ROW to be an observation, looks the same a matlap
        #

        if kmns == True:
            print('running k-means')
            kmeans = KMeans(n_clusters=numClusters).fit(self.Psi[:, :intrinsicDim])
            self.IDX = kmeans.labels_
        else:
            print('calculating distances')
            
            if (distance_measure == None):
                print('Euclidean distances used in clustering')
                row, col = self.Psi.shape
                combined = []
                for i1 in range(row):
                    combined.append(self.Psi[i1, :intrinsicDim])
                distmat = calculate_distance_matrix(combined)
            else:
            #elif (distance_measure == 'dtw'):
                print('DTW distances used in clustering')
                distmat = self.dtw_distmat
            
            print('sampling initial medoids')
            sample_idx = random.sample(range(distmat.shape[0]), numClusters)
            initial_medoids = sample_idx

            print('running k-medoids')
            self.kmeds = kmedoids(distmat, initial_medoids, data_type='distance_matrix')
            self.kmeds.process()
            temp_idx = np.array(self.kmeds.get_clusters())          
            final_idx = []
            for i1 in range(distmat.shape[0]):
                for j1 in range(numClusters):
                    if (i1 in temp_idx[j1]):
                        final_idx.append(j1)
            self.IDX = np.array(final_idx)
            print(self.IDX.shape)
            if (distance_measure != None):
                return


        # TODO decide how to plot multiple snips
        # think that x_ref[1,:] is just
        for snip in range(len(self.z)):
            if snip == 0:
                x = self.z[snip][5, :]
                x = x[0:x.shape[0]-self.H]
                xref1 = x[::self.stepSize]  # downsample, to match the data steps
            else:
                x = self.z[snip][5, :]
                x = x[0:x.shape[0]-self.H]
                x = x[::self.stepSize]
                xref1 = np.append(xref1, x)

        print(xref1.shape)

        xs = self.Psi[:, 0]
        ys = self.Psi[:, 1]
        zs = self.Psi[:, 2]

        # normalize these to amplitude one?
        print('normalizing amplitudes of Psi in Python ...')
        xs /= np.max(np.abs(xs))
        ys /= np.max(np.abs(ys))
        zs /= np.max(np.abs(zs))

        # xs -= np.mean(xs)
        # ys -= np.mean(ys)
        # zs -= np.mean(zs)

        # xs /= np.std(xs)
        # ys /= np.std(ys)
        # zs /= np.std(zs)

        print(xs.shape)

        lim = 2000
        val = xref1[:lim]
        idx = self.IDX[:lim]

        plt.figure(figsize=[15, 3])

        plt.plot(xref1[:lim], color='black', label='Timeseries')
        plt.plot(xs[:lim], linewidth=.5, label='psi_0')
        plt.plot(ys[:lim], linewidth=.5, label='psi_1')
        plt.plot(zs[:lim], linewidth=.5, label='psi_2')

        plt.plot(idx / np.max(idx) + 1, linewidth=.8, label='IDX')

        if np.max(self.snip_number[:lim])>0:
            plt.plot(self.snip_number[:lim] / np.max(self.snip_number[:lim]) - 2, linewidth=.8, label='Snip Number')

        plt.legend()

        # rightarrow causes an image error, when displayed in github!
        # plt.xlabel('Time $ \\rightarrow $')
        plt.xlabel('Time')
        plt.ylabel('Value')

        # plt.gca().autoscale(enable=True, axis='both', tight=None )
        # plt.gca().xaxis.set_ticklabels([])
        # plt.gca().yaxis.set_ticklabels([])

        plt.title('Example Timeseries and Manifold Projection')

        print('done')

        ###
        ### additional parsing, for color graphs
        ###
        import matplotlib

        cmap = matplotlib.cm.get_cmap('Spectral')

        r = xs[:lim]
        g = ys[:lim]
        b = zs[:lim]

        # prevent the jump in data value
        r[:self.H] = r[self.H]
        g[:self.H] = g[self.H]
        b[:self.H] = b[self.H]

        r -= np.min(r)
        r /= np.max(r)

        g -= np.min(g)
        g /= np.max(g)

        b -= np.min(b)
        b /= np.max(b)

        plt.figure(figsize=[15, 3])

        for i in range(lim - 1):
            col = [r[i], g[i], b[i]]
            plt.plot([i, i + 1], [val[i], val[i + 1]], color=col)

        plt.title('data, colored according to Psi (color three-vector)')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.show()

