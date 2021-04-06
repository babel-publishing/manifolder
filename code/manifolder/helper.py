__all__ = (
    'count_cluster_lengths',
    'show_cluster_lengths',
    'print_cluster_lengths',
    'make_transition_matrix',
    'make_matrix_markov',
    'reorder_cluster',
    'image_M',
)

import math
import collections

import numpy as np
from numpy import linalg as LA

from scipy.sparse import linalg as LAs
from scipy.stats import norm, kurtosis
from scipy import stats
from scipy.stats import skew

import matplotlib.pyplot as plt


def histogram_bins_centered(data, nbins):

    """ helper function for numpy histograms, to generate bins that are
        centered, similar to the MATLAB hist. for python, the bins are
        specified by nbins + 1 numbers, marking the boundaries of the bins.
        this allows for bins of different widths, ranging across the data.
    """

    # used by histograms_overlap

    bins = np.linspace(np.min(data), np.max(data), nbins + 1)

    return bins


def histogram_bins_all_snips(data, nbins):

    """ helper function to find bin spacing across snippets,
        similar to histogram_bins_centered on one series. """

    N = data[0].shape[0]
    n = len(data)
    for dim in range(N):  # loop over dimensions of signal
        maxval = -math.inf
        minval = math.inf
        for snip in range(n):  # loop over snippets to get same dimension each time
            maxval = np.maximum(maxval, np.max(data[snip][dim, :]))
            minval = np.minimum(minval, np.min(data[snip][dim, :]))

        bins = np.linspace(minval, maxval, nbins + 1)  # bins now has nbins+1 values and ranges across data

        if dim == 0:
            hist_bins = [bins]
        else:
            hist_bins.append(bins)

        # results in list of arrays

    return hist_bins


# svd_like_matlab used by histograms overlap
def svd_like_matlab(A):
    """ The MATLAB and python SVDs return different values
        this function uses the python libraries, but the return values are
        those specfied in MATLAB https://www.mathworks.com/help/matlab/ref/double.svd.html

          [U,S,V] = svd(A)

          performs a singular value decomposition of matrix A, such that

            A = U*S*V'

        IN PYTHON,

          u, s, vh = np.linalg.svd(A)

          Factors the matrix a as

          u @ np.diag(s) @ v

        where u and v are unitary and s is a 1d array of a's singular values

        note that Python uses @ for matrix multiplication, and .T for transpose"""

    # U, S, V = svd(A)

    # use lowercase variable names for the python call
    u, s, vh = np.linalg.svd(A)

    # MATLAB users expect
    #  U, S, V = svd(A)

    # rename MATLAB like variable here
    # note that Python and MATLAB algos effectively flip U and V
    U = u  # no need to transpose!
    S = np.diag(s)  # in MATLAB, S is a diagonal matrix
    V = vh.T

    # print(U.shape)
    # print(S.shape)
    # print(V.shape)

    return U, S, V


def svds_like_matlab(A,k=None):
    """ The MATLAB and python SVDs return different values
        this function uses the python libraries, but the return values are
        those specfied in MATLAB https://www.mathworks.com/help/matlab/ref/double.svd.html

          [U,S,V] = svd(A)

          performs a singular value decomposition of matrix A, such that

            A = U*S*V'

        IN PYTHON,

          u, s, vh = np.linalg.svd(A)

          Factors the matrix a as

          u @ np.diag(s) @ v

        where u and v are unitary and s is a 1d array of a's singular values

        note that Python uses @ for matrix multiplication, and .T for transpose"""

    if k is None:
        k = A.shape[0]
    
    # U, S, V = svd(A)

    # use lowercase variable names for the python call
    u, s, vh = LAs.svds(A,k)

    # MATLAB users expect
    #  U, S, V = svds(A,dim)

    idx = [i[0] for i in sorted(enumerate(s), reverse=True, key=lambda x:x[1])]

    # rename MATLAB like variable here
    # note that Python and MATLAB algos effectively flip U and V
    U = u  # no need to transpose!
    V = vh.T

    U = U[:,idx]
    S = np.diag(s[idx])  # in MATLAB, S is a diagonal matrix
    V = V[:,idx]

    # print(U.shape)
    # print(S.shape)
    # print(V.shape)

    return U, S, V


# eig_like_matlab used by embeddings
def eig_like_matlab(A, k=None):
    """ like matlab's
           d = eigs(A,k)

        https://www.mathworks.com/help/matlab/ref/eigs.html
        returns the k biggest (???) eigenvectors """

    print('Using full eigensolver from numpy')

    if k is None:
        k = A.shape[0]

    # this is how eig is usually called in python
    w, v = LA.eig(A)

    D = np.diag(w[:k])  # MATLAB returns a diagonal matrix

    # NOTE - do I need to sort these first, or are they automatically largerst?
    #  (check by looking at D?)
    V = v[:, :k]  # rename, and remove later eigenvectors

    # TODO - we are not sorting the vectors according to eigenvalues?  (check?)

    return V, D

def eigs_like_matlab(A, k=None):
    """ like matlab's
           d = eigs(A,k)

        https://www.mathworks.com/help/matlab/ref/eigs.html
        returns the k biggest (???) eigenvectors """

    print('Using partial symmetric eigensolver from scipy')
    
    if k is None:
        k = A.shape[0]

    # this is how eig is usually called in python
    w, v = LAs.eigsh(A, k, which = 'LM', ncv = min(A.shape[0], max(5*k + 1, 20)))

    idx = [i[0] for i in sorted(enumerate(w), reverse=True, key=lambda x:x[1])]

    D = np.diag(w[idx])  # MATLAB returns a diagonal matrix

    # NOTE - do I need to sort these first, or are they automatically largerst?
    #  (check by looking at D?)
    V = v[:, idx]  # rename, and remove later eigenvectors

    # TODO - we are not sorting the vectors according to eigenvalues?  (check?)

    return V, D

def eig_like_matlab_test():
    """ code, and make sure it looks like the MATLAB """

    # Python eigenvalaues
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    #
    #   w,v = eig(a)
    #
    #  v[:,i] is the RIGHT eigenvector corresponding to values w[i]
    #
    #  (appears that in python, w[0] ≈ 1,
    #   meaning eigenvectors have been normed and sorted)
    #
    # MATLAB eigs
    # https://www.mathworks.com/help/matlab/ref/eigs.html
    #       [V, D] = eigs(A,k)
    #
    #   (to make matters more confusing, they use [V, E] = eigs(W2,10))
    #
    #  V[:,i] and D[i,i]
    #  V columns are eigenvectors
    #  D is the diagonal matrix of the eigen values
    #

    #    returns the k biggest (???) eigenvectors """

    a = np.random.rand(3, 3)

    V, E = eig_like_matlab(a, 2)

    print('\n original matrix a:\n' + str(a))
    print('\n V:\n' + str(V))
    print('\n E:\n' + str(E))

    #     w, v = LA.eig(a)

    #     print('\n w (as diags):\n' + str(np.diag(w)))

    #     print('\n v\n' + str(v))

    #     i = 0
    #     eigvec = v[:,i]
    #     eigval = w[i]

    #     print('\n eigval\n' + str(w[i]))
    #     print('\n eigvec\n' + str(eigvec))

    #     print('A * eigvec / eigval ')
    #     print('\n multilpy:' + str(a @ eigvec / eigval))


import pandas as pd


def simplify_data(z_shape=(8, 87660)):
    """ takes in a datastream of data, z, and makes a simple version
        specifically, z should be [(8, 87660)] """
    # set at era to me about 200*5 points ...
    # this will give about 10 eras over 10k points,
    # which can be downsampled by 5, and viewed as 2000 points

    era = 1000  # how long is each sub-section (cos, gaussian noise, etc.)

    total_length = z_shape[1]  # total length of the z (and z_mod)

    # create a datastructure, containing low-level noise, to contain the data
    z_mod = 1e-3 * np.random.randn(z_shape[0], z_shape[1])  # fill with low-level noise

    # this will hold the actual signal (only for the first row)
    sig = np.array([])

    while (sig.size < total_length):
        # print('appending some more data ...')

        sig_zero = np.zeros(era)

        # gaussian noise
        sig_gauss = np.random.randn(era)

        # uniform noise
        sig_unif = np.random.rand(era)

        # cos, 10 cycles over the era
        sig_cos = np.cos(2 * np.pi * np.arange(era) / era * 10.)

        # sin, 5 cycles over the era
        sig_sin = .1 * np.sin(2 * np.pi * np.arange(era) / era * 5.)

        sig_root_sin = np.sqrt(np.abs(np.sin((2 * np.pi * np.arange(era) / era * 10.))))

        # append all together ...
        sig = np.concatenate((sig, sig_zero, sig_gauss, sig_unif, sig_cos, sig_sin, sig_root_sin))

    # signal may be too long, cut to correct length
    sig = sig[:total_length]

    # fade in and out each section, usin a cosine wave
    sin_mask = np.sin(2 * np.pi * np.arange(total_length) / (2 * era))
    sig = sig * sin_mask

    # add to the final dataset
    z_mod[0, :] += sig
    z_mod[1, :-1] += np.diff(sig)  # add the diff to the second row

    # optional?  Normalize each row from 0 to 1
    z_mod = z_mod - np.min(z_mod, axis=1).reshape(-1, 1)
    z_mod = z_mod / np.max(z_mod, axis=1).reshape(-1, 1)

    z_mod = np.round(z_mod, 6)

    # save the data
    # note that it was originally created with time as columns,
    # the standard python format is time as rows (and features as columns),
    # so transpose

    # the fmt command suppress scientific notation / small number-junk
    np.savetxt('data/simple_data.csv', z_mod.T, delimiter=',', fmt='%1.6f')

    return z_mod.T


def test_moms():
    """ look at numpy's calculation of higher order moments (through kurtosis), see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html#scipy.stats.skew
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html#scipy.stats.kurtosis
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
    """

    # create some test data
    x = np.random.randn(100) + (.1 + .2 * np.random.rand(100))

    print('*** calculating moments, using numpy')
    print('np.mean(x)', np.mean(x))
    print('np.var(x)', np.var(x))

    print('*** higher moments, using scipy')
    print('skew', skew(x))
    print('kurtosis (Fisher is default)', kurtosis(x))

    print('scipy describe', stats.describe((x)))

    # des = stats.describe(x)

    # note, all the values are the same, except var, which is 1/(n-1) in describe, but 1/n for numpy ... weird ...


def get_log_spaced_bins(max_value=350.1):
    """ generate the bin boundaries for use with np.histogram, with log spacing
        the CENTER bin values will be 1,2,5,10,20, etc
        the BOUNDARY values will be 0, 1.5,2.5,7.5,15, etc.
        """
    assert max_value > 0, 'max_value must be positive'

    bin_values = [0]

    mult = 1
    while (True):
        bin_values.append(mult * 1)
        if bin_values[-1] > max_value * 10: break

        bin_values.append(mult * 2)
        if bin_values[-1] > max_value * 10: break

        bin_values.append(mult * 5)
        if bin_values[-1] > max_value * 10: break

        mult *= 10

    bin_values = np.array(bin_values)
    # [0    1    2    5   10   20   50  100  200  500]

    bin_boundaries = np.diff(bin_values) / 2 + bin_values[:-1]
    # [0.5    1.5    3.5    7.5   15.    35.    75.   150.   350.]

    bin_boundaries[0] = 0
    # [0    1.5    3.5    7.5   15.    35.    75.   150.   350.]

    # made extra bins - now, cut down the bin boundaries so that the data just fits
    while (bin_boundaries[-2] > max_value):
        bin_boundaries = bin_boundaries[:-1]  # cut out last element

    # cut down the bin values, to match
    # (do not need first element, which is zero, and should be one less elements that the bin_boundaries
    bin_values = bin_values[1:bin_boundaries.size]
    print(bin_values)
    print(bin_boundaries)


def count_cluster_lengths(x):
    """ takes an array in, x, which contains a series of cluster labels,
        and returns the result in a dictionary
    """

    # find the unique values (these are the cluster names)
    keys = np.unique(x)
    np.sort(keys)  # happens in place
    print('cluster names (keys)', keys)

    # create dictionary of 'cluster_lengths' with empty lists
    cluster_lens = collections.OrderedDict()
    for key in keys:
        cluster_lens[key] = []

    # find the matching values
    # loop through the sequence
    i = 0
    while (i < x.size):
        this_val = x[i]
        d = np.where(x[i:] != this_val)[0]  # find where the value changes
        if d.size > 0:
            # now know how many points, before the value changes
            this_len = d[0]  # first location that is not this_val

        else:
            this_len = x.size - i  # this was the last cluster, goes to the end

        # this_val is the custer
        # this_len is the length of that cluster
        # ... store it!
        cluster_lens[this_val].append(this_len)
        # print(str(this_val),str(this_len))

        i += this_len

    return cluster_lens


def print_cluster_lengths(cluster_lens):
    """ prints out the dictionary object created, that stores cluster lengths
    """
    for key in cluster_lens.keys():
        # also, sort the lists here
        cluster_lens[key].sort()  # on the list, happens in place
        print('key', key, 'value', cluster_lens[key], '\n')


def show_cluster_lengths(cluster_lens, sharey=True):
    """ plots the lengths of the clusters, as determined above
        sharey=False allows each subplot to have different y-axis limits
    """
    keys = list(cluster_lens.keys())
    nkeys = len(keys)

    plt.figure()

    fig, axes = plt.subplots(nkeys, 1, sharex=True, sharey=sharey, figsize=[7, nkeys * 1 + 1])

    # loop through, and make all the histograms, as subplots
    for k in range(nkeys):
        key = keys[k]

        different_lengths_this_cluster = np.unique(cluster_lens[key])

        for l in different_lengths_this_cluster:
            number_of_occurrences = np.sum(np.where(cluster_lens[key] == l)[0])
            # print('length', l, 'number_of_occurrences', number_of_occurrences)

            # plot as a vertical bar
            axes[k].plot([l, l], [0, number_of_occurrences])

        axes[k].set_ylabel('cl ' + str(k))

    plt.tight_layout()
    plt.suptitle('cluster length histograms')
    plt.xlabel('cluster lengths')
    plt.show()


###
### transition matrix
###
def make_transition_matrix(states):
    """ transition matrix of a bunch of states (like IDX)"""
    states_max = np.max(states)

    tmat = np.zeros((states_max + 1, states_max + 1))

    # note, transition matrix is written with the
    #   STATES AS COLUMNS, as in quantum mechanics / physics
    #
    # evolution is then:
    #   |state+1> = tmat @ |state>
    #
    # (but need to normalize tmat)
    #

    for i in range(states.size - 1):
        # states[i+1] is valid
        state_from = states[i]
        state_to = states[i + 1]
        tmat[state_to, state_from] += 1

    return tmat


def make_matrix_markov(A):
    """ takes a matrix, and normalizes so that each column sums to one
        (assumes matrix values are already positive!)"""

    # makes more sense to normalize so that the columns sum to one
    col_sum = np.sum(A, axis=0).reshape(1, -1)

    A_markov = A / col_sum  # col_sum will broadcast

    return A_markov


def image_M(data, vmax=None):
    """ t for transition matrix"""
    plt.figure(figsize=(7, 7))

    # create scaled data
    if vmax is None:
        vmax = np.max(np.abs(data))

    # cmap = 'gist_heat'
    # cmap = 'bone'
    cmap = 'hot'
    # cmap = 'binary'
    plt.imshow(data, vmin=0, vmax=vmax, cmap=cmap)

    plt.grid(b=None)

    plt.xlabel('from')
    plt.ylabel('to')

    plt.colorbar()

    plt.title('transition matrix')

    plt.show()


def reorder_cluster(IDX, M):
    """ renames the clusters, so the diagonal elements of M will be in decreasing order
        note, M must be regenerated from the new_IDX that is returned"""
    print('NOTE, need to fix bug, sometimes orders backwards')

    idx_freq = M.diagonal()

    new_idx = np.zeros_like(IDX)

    # sort the values of the index, from largest to smallest
    new_order = np.argsort(idx_freq)

    # so weird ... new_order is alternately lowest-to-highest, and highest-to-lowest
    # just reorder, if needed
    # if idx_freq[new_order[-1]] > idx_freq[new_order[-1]]:
    #     # frequency INCREASES at the end ... reorder!
    #     print('yup!')
    #     new_order = new_order[::-1]
    # else:
    #     print('nerp!!')

    new_order = new_order[::-1]

    for i in range(len(new_order)):
        # find all the locations matching next index needed
        loc = np.where(IDX == new_order[i])
        new_idx[loc] = i  # reorder, starting with i

    return new_idx
