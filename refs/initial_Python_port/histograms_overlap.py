#
# histograms_overlap
#

H = 40        # non-overlapping window length for histogram/empirical densities estimation
stepSize = 5
nbins = 5     # of histogram bins

from manifolder_helper import histogram_bins_centered

## Concatenate 1D histograms (marginals) of each sensor in short windows
z_hist_list = []   # in Python, lists are sometimes easier than concatinate

print('calculating histograms for', N, 'dimensions (univariate timeseries) ', end='')

# for dim=1:N
for dim in range(N):    # loop run standard Python indexing, starting at dim = 0
    print('.', end='')
    series = z[dim, :]  # grab a row of data

    # NOTE, MATLAB and python calculate histograms differently
    # MATLAB uses nbins values, as bins centerpoints, and
    # Python uses nbins+1 values, to specify the bin endpoints

    # note, hist_bins will always be [0 .25 .5 .75 1], in MATLAB
    # equivalent for python hist is
    #   [-0.12   0.128  0.376  0.624  0.872  1.12 ]
    hist_bins = histogram_bins_centered(series, nbins)

    z_hist_dim_list = []

    # for i=1:floor((size(z,2)-H)/stepSize)
    i_range = int(np.floor(z.shape[1] - H) / stepSize)
    for i in range(i_range):
        # interval = z(dim, 1 + (i - 1) * stepSize: (i - 1) * stepSize + H);
        interval = series[i * stepSize:i*stepSize + H]

        # take the histogram here, and append it ... should be nbins values
        # first value returned by np.histogram the actual histogram
        #
        #  NOTE!!! these bins to not overlap completely with the MATLAB version,
        #   but are roughly correct ... probably exact boundaries are not the same,
        #   would need to look into this ...
        #
        hist = np.histogram(interval, hist_bins)[0]
        z_hist_dim_list.append(hist)

    # convert from a list, to array [nbins x (series.size/stepSize?)]
    z_hist_dim = np.array(z_hist_dim_list).T

    # z_hist = [z_hist; z_hist_dim];
    z_hist_list.append(z_hist_dim)

# convert from list back to numpy array ... need to check values!
# end result is [40x17528]
z_hist = np.concatenate(z_hist_list)

print(' done')
