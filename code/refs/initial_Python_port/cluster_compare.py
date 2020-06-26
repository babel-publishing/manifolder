# NOTE THAT 'data' IN THIS CODE IS TRANSPOSE, COMPARED WITH THE MANIFOLDER CODE
# data is shaped [13848,8] here ... ogm ...

# loads in data, dimlist, n_channels, npts, x
# file_location_in_data
# file_location_out_data

import numpy as np

# Coweb, if not installed, can 'pip install -U concept_formation'
# https://github.com/cmaclell/concept_formation
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cluster import cluster

import time

import os

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

dim_list = ['p_speed', 'p_density', 'xhelicity', 'O7to6', 'residualE', 'absB', 'Z_Fe', 'Fe_to_O']

file_location_in_data = '/Users/jonathan/Documents/repos/MANIFOLDER/whitened_short_set/whitened_short_set.csv'
file_location_out_data = '/Users/jonathan/Documents/repos/MANIFOLDER/whitened_short_set/whitened_short_set_2.csv'


# read data from csv file
def load_data():
    """ load the data from csv, and do initial parsing """
    x = np.genfromtxt(file_location_in_data, delimiter=',')
    data = x[:, :].astype('float64')   # data points: 13848 x 8 numpy array: 13848 points, 8 channels

    print('data.shape = ', data.shape)
    #npts = data.shape[0]
    #n_channels = data.shape[1]

    return data


def cluster_cobweb3(data):
    """ cluster the data, using cobweb3"""

    npts = data.shape[0]
    n_channels = data.shape[1]

    # convert data from np array to list of dictionariesw
    data_new = []
    for i in range(npts):
        pt = data[i, :]
        pt_dict = {dim_list[j]: pt[j] for j in range(n_channels)}
        data_new.append(pt_dict)

    # perform cobweb3 clustering and get labels

    print('starting cobweb3')
    print('note, this can take some time ...')
    start_time = time.time()

    tree = Cobweb3Tree()

    clusters = cluster(tree, data_new[:])[0]
    print('# points:', len(clusters))

    clust_names = [c for c in set(clusters)]
    print('  cluster names:', clust_names)

    clust_dict = {c: idx for idx, c in enumerate(clust_names)}
    print(clust_dict)
    lbs = [clust_dict[c] for c in clusters]
    print('length of lbs:', len(lbs))

    clust_dict = {c: idx for idx, c in enumerate(clust_names)}
    print(clust_dict)
    lbs = [clust_dict[c] for c in clusters]
    print('length of lbs:', len(lbs))

    elapsed_time = time.time() - start_time

    print('done, elapsed mins:', np.round(elapsed_time / 60, 2))

    # append labels to csv file of data
    lbs = np.asarray(lbs).reshape(len(lbs), 1)
    print(lbs.shape)
    new = np.concatenate((data, lbs), axis=1)
    print(new.shape)

    np.savetxt(file_location_out_data, new, delimiter=',')

    print('done with cluster_cobweb3')

    # main use of this function is to return the clusters, and the labels?
    return clusters, lbs


def show_clusters_cobweb3(data, clusters, lbs, base_path='results/cobweb/'):
    """ need clusters and lbs """
    npts = data.shape[0]
    n_channels = data.shape[1]

    # get point indices for plotting
    indices_lists = []
    n_clusters = len(set(clusters))
    for i in range(n_clusters):
        indices = np.where(lbs == i)[0].tolist()
        indices_lists.append(indices)
    print('# clusters:', len(indices_lists))
    npts_list = [len(i) for i in indices_lists]
    print('# points in clusters: {}'.format(npts_list))

    # sort indices_lists based on number of points in each sublist
    npts_list_sorted = np.argsort(np.asarray(npts_list))
    print('Sorted indices of lists of number of points:', npts_list_sorted)
    indices_lists_sorted = [indices_lists[i] for i in npts_list_sorted]
    indices_lists = indices_lists_sorted
    print('Sorted # poitns in clusters: {}'.format([len(i) for i in indices_lists]))
    npts_list = sorted(npts_list)
    print('Sorted # poitns in clusters: {}'.format(npts_list))

    # plot clusters - each figure shows two channels
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(20, 4))
    for k in range(n_channels):
        for i in range(n_channels):
            plt.subplot(2, 4, i + 1)
            for indices, c in zip(indices_lists, colors):
                c = c.reshape((1, -1))                                         # NOTE - for the warning, make into a single row?
                plt.scatter(data[indices, k], data[indices, i], c=c, s=1)      # c causes issues
                                                                               #plt.scatter(data[indices, k], data[indices,i], s=1)
                                                                               #plt.title('x:c{} y:c{}'.format(k, i))
        plt.tight_layout()
        plt.savefig('results/cobweb3/c{}_vs_all.pdf'.format(k))                #note save pdfs instead (slower, but better images)
                                                                               #savefig('c{}_vs_all.png'.format(k))
                                                                               #plt.show() # NOTE - code did not originally show()
        plt.clf()
        print('Figure c{} done.'.format(k))
    plt.close()
    print('Figures saved successfully')

    # plot original data (channel=speed) in time
    target_var = data[:, 0]

    # plot 10 intervals of 1000 time points
    interval = 1000
    plt.figure(figsize=(20, 4))
    for i in range(10):
        begin = i * interval
        end = (i+1) * interval
        # find indices of points in each cluster
        lb_indices = []
        for indices in indices_lists:
            indices_interval = [idx for idx in indices if idx >= begin and idx < end]
            lb_indices.append(indices_interval)
        # plot different colors for clusters
        plt.subplot(2, 5, i + 1)
        for indices, c in zip(lb_indices, colors):
            c = c.reshape((1, -1))     # NOTE - for the warning, make into a single row?
            plt.scatter(indices, target_var[indices], c=c, s=1)
        plt.title('x in [{}, {}]'.format(begin, end - 1))
        print('Plot: x in [{}, {}]'.format(begin, end - 1))
    plt.tight_layout()
                                       # savefig('speed_10examples.png')  # TODO, port the figure save
    plt.savefig('results/cobweb3/speed_10examples.pdf')
    plt.close()
    print('Figure saved successfully')


###
### Portion for DBScan
###

from sklearn.cluster import DBSCAN


def scan(data):
    # perform dbscan
    db = DBSCAN(eps=0.5, min_samples=20).fit(data)
    #print gmm.means_
    lbs = db.labels_
    print(lbs)

    lbs = lbs.reshape(lbs.shape[0], 1)
    print(lbs.shape)
    new = np.concatenate((data, lbs), axis=1)
    print(new.shape)
    np.savetxt('results/dbscan/whitened_short_set_labels_dbscan.csv', new, delimiter=',')

    # read data and labels
    # NOTE, this file was already generated ... maybe gets regenerated, from above?
    #file_location_in_data = '/Users/jonathan/Documents/repos/MANIFOLDER/whitened_short_set/whitened_short_set_labels.csv'
    # NOTE - maybe there is a better place for the file?
    ### WHAT???

    # originas is whitened_short_set_labels.csv, not sure ...
    file_location_in_data = '/Users/jonathan/Documents/repos/MANIFOLDER/whitened_short_set/cobweb3/whitened_short_set_labels.csv'

    # looks like this is loaded in from the wrong location ... ???

    data = np.genfromtxt(file_location_in_data, delimiter=',')
    lbs = data[:, -1]
    n_channels = data.shape[1] - 1
    print('Reading from file done.')

    return lbs     # clusters not needed?


def show_clusters_dbscan(data, lbs):
    npts = data.shape[0]
    n_channels = data.shape[1]

    # get point indices for plotting
    indices_lists = []
    num_clusters = 0
    num_points_used = 0
    while True:
        indices = np.where(lbs == num_clusters)[0].tolist()
        if len(indices) == 0:
            break
        indices_lists.append(indices)
        num_points_used += len(indices)
        num_clusters += 1
    n_cluster = len(indices_lists)
    print('# clusters:', n_cluster)
    print('# points used:', num_points_used)
    print('# outliers:', npts - num_points_used)
    npts_list = [len(i) for i in indices_lists]
    print('# points in clusters: {}'.format(npts_list))

    # sort indices_lists based on number of points in each sublist
    npts_list_sorted = np.argsort(np.asarray(npts_list))
    print('Sorted indices of lists of number of points:', npts_list_sorted)
    indices_lists_sorted = [indices_lists[i] for i in npts_list_sorted]
    indices_lists = indices_lists_sorted
    print('Sorted # poitns in clusters: {}'.format([len(i) for i in indices_lists]))
    npts_list = sorted(npts_list)
    print('Sorted # poitns in clusters: {}'.format(npts_list))

    # plot clusters - each figure shows two channels
    colors = cm.rainbow(np.linspace(0, 1, n_cluster))
    plt.figure(figsize=(20, 4))
    for k in range(n_channels):
        for i in range(n_channels):
            plt.subplot(2, 4, i + 1)
            for indices, c in zip(indices_lists, colors):
                c = c.reshape((1, -1))      # NOTE - for the warning, make into a single row?
                plt.scatter(data[indices, k], data[indices, i], c=c, s=1)
            plt.title('x:c{} y:c{}'.format(k, i))
        plt.tight_layout()
        plt.savefig('results/dbscan/c{}_vs_all.pdf'.format(k))
        plt.clf()
        print('Figure c{} done.'.format(k))
    print('Figures saved successfully')

    # plot original data (channel=speed) in time
    target_var = data[:, 0]

    # plot 10 intervals of 1000 time points
    interval = 1000
    plt.figure(figsize=(20, 4))
    for i in range(10):
        begin = i * interval
        end = (i+1) * interval
        # find indices of points in each cluster
        lb_indices = []
        for indices in indices_lists:
            indices_interval = [idx for idx in indices if idx >= begin and idx < end]
            lb_indices.append(indices_interval)
        # plot different colors for clusters
        plt.subplot(2, 5, i + 1)
        for indices, c in zip(lb_indices, colors):
            c = c.reshape((1, -1))     # NOTE - for the warning, make into a single row?
            plt.scatter(indices, target_var[indices], c=c, s=1)
        plt.title('x in [{}, {}]'.format(begin, end - 1))
        print('Plot: x in [{}, {}]'.format(begin, end - 1))
    plt.tight_layout()
                                       #savefig('speed_10examples.png')
    plt.savefig('results/dbscan/speed_10examples.pdf')
    plt.clf()
    print('Figure saved successfully')

    from matplotlib.legend_handler import HandlerLine2D
    # plot colors
    plt.figure(figsize=(4, 4))
    colors = cm.rainbow(np.linspace(0, 1, n_cluster))
    x = range(2)
    for i, c in zip(range(n_cluster), colors):
        y = [(1./n_cluster) * (i+1)] * 2
        #c = c.reshape((1,-1))  # NOTE - for the warning, make into a single row?
        plt.plot(x, y, c=c, label=str(npts_list[i]))
    plt.legend()
    plt.savefig('results/dbscan/colors and number of points.pdf')
    plt.clf()
    print('Plot colors and legend done.')


# if not installed, can 'pip install scikit-fuzzy'
#import skfuzzy_cluster as fcm
# looks like a newer version
import skfuzzy as fuzz


def fuzzy(data):
    # jd - feel like this has been done?
    # # read data from csv file
    # x = np.genfromtxt('../whitened_short_set.csv', delimiter=',')
    # data = x[:, :].astype('float64')   # data points: 13848 x 8 numpy array: 13848 points, 8 channels
    # dim_list = ['p_speed', 'p_density', 'xhelicity', 'O7to6', 'residualE', 'absB', 'Z_Fe', 'Fe_to_O']
    # print('data.shape = ', data.shape)
    # npts = data.shape[0]
    # n_channels = data.shape[1]

    n_channels = data.shape[1]

    # perform fcm
    n_clusters = 8
    # cntr, u, u0, d, jm, p, fpc = fcm.cmeans(data.transpose(), n_clusters, 1.1, error=0.005, maxiter=1000)
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data.transpose(), n_clusters, 1.1, error=0.005, maxiter=1000)
    lbs = np.argmax(u, axis=0)
    print('FCM done...')
    print(lbs)
    print(p)

    # # append labels to csv file of data
    # lbs = lbs.reshape(lbs.shape[0], 1)
    # print lbs.shape
    # new = np.concatenate((data, lbs), axis=1)
    # print new.shape
    # np.savetxt('whitened_short_set_labels.csv', new, delimiter=',')

    # get point indices for plotting
    indices_lists = []
    for i in range(n_clusters):
        indices = np.where(lbs == i)[0].tolist()
        indices_lists.append(indices)
    print('# clusters:', len(indices_lists))
    npts_list = [len(i) for i in indices_lists]
    print('# points in clusters: {}'.format(npts_list))

    # sort indices_lists based on number of points in each sublist
    npts_list_sorted = np.argsort(np.asarray(npts_list))
    print('Sorted indices of lists of number of points:', npts_list_sorted)
    indices_lists_sorted = [indices_lists[i] for i in npts_list_sorted]
    indices_lists = indices_lists_sorted
    print('Sorted # poitns in clusters: {}'.format([len(i) for i in indices_lists]))
    npts_list = sorted(npts_list)
    print('Sorted # poitns in clusters: {}'.format(npts_list))

    # plot clusters - each figure shows two channels
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(20, 4))
    for k in range(n_channels):
        for i in range(n_channels):
            plt.subplot(2, 4, i + 1)
            for indices, c in zip(indices_lists, colors):
                c = c.reshape((1, -1))      # NOTE - for the warning, make into a single row?
                plt.scatter(data[indices, k], data[indices, i], c=c, s=1)
            plt.title('x:c{} y:c{}'.format(k, i))
        plt.tight_layout()
        plt.savefig('results/fcm/c{}_vs_all.pdf'.format(k))
        plt.clf()
        print('Figure c{} done.'.format(k))
    plt.close()
    print('Figures saved successfully')

    # plot original data (channel=speed) in time
    target_var = data[:, 0]

    # plot 10 intervals of 1000 time points
    interval = 1000
    plt.figure(figsize=(20, 4))
    for i in range(10):
        begin = i * interval
        end = (i+1) * interval
        # find indices of points in each cluster
        lb_indices = []
        for indices in indices_lists:
            indices_interval = [idx for idx in indices if idx >= begin and idx < end]
            lb_indices.append(indices_interval)
        # plot different colors for clusters
        plt.subplot(2, 5, i + 1)
        for indices, c in zip(lb_indices, colors):
            c = c.reshape((1, -1))     # NOTE - for the warning, make into a single row?
            plt.scatter(indices, target_var[indices], c=c, s=1)
        plt.title('x in [{}, {}]'.format(begin, end - 1))
        print('Plot: x in [{}, {}]'.format(begin, end - 1))
    plt.tight_layout()
    plt.savefig('results/fcm/speed_10examples.pdf')
    plt.close()
    print('Figure saved successfully')

    from matplotlib.legend_handler import HandlerLine2D

    # plot colors
    plt.figure(figsize=(4, 4))
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    x = range(2)
    for i, c in zip(range(n_clusters), colors):
        y = [(1./n_clusters) * (i+1)] * 2
        plt.plot(x, y, c=c, label=str(npts_list[i]))
    plt.legend()
    plt.savefig('results/fcm/colors and number of points.pdf')
    plt.close()
    print('Plot colors and legend done.')
