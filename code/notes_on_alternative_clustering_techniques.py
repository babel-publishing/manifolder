#!/usr/bin/env python

import time
import pandas as pd
import manifolder as mr

# ### Alterantive Custering Techniques
# 
# Look at 'null-set' clustering, as well as tSNE, and Manifolder

start_time = time.time()

# load the data
data_location = 'data/solar_wind_data.csv'
df = pd.read_csv(data_location, header=None)
z = df.values
print(f'loaded {data_location}, shape: {z.shape}')

# create manifolder object
manifolder = mr.Manifolder()

# add the data, and fit (this runs all the functions)
manifolder.fit_transform(z)

manifolder._clustering()  # display

elapsed_time = time.time() - start_time
print(f'Program Executed in {elapsed_time:.2f} seconds')

# using a stepSize of 5 for the clustering;
# cut down to this size, to compare ... 
# note this drops points for the analysis (but use this as a startng point)
data_standard = z.T

data_standard = data_standard[::5, :]
print('data_standard.shape', data_standard.shape)

# data_standard = data_standard[:4000,:] # shorten more, so algos run faster
print('data_standard.shape', data_standard.shape)

# also, keep track of the manifolder data ... 
# since manifolder uses a stepSize of 5, do not downsample
# before passing the data in - output data is automatically downsampled ...
data_manifolder = z.T
# data_manifolder = data_manifolder[:4010*5,:]

# loop through reconstruction techniques

from sklearn.manifold import TSNE

start_time = time.time()
techniques = ['null','tsne','manifolder']

for technique in techniques:
    print('*** reconstructing using', technique)

    if technique == 'null':
        print('calculating for null (this fast!)')
        z_embedded_null = data_standard # no embedding, embedded is the same as data
        
    elif technique == 'tsne':
        print('calculating TSNE - this can take some time')
        
        z_embedded_tsne = TSNE(n_components=3).fit_transform(data_standard)

    elif technique == 'manifolder':
        print('calculating manifolder')
        manifolder = mr.Manifolder() 
        z_embedded_mani = manifolder.fit_transform(data_manifolder)
        
    else:
        raise TypeError(f'Technique not understood: {technique}')

# cut this down to the same length - passed in data that was slightly too long
# z_embedded_mani = z_embedded_mani[:4000,:]

elapsed_mins = (time.time() - start_time) / 60
print(f'elapsed mins: {elapsed_mins:.2f}')

print(z_embedded_null.shape)
print(data_standard.shape)


from sklearn.cluster import KMeans

# cluster the values from tSNE ...
numClusters = 7

def display_embedding(data, embedding):
    print(  'data.shape',data.shape,'embedding.shape[0]',embedding.shape[0] )
    assert data.shape[0] == embedding.shape[0], 'data and must have same length'
    
    # Configuration
    numClusters = 7         
    intrinsicDim = embedding.shape[1]
    
    print('numClusters',numClusters,'intrinsicDim',intrinsicDim)

    print('running k-means')
    
    # cluster on the embedding
    kmeans = KMeans(n_clusters=numClusters).fit( embedding[:, :intrinsicDim])
    IDX = kmeans.labels_

    xref1 = data[:, 0]
    
    # assume data and the embeddings are the same size
    # (they have been decimated the same amount, before calling this function)

    print(xref1.shape)

    xs = embedding[:, 0]
    ys = embedding[:, 1]
    zs = embedding[:, 2]

    # normalize these to amplitude one?
    print('normalizing amplitudes of embedding dimension in Python ...')
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

    lim = xs.shape[0]
    val = xref1[:lim]
    idx = IDX[:lim]

    figx = 5 + xref1.size / 400
    print('figure x size',figx)
    plt.figure(figsize=[figx, 3])

    plt.plot(xref1[:lim], color='black', label='Timeseries')
    # plt.plot(xs[:lim], linewidth=.5, label='$\psi_0$')
    # plt.plot(ys[:lim], linewidth=.5, label='$\psi_1$')
    # plt.plot(zs[:lim], linewidth=.5, label='$\psi_2$')

    plt.plot(xs[:lim], linewidth=.5, label='psi_0')
    plt.plot(ys[:lim], linewidth=.5, label='psi_1')
    plt.plot(zs[:lim], linewidth=.5, label='psi_2')

    plt.plot(idx / np.max(idx) + 1, linewidth=.8, label='IDX')

    plt.legend()

    # rightarrow causes an image error, when displayed in github!
    # plt.xlabel('Time $ \\rightarrow $')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # plt.gca().autoscale(enable=True, axis='both', tight=None )
    # plt.gca().xaxis.set_ticklabels([])
    # plt.gca().yaxis.set_ticklabels([])

    plt.title('Example Timeseries and Projection')

    print('done')

    ###
    ### additional parsing, for color graphs
    ###
    import matplotlib

    cmap = matplotlib.cm.get_cmap('Spectral')

    r = xs[:lim]
    g = ys[:lim]
    b = zs[:lim]

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

    plt.title('data, colored by embedding dimension')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.show()
    
    return IDX


### Results for Null (no embedding) Clustering 

print('Null clustering')
# calculate clusters, and display; returns clusters ...
IDX = display_embedding(data_standard, z_embedded_null)
cluster_lens = mh.count_cluster_lengths(IDX)
mh.show_cluster_lens(cluster_lens, sharey=False)
print('DONE')


### Results for TSNE Clustering 

print('TSNE')

# calculate clusters, and display; returns clusters ...
IDX = display_embedding(data_standard, z_embedded_tsne)

cluster_lens = mh.count_cluster_lengths(IDX)
mh.show_cluster_lens(cluster_lens, sharey=False)


### Results for Manifolder Clustering 

print('Trying manifolder')

# calculate clusters, and display; returns clusters ...
IDX = display_embedding(data_standard, z_embedded_mani)
cluster_lens = mh.count_cluster_lengths(IDX)
mh.show_cluster_lens(cluster_lens, sharey=False)
