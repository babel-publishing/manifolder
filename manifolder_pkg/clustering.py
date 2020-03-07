# Cluster embedding and generate figures and output files
# ***************************************************************@

import matplotlib.pyplot as plt

# Configuration
numClusters = 7         # NOTE, this was previously 14 (too many!)
intrinsicDim = Dim      # can be varied slightly but shouldn't be much larger than Dim

## Clusters

# clusterig
from sklearn.cluster import KMeans

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# Cluster embedding and generate figures and output files
# ***************************************************************@
# Configuration

## Clusters
# IDX = kmeans(Psi(:, 1:intrinsicDim), numClusters)

# Python kmeans see
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.vq.kmeans.html
# scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-05)
#
#  note, python expects each NOW to be an observation, looks the same a matlap
#

# a = np.random.randn(100,15)

print('running k-means')
# IDX = kmeans(Psi[:,:intrinsicDim], numClusters)

# IDX = kmeans(Psi(:, 1:intrinsicDim), numClusters)

kmeans = KMeans(n_clusters=numClusters).fit(Psi[:, :intrinsicDim])
IDX = kmeans.labels_

## for now, just plot like this:
# $$$$$$
# think that x_ref[1,:] is just
xref1 = z[0, :]
xref1 = xref1[::5]      # shorten, to look like steps?

print(xref1.shape)

xs = Psi[:, 0]
ys = Psi[:, 1]
zs = Psi[:, 2]

# normalize these to amplitude one?
print('normalizing amplitueds of Psi in Python ...')
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

plt.figure(figsize=[15, 3])

plt.plot(xref1[:lim], color='black', label='Timeseries')
plt.plot(xs[:lim], linewidth=.5, label='$\psi_0$')
plt.plot(ys[:lim], linewidth=.5, label='$\psi_1$')
plt.plot(zs[:lim], linewidth=.5, label='$\psi_2$')

plt.plot(IDX[:lim] / np.max(IDX) + 1, linewidth=.8, label='IDX')

plt.legend()

# rightarrow causes an image error, when displayed in github!
#plt.xlabel('Time $ \\rightarrow $')
plt.xlabel('Time')
plt.ylabel('Value')

#plt.gca().autoscale(enable=True, axis='both', tight=None )

# plt.gca().xaxis.set_ticklabels([])
# plt.gca().yaxis.set_ticklabels([])

plt.title('Example Timeseries and Manifold Projection')
# plt.plot(IDX[:lim])
# plt.plot(xs)

print('done')

###
### additional parsing, for color graphs
###
import matplotlib

cmap = matplotlib.cm.get_cmap('Spectral')

val = xref1[:lim]
idx = IDX[:lim]

r = xs[:lim]
g = ys[:lim]
b = zs[:lim]

# prevent the jump in data value
r[:H] = r[H]
g[:H] = g[H]
b[:H] = b[H]

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

         # Figures

         # TODO - much!
         # figure
         # subplot(2,2,1);
         # scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,1:size(Psi,1))
         # title('Color by Time');
         # axis image

         # subplot(2,2,2);
         # scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,IDX)
         # title('Color by Cluster');
         # axis image
         #
         # if size(x_ref,1)>=1
         # subplot(2,2,3);
         # scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,x_ref(1,:))
         # title('Color by x_ref(1,:)');
         # axis image
         # end
         #
         # if size(x_ref,1)>=2
         # subplot(2,2,4);
         # scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,x_ref(2,:))
         # title('Color by x_ref(2,:)');
         # axis image
         # end

         ## Output Files
         # time = (1:size(z,2))'

         # time = time(1 + H / 2:stepSize:end - H / 2)

         # data = z';
         # data = downsample(data(1+H/2:end-H/2,:), stepSize,floor(stepSize/2));

         # print('Currently assuming x_ref(2,:) contains old labels')
         # # dummy header
         # cHeader = {'Time' 'Data 1' 'Data 2' 'Data 3' 'Data 4' 'Data 5' 'Data 6' 'Data 7' 'Data 8' 'Old Labels' 'New Labels'};
         # # commaHeader = [cHeader;repmat({','}, 1, numel(cHeader))]  # insert commaas
         # commaHeader = commaHeader(:)'
         # textHeader = cell2mat(commaHeader)  # cHeader in text with commas
         #
         # fid = fopen('HomaTimeSeriesClustering.csv', 'w')
         # fprintf(fid, '%s\n', textHeader)
         # fclose(fid)
         # # dlmwrite('HomaTimeSeriesClustering.csv', [time data x_ref(2,:)' IDX],' - append')
         #
         # save('homa_data_embedding_and_parameters.mat', 'H', 'stepSize', 'nbins', 'ncov', 'm', 'subidx', 'Psi')
