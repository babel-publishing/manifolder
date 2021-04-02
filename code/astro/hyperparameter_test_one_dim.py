# useful set of python includes

%load_ext autoreload
%autoreload 2

import numpy as np

np.set_printoptions(suppress=True, precision=4)

import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'svg'

import pandas as pd
import time
import random
import os
import sys
sys.path.append(r"C:\Users\acloninger\GDrive\ac2528Backup\DocsFolder\GitHub\manifolder")
sys.path.append(r"..")

sys.path.append("/home/jovyan/gen-mod-vol/avlab/manifolder/code")

import manifolder as mr
from manifolder import helper as mh
import dcor
from pyclustering.cluster.kmedoids import kmedoids

# load the data
# note, you must have started the notebook in the 
print('loading data ...')
df = pd.read_excel('astro_data/dataset_2.xlsx', index_col=0)
df.head()

# convert values from loaded spreadsheet, into a numpy matrices
# note that there is no need for the first value, which is time,
# as it is not part of the manifold
#
# also, note the spreadsheet is missing a column name for `Unnamed: 13`, and the values above
# this have the incorrect column labels; the first relevant vale is bx, which as a magnitude around 2
#
# note the final value of each row is the goal (0 or 1), and not part of z

data_raw = df.values[:, 1:]
print('first line of raw_data:\n', data_raw[0, :])
import pickle
#Load Data
segments = np.load('themis/segments-newdata-all.npy')

# Load Labels
labels = np.load('themis/labels-newdata-all.npy')
labels = np.asarray(pd.get_dummies(labels))

#Load Bounding Boxes/ Event Locations
with open('themis/bbox.pickle','rb') as f:
    bboxes = pickle.load(f)
    
# loop through the data, breaking out the clusters
# i will always point to the NaN (blank line) in the dataframe,
# and values [i-1440:i] is the snipped

snippet_len = 1440

# collect all line breaks (blank lines) in csv file
#lineBreaks = [0]
#for i in range(data_raw.shape[0]):
#    if data_raw[i,0] != data_raw[i,0]:  # replacement of isnan, since nan != nan
#        lineBreaks.append(i)    
#lineBreaks.append(data_raw.shape[0])
#
#num_snippet = len(lineBreaks)-1


# callect the snippets into two groups, one for each goal (target) value, 0 or 1
# these can be easily merged
zs_0 = []
zs_1 = []

locallabel_0 = []
locallabel_1 = []
snippet_index = 0;

df.values[0,:]

reduce_dimension = True

dimension_num = 5

for i in range(snippet_len,data_raw.shape[0]+1,snippet_len+1):
    # copy the snipped, excluding the last value, which is the goal
    snippet = data_raw[i-snippet_len:i,:-1]
    
    if reduce_dimension:
        snippet = snippet[:,dimension_num]
        snippet = snippet.reshape(snippet_len,1)

    # grab the goal value from the first row of each snippet
    goal = data_raw[i-snippet_len,-1]
    
    # check to make sure each snippet does not contain NaN
    # (should not, if parsing is correct)
    assert ~np.isnan(snippet).any(), 'oops, snippet contains a Nan!'
    
    print('snippet size',snippet.shape,'with goal',goal)
    
    snippetlabel = np.zeros(snippet_len)
    if goal == 1:
        bmin = int(bboxes[snippet_index][0][0])
        bmax = int(bboxes[snippet_index][0][2])
        snippetlabel[bmin:bmax] = 1
    
    if goal == 0:
        zs_0.append( snippet )
        locallabel_0.append( snippetlabel )
    elif goal == 1:
        zs_1.append( snippet )
        locallabel_1.append( snippetlabel )
    else:
        assert False, 'value of goal not understood'
        
    snippet_index = snippet_index + 1;
        

# shuffle this lists; this should not strictly be necessary, if all the data is being used,
# but prevents biases when shortening the list

c0 = list(zip(zs_0, locallabel_0))
random.shuffle(c0)
zs_0, locallabel_0 = zip(*c0)
zs_0 = list(zs_0)
locallabel_0 = list(locallabel_0)

c1 = list(zip(zs_1, locallabel_1))
random.shuffle(c1)
zs_1, locallabel_1 = zip(*c1)
zs_1 = list(zs_1)
locallabel_1 = list(locallabel_1)

shorten_data = False

if shorten_data:
    zs_0 = zs_0[:20]
    zs_1 = zs_1[:20]
    locallabel_0 = locallabel_0[:20]
    locallabel_1 = locallabel_1[:20]
        
zs = zs_0 + zs_1
locallabel = locallabel_0 + locallabel_1
z_breakpoint = len(zs_0)

print( '\done!')
print( '\t len(zs_0):',len(zs_0))
print( '\t len(zs_1):',len(zs_1))
print( '\t len(zs):',len(zs))

# build list of index pairs for the initial medoids
tot = 104
idx = []
for i1 in range(tot):
    for j1 in range((tot-1)):
        temp = [i1,j1]
        idx.append(temp)

def try_medoids(nclust, dm, idx=None):
    len_data = dm.shape[0]
    if idx == None:
        initial_medoids = random.sample(range(len_data), nclust)
    else:
        initial_medoids = idx
    kmedoids_instance = kmedoids(dm, initial_medoids, data_type = 'distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    #print(clusters)
    clusters[0].sort()
    clusters[1].sort()
    return clusters

dim_set = (4,)
H_set = range(20, 200, 20)
step_size_set = range(5, 30, 5)
nbins_set = (10,)
ncov_set = (10,)
rdims_set = range(1,14)

#dim=6
#H = 160
#step_size = 20
#nbins = 10
#ncov = 10

all_results = []
for dim in dim_set:
    for H in H_set:
        for step_size in step_size_set:
            for nbins in nbins_set:
                for ncov in ncov_set:
                    for rdims in rdims_set:
                        # data has been parsed, now run Manifolder
                        #start_time = time.time()

                        # create manifolder object
                        manifolder = mr.Manifolder(dim=dim,H=H,step_size=step_size,nbins=nbins, ncov=ncov)

                        # add the data, and fit (this runs all the functions)
                        # manifolder.fit_transform(zs, parallel=False, dtw="raw")
                        manifolder.fit_transform(zs, parallel=True)

                        #elapsed_time = time.time() - start_time
                        #print('\n\t Program Executed in', str(np.round(elapsed_time, 2)), 'seconds')  # about 215 seconds (four minutes)

                        snippet_psi = []
                        size = manifolder.Psi.shape[0] // len(zs)
                        for i in range(0, manifolder.Psi.shape[0], size):
                            snippet_psi.append(manifolder.Psi[i:i+size, :])

                        dcor_dm = np.zeros((len(zs), len(zs)))
                        #start_time = time.time()
                        for i in range(len(snippet_psi)):
                            for j in range(i):
                                distance = dcor.homogeneity.energy_test_statistic(snippet_psi[i], snippet_psi[j])
                                dcor_dm[i,j] = distance
                                dcor_dm[j,i] = distance
                        #print('\n\t Dcor Executed in', str(np.round(time.time() - start_time, 2)), 'seconds')
                        #print(dcor_dm)
                        
                        # clustering results from manifolder with energy distance

                        results = []
                        for i in range(len(idx)):
                            kmeds = try_medoids(2, dcor_dm, idx=idx[i])
                            found = False
                            for temp in results:
                                if (all(elem in temp[0][0] for elem in kmeds[0]) and all(elem in temp[0][1] for elem in kmeds[1]))or \
                                (all(elem in temp[0][0] for elem in kmeds[1]) and all(elem in temp[0][1] for elem in kmeds[0])):
                                    temp[1] += 1
                                    found = True
                                    break
                            if not found:
                                results.append([kmeds, 1])
                                
                        #print("########################################################################")
                        #print("before sorting")
                        #print(results)
                        #print("after sorting")
                        results = sorted(results, key=lambda array: array[1])
                        #print(results)
                        #print("########################################################################")

                        cluster_out = results[-1][0]
                        #print(cluster_out)
                        #res_temp = results[-1][1]
                        #print(res_temp)
                        truePositive = 0
                        falsePositive = 0
                        trueNegative = 0
                        falseNegative = 0
                        for i in range(len(cluster_out[0])):
                            if cluster_out[0][i] < len(zs_0):
                                trueNegative += 1
                            else:
                                falsePositive += 1
                        for i in range(len(cluster_out[1])):
                            if cluster_out[1][i] >= len(zs_0):
                                truePositive += 1
                            else:
                                falseNegative += 1
                        confusion_matrix = ((trueNegative, falsePositive),(falseNegative, truePositive))
                        parameter_list = (dim, H, step_size, nbins, ncov, rdims)
                        accuracy = max(truePositive+trueNegative, falsePositive+falseNegative)
                        all_results.append((parameter_list, confusion_matrix, accuracy))
                        #print("TN:", trueNegative)
                        #print("FP:", falsePositive)
                        #print("TP:", truePositive)
                        #print("FN:", falseNegative)



filename = ".\hyperparameter_sweep_results.pickle"
counter = 1
while(os.path.isfile(filename)):
    filename = ".\hyperparameter_sweep_results" + str(counter) + ".pickle"
    counter += 1
try:
    pfile = open(filename, "wb+")
    pickle.dump(all_results, pfile)
    pfile.close()
    print('Pickle file written to "' + filename + '"')
except Exception as ex:
    print(ex)
    print("Unable to write pickle file")