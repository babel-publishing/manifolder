
# clustering data ...

# IDX = manifolder.IDX

# print(len(IDX))
# print(np.unique(IDX))

type(manifolder.Psi[:, :manifolder.Dim])
print(manifolder.Psi.shape)
print(manifolder.Psi[:, :manifolder.Dim].shape)
print(type(manifolder.Psi))

a = []
print(type(a))
a.append([1,2,3])
print(a)
a.append([4,5,6])
print(a)
b = np.array(a).T
print(b)

b1 = b[1]
b2 = b[2]
print(type(b))
print(type(b1))

row, col = manifolder.Psi.shape
print(row)
print(col)

print('Manifold example')
t1 = manifolder.Psi[100,:manifolder.Dim]
t2 = manifolder.Psi[900,:manifolder.Dim]

dist = dtw(t1, t2)
print(dist)

# combined = ([row for row_group in zip(t1, t2) for row in row_group])
combined = []
combined.append(t1)
combined.append(t2)
print(combined)
print(type(combined))
print(type(t1))

print('Using dtw')
distmat = cdist_dtw(combined)
print(distmat.shape)
print(type(distmat))
print(distmat)

print('Using Euclidean')
distmat = (calculate_distance_matrix(combined))
# print(distmat.shape)
print(type(distmat))
print(distmat)

# print('Synthetic example')
# nArr2D = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
# print(nArr2D)
# print(type([21, 22, 23]))
# print(type(nArr2D))

# distmat = cdist_dtw(nArr2D)
# print(distmat.shape)
# print(distmat)

cluster_lens = mh.count_cluster_lengths(IDX)

# cluster_lens is a dictionary a dictonary, where each key is the cluster number (0:6),
# and the values are a list of clusner lengths

mh.show_cluster_lens(cluster_lens)

