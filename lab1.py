import statistics
import time

import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans


def data_sampling(array, shift_x, shift_y):
    for point in array:
        point[0] = point[0] + shift_x
        point[1] = point[1] + shift_y


data_1 = numpy.random.normal(0, 1, size=(500, 2))
data_sampling(data_1, 3, 3)
data_2 = numpy.random.normal(0, 1, size=(500, 2))
data_sampling(data_2, 5, 5)
data_3 = numpy.random.normal(0, 1, size=(500, 2))
data_sampling(data_3, 7, 4)
data_testing = numpy.vstack((data_1, data_2, data_3))

model = KMeans(n_clusters=3)

time_start = time.time()
points_clustered_index = model.fit_predict(data_testing)
time_end = time.time()
print("Time: {:.3f} c.".format(time_end - time_start))
cluster_nums = numpy.unique(points_clustered_index)

clusters = []
for cluster_num in cluster_nums:
    row_ix = numpy.where(points_clustered_index == cluster_num)
    clusters.append(data_testing[row_ix])
    pyplot.scatter(data_testing[row_ix, 0], data_testing[row_ix, 1])
pyplot.title(model.__class__.__name__)
pyplot.show()

print("Mode:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("Cluster {}:".format(index + 1))
    print("x: {:.3f}".format(statistics.mode(cluster_x)))
    print("y: {:.3f}".format(statistics.mode(cluster_y)))

print("Median:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("Cluster {}:".format(index + 1))
    print("x: {:.3f}".format(statistics.median(cluster_x)))
    print("y: {:.3f}".format(statistics.median(cluster_y)))

print("Mean:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("Cluster {}:".format(index + 1))
    print("x: {:.3f}".format(statistics.mean(cluster_x)))
    print("y: {:.3f}".format(statistics.mean(cluster_y)))

print("Standard deviation:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("Cluster {}:".format(index + 1))
    print("x: {:.3f}".format(statistics.stdev(cluster_x)))
    print("y: {:.3f}".format(statistics.stdev(cluster_y)))
