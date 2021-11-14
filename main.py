# Frauscher Dora, Chromy Aaron

import random

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial.distance import cityblock
import csv

# ----------- open and read lines out of file ------------
file = open('input.csv', 'r', encoding='utf-8-sig')
line_numbers = [0, 1]  # line numbers to be read
lines = []
for i, line in enumerate(file):
    # read line 0 and 1
    if i in line_numbers:
        lines.append(line.strip())
    elif i > 1:
        # don't read after line 7 to save time
        break

# ------------ assign read information to variables ----------

n_clusters = [int(lines[0].strip(';'))]  # assign 1st row of file to variable
[rows, columns] = lines[1].split(';')  # assign 2nd row of file to variables
rows = int(rows)
columns = int(columns)
file.close()

# ------------ read data from file into a pandas DataFrame -------------
data = pd.read_csv('input.csv', sep=';', decimal=',', skiprows=2, header=None,
                   encoding='utf8')  # each row is data object of 2 domensions(columns)
data.columns = ["X", "Y"]

# -------------- create random centers ------------------

min_X = min(data.loc[:, "X"])
max_X = max(data.loc[:, "X"])
min_Y = min(data.loc[:, "Y"])
max_Y = max(data.loc[:, "Y"])

x_draw = np.random.uniform(min_X, max_X, n_clusters)
y_draw = np.random.uniform(min_Y, max_Y, n_clusters)
rdata = pd.DataFrame({"X": x_draw, "Y": y_draw})  # assigns random centers to DataFrame

plt.scatter(data.loc[:, "X"], data.loc[:, "Y"])  # plots each data object as data point
plt.scatter(rdata.loc[:, "X"], rdata.loc[:, "Y"])
plt.title("data + random centers")
plt.show()


# ------------ k means calculation ---------------------------

def get_min_dist(point, points=rdata, dist_func=cityblock):  # calculates distances using manhattan distance (citiblock)
    distances = points.apply(lambda x: dist_func(point, x), axis=1)
    return distances.idxmin()


centers = rdata  # storing random start centers
counter = 0

while True:
    counter += 1
    data["nearest"] = data[["X", "Y"]].apply(lambda x: get_min_dist(x, points=centers), axis=1)

    new_centers = data.groupby('nearest').mean()
    dist = ((new_centers - centers) ** 2).sum().sum()
    centers = new_centers

    plt.scatter(data.loc[:, "X"], data.loc[:, "Y"], c=data.nearest)  # plots each data object as data point
    plt.scatter(centers.loc[:, "X"], centers.loc[:, "Y"], color='red')
    plt.title("Iteration: " + str(counter))
    plt.show()

    if dist == 0:
        break

print("Iterations: " + str(counter))
print('Centers: ', centers)

plt.scatter(data.loc[:, "X"], data.loc[:, "Y"], c=data.nearest)  # plots each data object as data point
plt.scatter(centers.loc[:, "X"], centers.loc[:, "Y"], color='red')
plt.title("data + optimized centers")
plt.show()

with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(n_clusters)  # number of centers
    for i in range(n_clusters[0]):
        writer.writerow([centers.loc[i, "X"], centers.loc[i, "Y"]])  # list of seeds
    writer.writerow([counter])  # number of iterations
    for i in range(len(data)):
        writer.writerow([data.nearest[i], data.loc[i, "X"], data.loc[i, "Y"]])  # data entries (rows, columns)

# The number of iterations is the number of steps until
# the cluster centers do not change their position anymore.

# https://stackoverflow.com/questions/6253729/kmeans-with-l1-distance-in-python


# print(cluster.cluster_centers_)

# print(cluster.labels_)

# plt.scatter(data.loc[:, "X"], data.loc[:, "Y"], c=cluster.labels_)
# plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], color='black')
# plt.show()
