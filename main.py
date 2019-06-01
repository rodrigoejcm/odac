"""Demonstration for clustering time series via ODAC algorithm"""

import skmultiflow as sk
from timeseries import Timeseries, SinGenerator
from tree import Node_of_tree
from anytree.search import findall
from anytree import RenderTree
import numpy as np
import math

print("#################")
print("##### INIT. #####")
print("#################")

# generate some timeseries
# S0-S1 with sine waves
# S2-S7 with sine waves, moved by 180°
np.random.seed(seed=42)
series = {}
series['S0'] = Timeseries('S0', SinGenerator())
series['S1'] = Timeseries('S1', SinGenerator())
series['S2'] = Timeseries('S2', SinGenerator(start=math.pi))
series['S3'] = Timeseries('S3', SinGenerator(start=math.pi))
series['S4'] = Timeseries('S4', SinGenerator(start=math.pi))
series['S5'] = Timeseries('S5', SinGenerator(start=math.pi))
series['S6'] = Timeseries('S6', SinGenerator(start=math.pi))
series['S7'] = Timeseries('S7', SinGenerator(start=math.pi))

# Initialize root node of the tree (the initial cluster) and set
# timeseries to initial cluster
root_node = Node_of_tree('root_node')
root_node.set_cluster_timeseries(series)

# initial run: let the tree grow to cluster the timeseries
# for each active cluster - get next value for each series
# inside the cluster and calculate and update statistics
for i in range(1000):
    for active_cluster in \
    findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            root_node.print()

# we can observe, that the algorithm correctly clusters the time series
# one cluster with S0-S1
# one cluster with S2-S7
print('#################')
root_node.print()

# Now, let's introduce a concept drift, that the algorithm has to adjust
print("#################")
print("##### DRIFT 1 ###")
print("#################")

# We now change the type of S6-S7 to be like S0-S1:
series['S6'].generator = \
    SinGenerator(start=series['S6'].generator.state + math.pi)
series['S7'].generator = \
    SinGenerator(start=series['S7'].generator.state + math.pi)

for i in range(1000, 2000):
    for active_cluster in \
    findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            root_node.print()

# The algorithm split cluster S2-S7 into two clusters
# one with S2-S5
# one with S6-S7
# We know, that S6-S7 are now of the same generating structure as S0-S1
# still, we are missing one criteria for aggregation:
# the cluster size of the child cluster has to grow bigger than the
# parent cluster
print('#################')
root_node.print()

# Let's now change S6-S7 back to it's original generating function
# that S2-S7 would be of the same shape
# but in order for the aggregation function to apply, we need to
# increase the cluster sizes
# that it is bigger than their parent clusters size + hoeffding bound
print("#################")
print("##### DRIFT 2 ###")
print("#################")

# We do this by adding some dissimilarity within the clusters
series['S2'].generator = SinGenerator(start=series['S2'].generator.state - 0.4)
series['S3'].generator = SinGenerator(start=series['S3'].generator.state - 0.2)
series['S4'].generator = SinGenerator(start=series['S4'].generator.state + 0)
series['S5'].generator = SinGenerator(start=series['S5'].generator.state + 0.2)

series['S6'].generator = SinGenerator(start=series['S6'].generator.state \
    - math.pi + 0.2)
series['S7'].generator = SinGenerator(start=series['S7'].generator.state \
    - math.pi - 0.3)

for i in range(2000,3000):
    for active_cluster in \
    findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            root_node.print()

# We can see, that the cluster readjusts to model the right structure
# one cluster with S0-S1
# one cluster with S2-S7
print('#################')
root_node.print()
