import skmultiflow as sk
from timeseries import Timeseries, SinGenerator, CosGenerator
from tree import Node_of_tree
from anytree.search import findall
from anytree import RenderTree
import numpy as np
import math

def print_tree(root_node, obs_number):
    print("tree at observation #{}".format(i))
    for pre, fill, node in RenderTree(root_node):
        print("%s%s %s %s" % ( \
            pre, \
            node.name, \
            node.statistics.dist_dict_coef.get('d1_val'), \
            node.list_timeseries_names() if node.active_cluster else " [NOT ACTIVE]" ))

print("#################")
print("##### INIT. #####")
print("#################")

# generate some timeseries
# S0-S1 with sine waves
# S2-S7 with cosine waves
n_ts = 8
np.random.seed(seed=45)
series = {}
series['S0'] = Timeseries('S0', SinGenerator())
series['S1'] = Timeseries('S1', SinGenerator())
series['S2'] = Timeseries('S2', CosGenerator())
series['S3'] = Timeseries('S3', CosGenerator())
series['S4'] = Timeseries('S4', CosGenerator())
series['S5'] = Timeseries('S5', CosGenerator())
series['S6'] = Timeseries('S6', CosGenerator())
series['S7'] = Timeseries('S7', CosGenerator())

# Initialize root node of the tree (the initial cluster) and set timeseries to initial cluster
root_node = Node_of_tree('root_node')
root_node.set_cluster_timeseries(series)

# initial run: let the tree grow to cluster the timeseries
# for each active cluster - get next value for each series inside the cluster and calculate and update statistics

for i in range(1000):
    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print_tree(root_node, i)

print('#################')
print_tree(root_node, i)

print("#################")
print("##### DRIFT 1 ###")
print("#################")

# We now change the type of S6 and S7 completely: they are now sine waves
# we also introduce a little dissimilarity in the cluster S2-S5


series['S2'].generator = CosGenerator(start=0.1)
series['S3'].generator = CosGenerator(start=0.2)
series['S4'].generator = CosGenerator(start=0.3)
series['S5'].generator = CosGenerator(start=0.4)

series['S6'].generator = SinGenerator(start=0.1)
series['S7'].generator = SinGenerator(start=0.4)

for i in range(1000, 2000):
    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print_tree(root_node, i)

print('#################')
print_tree(root_node, i)

print("#################")
print("##### DRIFT 2 ###")
print("#################")

# We now change S6 and S7 back to sine waves
# and add some more dissimilarity in the cluster S2-S5

series['S4'].generator = SinGenerator(start=0.1)
series['S5'].generator = SinGenerator(start=0.4)

series['S6'].generator = SinGenerator(start=0.2)
series['S7'].generator = SinGenerator(start=0.3)

for i in range(2000,3000):
    #print("# " + str(i))
    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print_tree(root_node, i)

print('#################')
print_tree(root_node, i)
