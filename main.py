import skmultiflow as sk
from timeseries import Timeseries, generate_multi_timeseries, get_timeseries_next_value, SinGenerator, CosGenerator
from tree import Node_of_tree
from anytree.search import findall
from anytree import RenderTree
import numpy as np
import math

print("#################")
print("##### INIT. #####")
print("#################")

### generate some timeseries
### every timeseries with an even id has a sine generator function
### every timeseries with an odd id has a cosine generator function
n_ts = 41
np.random.seed(seed=42)
series = generate_multi_timeseries(n_ts=n_ts)

### Initialize root node of the tree (the initial cluster) and set timeseries to initial cluster
root_node = Node_of_tree('root_node')
root_node.set_cluster_timeseries(series)

### initial run: let the tree grow to cluster the timeseries
### for each active cluster - get next value for each series inside the cluster and calculate and update statistics

for i in range(200):
    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            for pre, fill, node in RenderTree(root_node):
                print("%s%s %s %s" % ( \
                    pre, \
                    node.name, \
                    node.statistics.dist_dict_coef.get('d1_val'), \
                    node.list_timeseries_names() if node.active_cluster else " [NOT ACTIVE]" ))

print("#################")
print("##### DRIFT #####")
print("#################")

### simulate concept drift: in the lower half of the timeseries,
### those with a cosine generator have a sine generator now as well
for key, value in series.items():
    if value.id < n_ts/2 and value.id%2 == 1:
        print("{}: sine generator".format(value.name))
        value.generator = SinGenerator()

for i in range(200):
    #print("# " + str(i))
    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            for pre, fill, node in RenderTree(root_node):
                print("%s%s %s %s" % ( \
                    pre, \
                    node.name, \
                    node.statistics.dist_dict_coef.get('d1_val'), \
                    node.list_timeseries_names() if node.active_cluster else " [NOT ACTIVE]" ))
