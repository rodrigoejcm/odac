import skmultiflow as sk
from timeseries import Timeseries, generate_multi_timeseries,get_timeseries_next_value
from tree import Node_of_tree
from anytree.search import findall
from anytree import RenderTree

### INITIALIZE TIMESERIES ( n == number of timeseries)

series = generate_multi_timeseries(n=41)

### Initialize root node of the tree ( the initial cluster )
### and tet timeseries to initial cluster
root_node = Node_of_tree('root_node')
root_node.set_cluster_timeseries(series)

### example run 10 times:
### for each active cluster - get next value for each series inside the cluster and
### calculate and update statistics

for i in range(50):

    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        if active_cluster.test_split() or active_cluster.test_aggregate():
            print("tree at observation #{}".format(i))
            for pre, fill, node in RenderTree(root_node):
                print("%s%s %f %s" % (pre, node.name, node.statistics.dist_dict_coef['d1_val'], node.list_timeseries_names() if node.active_cluster else " [NOT ACTIVE]" ))
