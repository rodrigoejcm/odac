import skmultiflow as sk
from timeseries import Timeseries, generate_multi_timeseries,get_timeseries_next_value
from tree import Node_of_tree
from anytree.search import findall

### INITIALIZE TIMESERIES ( n == number of timeseries)

series = generate_multi_timeseries(n=3)

### Initialize root node of the tree ( the initial cluster )
### and tet timeseries to initial cluster
root_node = Node_of_tree('root_node')
root_node.set_cluster_timeseries(series)

### example run 10 times:
### for each active cluster - get next value for each series inside the cluster and
### calculate and update statistics

for i in range(10):

    for active_cluster in findall(root_node,filter_=lambda node: node.active_cluster is True):
        active_cluster.update_statistics()
        active_cluster.test_split()
