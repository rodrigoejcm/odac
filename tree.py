from anytree import NodeMixin, RenderTree
from itertools import combinations_with_replacement
from collections import OrderedDict
import numpy as np
from math import sqrt, log


class Statistics:
    """Maintains the sufficient statistics for a cluster"""

    def __init__(self, ts_quantity):
        self.reset_sufficient_statistics(ts_quantity)
        self.cluster_diameter = None
        self.dist_dict_coef = {}
        self.hoeffding_bound = None

    def reset_sufficient_statistics(self, ts_quantity):
        self.sum_dict = {}       # key: cluster number
        self.prd_dict = {}       # key: tuple of two cluster numbers
        self.corr_dict = {}      # key: tuple of two cluster numbers
        self.rnomc_dict = {}     # key: tuple of two cluster numbers
        self.n_of_instances = 0
        for i in range(ts_quantity):
            self.sum_dict[i] = 0.
            for j in range(ts_quantity):
                if j >= i:
                    self.prd_dict[(i,j)] = 0.
                    if j > i:
                        self.corr_dict[(i,j)] = 0.
                        self.rnomc_dict[(i,j)] = 0.

    def print(self):
        print("# n_of_instances = {}".format(self.n_of_instances))


class Cluster:
    """A cluster. Maintains all associated variables and provides all necessary functions to modify it"""

    def __init__(self, confidence_level = 0.9, n_min = 5, tau = 0.1):
        self.active_cluster = True
        self.statistics = None
        self.confidence_level = confidence_level
        self.n_min = n_min
        self.tau = tau

    def set_cluster_timeseries(self,list_ts):
        self.list_of_timeseries = OrderedDict(sorted(list_ts.items(), key=lambda t: t[0]))
        cluster_size = len(self.list_of_timeseries)
        self.statistics = Statistics(cluster_size)
        self.update_statistics(init=True) ### does not generate ts samples only calculates matrices

    def get_cluster_timeseries(self):
        return self.list_of_timeseries

    def list_timeseries_names(self):
        return list(self.list_of_timeseries.keys())

    def calcula_sum_dict(self,init=False):

        for k in self.statistics.sum_dict:
            self.statistics.sum_dict[k] = self.statistics.sum_dict[k] \
                + list(self.list_of_timeseries.values())[k].current_value

        return self.statistics.sum_dict


    def calcula_prod_dict(self,init=False):

        for k in self.statistics.prd_dict:
            self.statistics.prd_dict[k] = self.statistics.prd_dict[k] \
                + ( list(self.list_of_timeseries.values())[k[0]].current_value \
                * list(self.list_of_timeseries.values())[k[1]].current_value )

        return self.statistics.prd_dict

    def calcula_corr_dict(self,init=False):

        for k in self.statistics.corr_dict:
            i = k[0]
            j = k[1]
            p = self.statistics.prd_dict[(i,j)]
            a = self.statistics.sum_dict[i]
            a2 = self.statistics.prd_dict[(i,i)]
            b = self.statistics.sum_dict[j]
            b2 = self.statistics.prd_dict[(j,j)]
            n = self.statistics.n_of_instances

            term_p = p - ((a*b)/n)
            term_a = sqrt(a2 - ((a*a)/n))
            term_b = sqrt(b2 - ((b*b)/n))

            self.statistics.corr_dict[(i,j)] = term_p/(term_a*term_b)

        return self.statistics.corr_dict


    def calcula_rnomc_dict(self,init=False):

        max_rnomc = None
        for k in self.statistics.rnomc_dict:
            self.statistics.rnomc_dict[k] = sqrt( (1-self.statistics.corr_dict[k]) / 2 )
            if max_rnomc is None or self.statistics.rnomc_dict[k] > max_rnomc:
               max_rnomc = self.statistics.rnomc_dict[k]

        self.cluster_diameter = max_rnomc
        return self.statistics.rnomc_dict

    def calcula_distances_coefficients(self):

        if len(self.statistics.rnomc_dict) == 0:
            return

        rnorm_copy = self.statistics.rnomc_dict.copy()

        self.statistics.dist_dict_coef['avg'] = sum(rnorm_copy.values()) / len(rnorm_copy)

        d0_pair = min(rnorm_copy, key=rnorm_copy.get)
        d0 =  rnorm_copy[min(rnorm_copy, key=rnorm_copy.get)]

        self.statistics.dist_dict_coef['d0_val'] = d0
        self.statistics.dist_dict_coef['d0_pair'] = d0_pair

        d1_pair = max(rnorm_copy, key=rnorm_copy.get)
        d1 =  rnorm_copy[max(rnorm_copy, key=rnorm_copy.get)]

        self.statistics.dist_dict_coef['d1_val'] = d1
        self.statistics.dist_dict_coef['d1_pair'] = d1_pair

        rnorm_copy.pop(d1_pair, None)

        if (rnorm_copy):
            ## in case rnorm dict has only one or 2 elements. in that case
            ## we are calculating this but we are not allowed to split.
            ## TODO >>> we can also check the rnorm lenght in the begining.

            d2_pair =  max(rnorm_copy, key=rnorm_copy.get)
            d2 =   rnorm_copy[max(rnorm_copy, key=rnorm_copy.get)]
            self.statistics.dist_dict_coef['d2_val'] = d2
            self.statistics.dist_dict_coef['d2_pair'] = d2_pair
            self.statistics.dist_dict_coef['delta'] = d1-d2
        else:
            self.statistics.dist_dict_coef['d2_val'] = None
            self.statistics.dist_dict_coef['d2_pair'] = None
            self.statistics.dist_dict_coef['delta'] = None


    def calcula_hoeffding_bound(self,init=False):

        r_sqrd = 1  # because the data is normalized
        self.statistics.hoeffding_bound = sqrt(r_sqrd * log(1/self.confidence_level) \
            / (2 * self.statistics.n_of_instances))

        return self.statistics.hoeffding_bound


    def update_statistics(self , init=False):

        if init == False:
            self.get_new_timeseries_values()

        #print("### new observation:", end='')
        #for ts in list(self.list_of_timeseries.values()):
        #    print(", {}".format(ts.current_value), end='')

        self.statistics.n_of_instances += 1

        ### calculate Matrice
        self.calcula_sum_dict()

        ### calculate prod matrix
        self.calcula_prod_dict()

        if self.statistics.n_of_instances >= self.n_min:

            ### calculate Matrice coor
            self.calcula_corr_dict()

            ### calculate Matrice dif
            self.calcula_rnomc_dict()

        # hoeffding bound as epsilon proposed in 3.4.1
        self.calcula_hoeffding_bound()

        # distance parameters needed for checks in 3.4.1 and 3.4.3 of the paper
        self.calcula_distances_coefficients()


        #self.statistics.print()
        #print("")


    def get_new_timeseries_values(self):
        for ts in self.list_of_timeseries.values():
            ts.next_val()


    def get_smaller_distance_with_pivot(self, pivot_1, pivot_2, current):
        """ Look for the distance
        in rnomc_dict given 2 pivots and an index
        """

        ## Following the rule that the dict key present a tuple(x,y) where x < y
        ## here the method max and min are used to find the correct poition
        dist_1 = self.statistics.rnomc_dict[(min(pivot_1,current),max(pivot_1,current))]
        dist_2 = self.statistics.rnomc_dict[(min(pivot_2,current),max(pivot_2,current))]

        return 2 if dist_1 >= dist_2 else 1


    def split_this_cluster(self, pivot_1, pivot_2):

        pivot_1_list = {}

        temp_1 = list(self.list_of_timeseries.items())[pivot_1]
        pivot_1_list[temp_1[0]] = temp_1[1]

        pivot_2_list = {}

        temp_2 = list(self.list_of_timeseries.items())[pivot_2]
        pivot_2_list[temp_2[0]] = temp_2[1]

        for i in range(len(self.list_of_timeseries.values())):
            if (i != pivot_1) & (i != pivot_2):
                cluster = self.get_smaller_distance_with_pivot(pivot_1,pivot_2,i)
                if cluster == 1:
                    temp_1 = list(self.list_of_timeseries.items())[i]
                    pivot_1_list[temp_1[0]] = temp_1[1]
                else: #2
                    temp_2 = list(self.list_of_timeseries.items())[i]
                    pivot_2_list[temp_2[0]] = temp_2[1]

        ### After creating 2 lists based on 2 pivots, its time
        ### to creates the new nodes and set self as the parent node

        ## this is just to create the node name.
        ## this is irrelevant...
        if(self.name == 'root_node'):
            new_name = "1"
        else:
            new_name = str(int(self.name[8:]) + 1)

        cluster_child_1 = Node_of_tree('CH1_LVL_'+new_name, parent=self)
        cluster_child_1.set_cluster_timeseries(pivot_1_list)

        cluster_child_2 = Node_of_tree('CH2_LVL_'+new_name, parent=self)
        cluster_child_2.set_cluster_timeseries(pivot_2_list)

        ### Finally, the current cluster is deactivated.

        self.active_cluster = False


    def aggregate_this_cluster(self):

        self.statistics.reset_sufficient_statistics(len(self.list_of_timeseries))
        self.children = []
        self.active_cluster = True


    def test_split(self):

        if self.statistics.n_of_instances >= self.n_min and self.statistics.dist_dict_coef.get('d2_val') is not None:

            d0 = float(self.statistics.dist_dict_coef['d0_val'])
            d1 = float(self.statistics.dist_dict_coef['d1_val'])
            d2 = float(self.statistics.dist_dict_coef['d2_val'])
            avg = float(self.statistics.dist_dict_coef['avg'])
            t = float(self.tau)
            e =  float(self.statistics.hoeffding_bound)

            ### following 3.4.4 Split Algorithm
            if ( (d1 - d2) > e ) | ( t > e ) :
                if ( (d1 - d0) * abs( (d1 - avg) - (avg - d0)) ) > e:

                    x1 = self.statistics.dist_dict_coef['d1_pair'][0]
                    y1 = self.statistics.dist_dict_coef['d1_pair'][1]

                    print("#################")
                    print("##### SPLIT #####")
                    print("#################")
                    print(" >>> PIVOTS: x1={} y1={}".format(x1,y1))
                    print("#################")

                    self.split_this_cluster(pivot_1 = x1, pivot_2 = y1)

                    return True
        return False


    def test_aggregate(self):

        if self.parent is not None and self.statistics.n_of_instances >= self.n_min and self.statistics.dist_dict_coef.get('d1_val') is not None:
            if self.statistics.dist_dict_coef['d1_val'] - self.parent.statistics.dist_dict_coef['d1_val'] \
                > max(self.statistics.hoeffding_bound, self.parent.statistics.hoeffding_bound):

                print("#################")
                print("##### AGGR. #####")
                print("#################")

                print(">>> VARIABLES: c_k={} c_j={} e_k={} e_j={}".format( \
                    self.statistics.dist_dict_coef['d1_val'], \
                    self.parent.statistics.dist_dict_coef['d1_val'], \
                    self.statistics.hoeffding_bound, \
                    self.parent.statistics.hoeffding_bound))
                print("#################")

                self.parent.aggregate_this_cluster()

                return True
        return False




class Node_of_tree(Cluster, NodeMixin):  # Extension to class Cluster to use tree structure
    def __init__(self, name, parent=None, children=None):
        super(Node_of_tree, self).__init__()
        self.name = name
        self.parent = parent
        if children:
             self.children = children

    def print(self):
        for pre, fill, node in RenderTree(self):
            print("%s%s %s %s" % ( \
                pre, \
                node.name, \
                node.statistics.dist_dict_coef.get('d1_val'), \
                node.list_timeseries_names() if node.active_cluster else " [NOT ACTIVE]" ))
