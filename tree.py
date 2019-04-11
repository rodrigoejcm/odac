from anytree import NodeMixin, RenderTree
from itertools import combinations_with_replacement
from collections import OrderedDict
import numpy as np
from math import sqrt

class Statistics:

    def __init__(self, ts_quantity):

        self.sum_dict = {}       # key: cluster number
        self.prd_dict = {}       # key: tuple of two cluster numbers
        self.corr_dict = {}      # key: tuple of two cluster numbers
        self.rnomc_dict = {}     # key: tuple of two cluster numbers
        self.cluster_diameter = 0
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
        print("# cluster_diameter = {}".format(self.cluster_diameter))
        print("# sum_dict:")
        print(self.sum_dict)
        print("# prd_dict:")
        print(self.prd_dict)
        print("# corr_dict:")
        print(self.corr_dict)
        print("# rnomc_dict:")
        print(self.rnomc_dict)


class Cluster:

    def __init__(self):
        self.active_cluster = True
        self.statistics = None
        self.child_left = None
        self.child_right = None


    def set_cluster_timeseries(self,list_ts):
        self.list_of_timeseries = OrderedDict(sorted(list_ts.items(), key=lambda t: t[0]))
        cluster_size = len(self.list_of_timeseries)
        self.statistics = Statistics(cluster_size)
        self.update_statistics(init=True) ### does not generate ts samples only calculate matrices


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


    def update_statistics(self , init=False):

        if init == False:
            self.get_new_timeseries_values()

        print("### new observation:", end='')
        for ts in list(self.list_of_timeseries.values()):
            print(", {}".format(ts.current_value), end='')
        print("")

        self.statistics.n_of_instances += 1

        ### calculate Matrice
        self.calcula_sum_dict()

        ### calculate prod matrix
        self.calcula_prod_dict()

        if self.statistics.n_of_instances > 5:

            ### calculate Matrice coor
            self.calcula_corr_dict()

            ### calculate Matrice dif
            self.calcula_rnomc_dict()

        self.statistics.print()


    def get_new_timeseries_values(self):
        for ts in self.list_of_timeseries.values():
            ts.next_val()



class Node_of_tree(Cluster, NodeMixin):  # Extension to class Cluster to use tree structure
    def __init__(self, name, parent=None, children=None):
        super(Node_of_tree, self).__init__()
        self.name = name
        self.parent = parent
        if children:
             self.children = children
