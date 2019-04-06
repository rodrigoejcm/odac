from anytree import NodeMixin, RenderTree
from itertools import combinations_with_replacement
from collections import OrderedDict
import numpy as np
from math import sqrt


class Cluster:

    sum_matrix = None
    prod_matrix = None
    coor_matrix = None
    rnomc_matrix = None
    list_of_calculations = None
    cluster_diameter = 0
    n_of_instances = 0

    def __init__(self):
        self.active_cluster = True
        #self.list_of_timeseries = 

    def set_cluster_timeseries(self,list_ts):
        self.list_of_timeseries = OrderedDict(sorted(list_ts.items(), key=lambda t: t[0]))
        cluster_size = len(self.list_of_timeseries)

        #### initialize matrices
        
        self.list_of_calculations = list(combinations_with_replacement(list(range(0,cluster_size)), 2))

        self.sum_matrix = np.zeros(shape=(cluster_size,cluster_size))
        self.prod_matrix = np.zeros(shape=(cluster_size,cluster_size))
        self.coor_matrix = np.zeros(shape=(cluster_size,cluster_size))
        self.rnomc_matrix = np.zeros(shape=(cluster_size,cluster_size))


        self.update_statistics(init=True) ### does not generate ts samples only calculate matrices
        

    def get_cluster_timeseries(self):
        return self.list_of_timeseries

    
    def list_timeseries_names(self):
        return list(self.list_of_timeseries.keys())

    
    def calcula_sum_matrix(self,init=False):
        
        for comb in self.list_of_calculations:
            i = comb[0]
            j = comb[1]
            if i == j:           
                self.sum_matrix[i,j] = self.sum_matrix[i,j] \
                                    + list(self.list_of_timeseries.values())[i].current_value 
                                   
        return self.sum_matrix

    def calcula_prod_matrix(self,init=False):
        
        for comb in self.list_of_calculations:
            i = comb[0]
            j = comb[1]
           
            self.prod_matrix[i,j] = self.prod_matrix[i,j] \
                                    + ( list(self.list_of_timeseries.values())[i].current_value \
                                    * list(self.list_of_timeseries.values())[j].current_value ) 
        return self.prod_matrix

    def calcula_coor_matrix(self,init=False):
        
        for comb in self.list_of_calculations:
            i = comb[0]
            j = comb[1]

            if i != j:    
                p = self.prod_matrix[i,j]
                a = self.sum_matrix[i,i]
                a2 = self.prod_matrix[i,i]
                b = self.sum_matrix[j,j]
                b2 = self.prod_matrix[j,j]
                n = self.n_of_instances

                #print(p,a,a2,b,b2,n)


                term_p = p - ((a*b)/n)
               # print(p," - ((" , a,"*", b,")/",n,")")
                term_a = sqrt(a2 - ((a*a)/n))
                term_b = sqrt(b2 - ((b*b)/n))
                #print(term_p,term_a,term_b)

                self.coor_matrix[i,j] = term_p/(term_a*term_b)
        
        return self.coor_matrix


    def calcula_rnomc_matrix(self,init=False):
        
        list_of_rnorms = [] 
        ### the highest value from list will be the diameter , set at the end


        for comb in self.list_of_calculations:
            i = comb[0]
            j = comb[1]

            if i != j:
                coor_a_b = self.coor_matrix[i,j]    
                self.rnomc_matrix[i,j] = sqrt( (1-coor_a_b) / 2 )
                list_of_rnorms.append(self.rnomc_matrix[i,j])


        self.cluster_diameter = max(list_of_rnorms)
        return self.rnomc_matrix


    
    
    
    def update_statistics(self , init=False):

        if init == False:
            self.get_new_timeseries_values()

        self.n_of_instances += 1

        ### calculate Matrice 
        self.calcula_sum_matrix()
        print(" --> SUM MATRIX")
        print(self.sum_matrix)

        ### calculate prod matrix
        self.calcula_prod_matrix()
        print(" --> PROD MATRIX")
        print(self.prod_matrix)

        if self.n_of_instances > 5:

            ### calculate Matrice coor
            self.calcula_coor_matrix()
            print(" --> CORR MATRIX")
            print(self.coor_matrix)

            ### calculate Matrice dif
            self.calcula_rnomc_matrix()
            print(" --> RNORM MATRIX")
            print(self.rnomc_matrix)
            print(" ----> CLUSTER DIAMETER")
            print(self.cluster_diameter)
            
    
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