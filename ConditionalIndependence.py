#############################################################################
# ConditionalIndependence.py
#
# Implements functionality for conditional independence tests via the
# library causal-learn (https://github.com/cmu-phil/causal-learn), which
# can be used to identify edges to keep or remove in a graph given a dataset.
#
# This requires installing the following (at Uni-Lincoln computer labs):
# 1. Type Anaconda Prompt in your Start icon
# 2. Open your terminal as administrator
# 3. Execute=> pip install causal-learn
#
# At the bottom of this file are the USAGE instructions to run this program.
#
# Version: 1.0, Date: 19 October 2022
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
from itertools import permutations
from causallearn.utils.cit import CIT


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []
    verbose = True

    def __init__(self, file_name):
        data = self.read_data(file_name)
        self.chisq_obj = CIT(data, "chisq")

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10])+"\n")
        return self.rv_all_values, self.rand_vars

    def parse_test_args(self, test_args):
        main_args = test_args[2:len(test_args)-1]
        variables = main_args.split('|')[0]
        Vi = variables.split(',')[0]
        Vj = variables.split(',')[1]
        parents_i = []
        for parent in (main_args.split('|')[1].split(',')):
            if parent.lower() == 'none':
                continue
            else:
                parents_i.append(parent)

        return Vi, Vj, parents_i

    def get_var_index(self, target_variable):
        for i in range(0, len(self.rand_vars)):
            if self.rand_vars[i] == target_variable:
                return i
        print("ERROR: Couldn't find index of variable "+str(target_variable))
        return None

    def get_var_indexes(self, parent_variables):
        if len(parent_variables) == 0:
            return None
        else:
            index_vector = []
            for parent in parent_variables:
                index_vector.append(self.get_var_index(parent))
            return index_vector

    def compute_pvalue(self, variable_i, variable_j, parents_i):

        var_i = self.get_var_index(variable_i)
        var_j = self.get_var_index(variable_j)
        par_i = parents_i

        if(self.verbose == False):
            self.verbose = False
            par_i = self.get_var_indexes(parents_i)
            p = self.chisq_obj(var_i, var_j, par_i)
            #print("X2test: Vi=%s, Vj=%s, pa_i=%s, p=%s" % (variable_i, variable_j, parents_i, p))

        p = self.chisq_obj(var_i, var_j, par_i)
        #print("X2test: Vi=%s, Vj=%s, pa_i=%s, p=%s" % (variable_i, variable_j, parents_i, p))
        return p


class Graph:
    def addEdge(self,adj, u, v):

        adj[u].append(v)
        adj[v].append(u)
        
    def delEdge(self, adj, u, v):
        
        for i in range(len(adj[u])):
        
            if (adj[u][i] == v):
                
                adj[u].pop(i)
                break

        for i in range(len(adj[v])):
        
            if (adj[v][i] == u):
                
                adj[v].pop(i)
                break
        
    def prGraph(self, adj, V):
        
        for v in range(V):
            
            print("vertex " + rand_vars[v], end = ' ')
            
            for x in adj[v]:
                print("->" , rand_vars[x], end = '')
                
            print()
        print()
        
'''
if len(sys.argv) != 3:
    print("USAGE: ConditionalIndepencence.py [train_file.csv] [I(Vi,Vj|parents)]")
    print("EXAMPLE1: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(O,PT|None)'")
    print("EXAMPLE2: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(PT,T|O)'")
    print("EXAMPLE3: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(PT,T|O,H)'")
    exit(0)
else:
    data_file = sys.argv[1]
    test_args = sys.argv[2]

    ci = ConditionalIndependence(data_file)
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    #print(parents_i)
    #print("Type", type(parents_i))
    #ci.compute_pvalue(Vi, Vj, parents_i)

    rand_vars_copy = ci.rand_vars
    rand_vars = ci.rand_vars
    graph = Graph()
    V = 14
    significance_level = 0.05
    adj = [[] for i in range(V)]
    no_of_nodes = 14
    rand_vars.append(rand_vars_copy[(no_of_nodes-1)])

    nodes = [[13,5,12,3,11,8,10,6],[8,10,4,6],[3,13,12,10,6],[7,13,0,4,12,6,9,2],[13,3,10,2,6],[13,3,10,2,6],[0,8,11,12],[4,0,10,13,1,2,3,8],[13,10,11,12,8,9,3],[7,12,13,5,0,6],[11,10,7,3],[2,9,13,0,4,7,6,1],[7,13,5,0,9],[7,13,5,0,3,2,8],[7,3,2,12,8,11,10,6,4,0]]
    parents = [[10,13,5],[10,6,4],[13,12,10,6],[7,13,0,4,12,6,9,2],[13],[12],[13,4],[13],[7,12,13,5,0,6,1],[7,10],[13,4,6],[7,13,5,0,9],[7,13],[]]

    for Curr_node in range(0, no_of_nodes):
        cur_list = sorted(nodes[Curr_node])
        for dep_node in cur_list: 
                graph.addEdge(adj, Curr_node, dep_node)
    
    for Curr_node in range(0, no_of_nodes):
        cur_list = sorted(nodes[Curr_node])
        for dep_node in cur_list: 
                vi = Curr_node
                vj = dep_node
                cur_parent = parents[dep_node].copy()
                #print(vj, cur_parent)
                if vi in parents[dep_node]:
                    cur_parent.remove(vi)
                if vj in parents[dep_node]:
                    cur_parent.remove(vj)
                p = ci.compute_pvalue(rand_vars[vi], rand_vars[vj], cur_parent)                
                if p >= significance_level:
                    print("vi,vj",rand_vars[vi],rand_vars[vj], cur_parent)
                    graph.delEdge(adj, vi, vj)
    #print(graph.prGraph(adj, V))
    exit(0)

    Multiple Compination Stable Algorithm
    for i in range(1,no_of_nodes):
        rand_vars.append(rand_vars_copy[i-1])

    rand_vars = ci.rand_vars
    for Curr_node in range(0, no_of_nodes):
        for dep_node in range(0, no_of_nodes): 
            if Curr_node == dep_node:
                continue
            if dep_node not in adj[Curr_node]:
                graph.addEdge(adj, Curr_node, dep_node);
    
    print(graph.prGraph(adj, V))
    parents_of_i = []
    for var in range(no_of_nodes):
        row = []
        for val in range(var+1):
            row.append(val)
        parents_of_i.append(row)

    graph.prGraph(adj, V);

    #perm = permutations([0,1,2,3,4])
    print(parents_of_i)
    for iter in range(no_of_nodes-1):
        print("ITER", iter)
        comb = iter+2
        perm = permutations(parents_of_i[no_of_nodes-1], comb)
        if (iter == 0):
            for index in perm:
                vi = index[0]
                vj = index[1]
                p = ci.compute_pvalue(rand_vars[vi], rand_vars[vj], parents_i)
                if p >= significance_level:
                    graph.delEdge(adj, vi, vj)
                    print(graph.prGraph(adj, V))

            for index in perm:
            vi = index[0]
            vj = index[1]
            parents_i = list(index[2:(iter)+2])
            #print("vi,vj,parents, iter",vi, vj, parents_i, iter)
            p = ci.compute_pvalue(rand_vars[vi], rand_vars[vj], parents_i)
            #print(p)
            if p >= significance_level:
                graph.delEdge(adj, vi, vj)
                #print(graph.prGraph(adj, V))
        print(graph.prGraph(adj, V))

    print(graph.prGraph(adj, V))
    print(ci.rand_vars)
    print(rand_vars)'''


    #p = ci.compute_pvalue('O','PT', parents_i)