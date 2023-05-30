#Author Manigandan
#Description: Remove unwanted edges in a model using conditional indepence test
# Perform Generating PC stable skeleton
import sys
from CreateGraph import CreateGraph as graph
from ConditionalIndependence import ConditionalIndependence as ci
from itertools import permutations
#open source
from causallearn.utils.cit import CIT

#Remove edges and PC stable skelton
class RemovingEdgesPCStable:
    
    rand_vars = []
    rv_values = []
    no_of_nodes = 0
    nodes = []
    parents = []
    parents_of_i = []
    adj = [[]]
    significance_level = 0.05
    structure = [()]
    copy_adj = [[]]

    #read data from train file to pass in conditional independence test
    def __init__(self, filename):
        
        data, self.rand_vars = ci.read_data(ci, filename)
        self.chisq_obj = CIT(data, "chisq")
        self.rv_values = data
        self.no_of_nodes = len(self.rand_vars)
        self.adj = [[] for i in range(self.no_of_nodes)]
        self.copy_adj = [[] for i in range(self.no_of_nodes)]
        pass

    #initiate different type of structure and parents
    def init_structure(self):

        #self.nodes = [[13,5,12,3,11,8,10,6],[8,10,4,6],[3,13,12,10,6],[7,13,0,4,12,6,9,2],[13,3,10,2,6],[13,3,10,2,6],[0,8,11,12],[4,0,10,13,1,2,3,8],[13,10,11,12,8,9,3],[7,12,13,5,0,6],[11,10,7,3],[2,9,13,0,4,7,6,1],[7,13,5,0,9],[7,13,5,0,3,2,8],[7,3,2,12,8,11,10,6,4,0]]
        #self.nodes = [[13,5,12,11,8,10,6],[8,10,6],[3,13,10,6],[7,13,4,12,6,9,2],[13,3,2,6],[13,10,2,6],[0,8,11],[4,0,10,13,1,2],[13,10,11,12,3],[7,12,13,6],[11,1,3],[2,9,13,7,6,1],[7,13,9],[7,13,0,3,2,8],[7,2,12,8,11,10,6,4,0]]
        self.nodes = [[13,5,11,8,10,6],[8,4,6],[3,13,10],[7,13,0,6,9,2],[13,3,2,6],[13,10,2,6],[8,11,12],[4,0,13,1,2,3,8],[13,10,11,12,9,3],[7,12,13,6],[11,10,7,3],[2,13,0,4,7,6],[7,13,5,0,9],[7,13,5,0,8],[7,3,2,12,8,4,0]]
        #self.nodes = [[13,5,11,8,10],[8,4,6,10],[3,13,10],[7,0,6,9,2],[7,13,3,2,6],[13,,2,6],[8,11,12],[4,0,13,2,3,8],[13,10,12,9,3],[7,12,13,6],[11,10,7][2,13,0,4,7,6],[7,13,5,0,9],[7,13,5,0,8],[7,3,2,12,8,4,0]]
        #variables represented in index so it can be easily passed in conditional indepence test 
        self.parents = [[10,13,5],[10,6,4],[13,12,10,6],[7,13,0,4,12,6,9,2],[13],[12],[13,4],[13],[7,12,13,5,0,6,1],[7,10],[13,4,6],[7,13,5,0,9],[7,13],[]]
        no_of_elements = 0 
        for i in self.nodes:
            no_of_elements = no_of_elements + len(i)
        self.structure = [() for i in range(no_of_elements)]
        pass
    #convert edges format in tuple to give input for hill climbing
    def convert_edges_format(self):
        for Curr_node in range(0, self.no_of_nodes):
            for dep_node in self.copy_adj[Curr_node]:
                self.structure[Curr_node] = self.structure[Curr_node] + (self.rand_vars[Curr_node],) 
                if Curr_node == dep_node:
                    continue
                if dep_node not in self.structure[Curr_node]:
                    self.structure[Curr_node] = self.structure[Curr_node] + (self.rand_vars[dep_node],)
            print(self.structure[Curr_node])
        print(self.structure)
        return self.structure

    #add edges for graph
    def add_graph_edges(self, choice):

        if (choice == '2'):
            for Curr_node in range(0, self.no_of_nodes):
                for dep_node in range(0, self.no_of_nodes): 
                    if Curr_node == dep_node:
                        continue
                    if dep_node not in self.adj[Curr_node]:
                        graph.addEdge(graph, self.adj, Curr_node, dep_node)
        elif (choice == '1'):
            for Curr_node in range(0, self.no_of_nodes):
                print("\n")
                for dep_node in self.nodes[Curr_node]: 
                    if Curr_node == dep_node:
                        continue
                    if dep_node not in self.adj[Curr_node]:
                        print(Curr_node, dep_node)
                        graph.addEdge(graph, self.adj, Curr_node, dep_node)
                        self.copy_adj[Curr_node].append(dep_node)
        graph.prGraph(graph, self.adj, self.no_of_nodes, self.rand_vars)
        print("test")
        print(self.structure)
    
    #initiate parent of node
    def parents_of_node(self):
        
        self.parents_of_i = []
        for var in range(self.no_of_nodes):
            row = []
            for val in range(var+1):
                row.append(val)
            self.parents_of_i.append(row)

    #remove edges between node
    def remove_edges_between_nodes(self):

        for Curr_node in range(0, self.no_of_nodes):
            cur_list = sorted(self.nodes[Curr_node])
            for dep_node in cur_list: 
                    vi = Curr_node
                    vj = dep_node
                    cur_parent = self.parents[dep_node].copy()
                    if vi in self.parents[dep_node]:
                        cur_parent.remove(vi)
                    if vj in self.parents[dep_node]:
                        cur_parent.remove(vj)
                    #remove edges between two nodes
                    p = self.chisq_obj(vi, vj, cur_parent) 
                    #if p-value greater than 0.05 remove edges
                    if p >= self.significance_level:
                        print("vi,vj",self.rand_vars[vi],self.rand_vars[vj], cur_parent)
    
    #generate PC stable skeleton
    def PC_stable_skeleton(self):

        for iter in range(self.no_of_nodes-1):
            print("ITER", iter)
            comb = iter+2
            perm = permutations(self.parents_of_i[self.no_of_nodes-1], comb)
            
            if (iter == 0):
                for index in perm:
                    vi = index[0]
                    vj = index[1]
                    parents_i = []
                    p = self.chisq_obj(vi, vj, parents_i)
                    if p >= self.significance_level:
                        graph.delEdge(graph, self.adj, vi, vj)

            for index in perm:
                vi = index[0]
                vj = index[1]
                #parents of Vj increases for every iteration
                parents_i = list(index[2:(iter)+2])
                p = self.chisq_obj(vi, vj, parents_i) 
                if p >= self.significance_level:
                    #delete edges
                    graph.delEdge(graph, self.adj, vi, vj)
                    print(graph.prGraph(graph, self.adj, self.no_of_nodes, self.rand_vars))    
            print(graph.prGraph(graph, self.adj, self.no_of_nodes, self.rand_vars))
        print(graph.prGraph(graph, self.adj, self.no_of_nodes, self.rand_vars))

