#Author: GeeksForGeeks
class CreateGraph:
    #add edges betwwen nodes
    def addEdge(self,adj, u, v):
        
        adj[u].append(v)
        adj[v].append(u)
    #remove edges between nodes
    def delEdge(self, adj, u, v):
        
        for i in range(len(adj[u])):
            if (adj[u][i] == v):
                adj[u].pop(i)
                break

        for i in range(len(adj[v])):
            if (adj[v][i] == u):
                adj[v].pop(i)
                break
    #print graph
    def prGraph(self, adj, V, rand_vars):
        
        for v in range(V):
            print("vertex " + rand_vars[v], end = ' ')
            for x in adj[v]:
                print("->" , rand_vars[x], end = '')
            print()
        print()
        return adj
#creategraph
CreateGraph()