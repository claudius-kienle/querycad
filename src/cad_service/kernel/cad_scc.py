# Kosaraju's algorithm to find strongly connected components in Python


from collections import defaultdict
from typing import List


# https://www.programiz.com/dsa/strongly-connected-components
class Graph:

    def __init__(self, vertex):
        self.V = vertex
        self.graph = defaultdict(list)

    # Add edge into the graph
    def add_edge(self, s, d):
        self.graph[s].append(d)

    # dfs
    def dfs(self, d, visited_vertex) -> List[int]:
        visited_vertex[d] = True
        # print(d, end='')
        nodes = [d]
        for i in self.graph[d]:
            if not visited_vertex[i]:
                child_nodes = self.dfs(i, visited_vertex)
                nodes += child_nodes
        return nodes

    def fill_order(self, d, visited_vertex, stack):
        visited_vertex[d] = True
        for i in self.graph[d]:
            if not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)
        stack = stack.append(d)

    # transpose the matrix
    def transpose(self):
        g = Graph(self.V)

        for i in self.graph:
            for j in self.graph[i]:
                g.add_edge(j, i)
        return g

    def get_sccs(self) -> List[List[int]]:
        stack = []
        visited_vertex = [False] * (self.V)

        for i in range(self.V):
            if not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)

        gr = self.transpose()

        visited_vertex = [False] * (self.V)

        sccs = []

        while stack:
            i = stack.pop()
            if not visited_vertex[i]:
                scc = gr.dfs(i, visited_vertex)
                sccs.append(scc)

        return sccs
