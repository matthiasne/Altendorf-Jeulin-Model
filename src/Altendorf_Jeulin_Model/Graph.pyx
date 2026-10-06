# cython: language_level=3, infer_type=True
import cython
from networkx.algorithms.traversal import breadth_first_search
from collections import defaultdict


class Graph:
    def __init__(self):
        self.n_edges = 0
        self.vertex_to_id = {}
        self.id_to_vertex = {}
        self.adjacency_list = []

    def add_vertex(self, vertex):
        if vertex not in self.vertex_to_id:
            vertex_id = len(self.adjacency_list)

            self.vertex_to_id[vertex] = vertex_id
            self.id_to_vertex[vertex_id] = vertex
            self.adjacency_list.append([])

        return self.vertex_to_id[vertex]

    def add_edge(self, u, v):
        u_id = self.add_vertex(u)
        v_id = self.add_vertex(v)

        self.adjacency_list[u_id].append(v_id)
        self.adjacency_list[v_id].append(u_id)
        self.n_edges += 1


class CC:
    def __init__(self, graph):
        self.marked = [False] * len(graph.adjacency_list)
        self.id = [-1] * len(graph.adjacency_list)
        self.n_cc = 0

        n_vertices = len(graph.adjacency_list)
        for vertex in range(n_vertices):
            if not self.marked[vertex]:
                self.depth_first_search(vertex, graph)
                self.n_cc += 1

    def depth_first_search(self, start, graph):
        stack = [start]
        self.marked[start] = True
        self.id[start] = self.n_cc

        while stack:
            vertex = stack.pop()

            for neighbor in graph.adjacency_list[vertex]:
                if not self.marked[neighbor]:
                    self.marked[neighbor] = True
                    self.id[neighbor] = self.n_cc
                    stack.append(neighbor)

    def components(self, graph):
        result = defaultdict(list)

        for vertex, vertex_id in graph.vertex_to_id.items():
            component_id = self.id[vertex_id]
            result[component_id].append(vertex)

        return dict(result)

    def vertices(self, graph, component_id):
        result = list()
        for vertex, vertex_id in graph.vertex_to_id.items():
            if self.id[vertex_id] != component_id:
                continue
            result.append(vertex)

        return result

    def edges(self, graph, component_id):
        if component_id < 0 or component_id >= self.n_cc:
            raise ValueError(f"Invalid component ID: {component_id}")

        result = set()

        for u_id, neighbors in enumerate(graph.adjacency_list):
            if self.id[u_id] != component_id:
                continue

            for v_id in neighbors:
                if self.id[v_id] == component_id:
                    u = graph.id_to_vertex[u_id]
                    v = graph.id_to_vertex[v_id]
                    result.add((u, v))
                    result.add((v, u))

        return result