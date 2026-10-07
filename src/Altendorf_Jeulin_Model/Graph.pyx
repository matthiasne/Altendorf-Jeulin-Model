# cython: language_level=3, infer_type=True
import cython

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
        n = len(graph.adjacency_list)

        self.id = [-1] * n
        self.vertices_by_component = []
        self.edges_by_component = []

        for start in range(n):
            if self.id[start] != -1:
                continue

            component_id = len(self.vertices_by_component)
            component_vertices = []
            component_edges = []

            stack = [start]
            self.id[start] = component_id

            while stack:
                u = stack.pop()
                component_vertices.append(u)

                for v in graph.adjacency_list[u]:
                    if self.id[v] == -1:
                        self.id[v] = component_id
                        stack.append(v)

                    # Store each undirected edge only once.
                    if u < v and self.id[v] == component_id:
                        component_edges.append((u, v))

            self.vertices_by_component.append(component_vertices)
            self.edges_by_component.append(component_edges)

        self.n_cc = len(self.vertices_by_component)

    def vertices(self, graph, component_id):
        if not 0 <= component_id < self.n_cc:
            raise ValueError(f"Invalid component ID: {component_id}")

        return [
            graph.id_to_vertex[vertex_id]
            for vertex_id in self.vertices_by_component[component_id]
        ]

    def edges(self, graph, component_id):
        if not 0 <= component_id < self.n_cc:
            raise ValueError(f"Invalid component ID: {component_id}")

        return [
            (graph.id_to_vertex[u], graph.id_to_vertex[v])
            for u, v in self.edges_by_component[component_id]
        ]

    def components(self, graph):
        return {
            component_id: [
                graph.id_to_vertex[vertex_id]
                for vertex_id in vertex_ids
            ]
            for component_id, vertex_ids in enumerate(self.vertices_by_component)
        }
