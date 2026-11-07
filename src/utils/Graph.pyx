# cython: language_level=3, infer_type=True
import cython

class Graph:
    """
    Undirected graph with hashable vertex labels.

    Vertices are stored with an integer ID assigned in insertion order.
    Edges are stored as adjacency lists.

    Attributes:
        n_edges (int): number of edges in the graph
        vertex_to_id (dict): mapping from vertex label to integer ID
        id_to_vertex (dict): mapping from integer ID to vertex label
        adjacency_list (list[list[int]]): adjacency list indexed by vertex ID
    """
    def __init__(self):
        self.n_edges = 0
        self.vertex_to_id = {}
        self.id_to_vertex = {}
        self.adjacency_list = []

    def add_vertex(self, vertex):
        """
        Add a vertex to the graph if it is not already present.

        :param vertex: hashable
            label of the vertex to add
        :return: int
            integer ID of the vertex (existing or newly assigned)
        """
        if vertex not in self.vertex_to_id:
            vertex_id = len(self.adjacency_list)

            self.vertex_to_id[vertex] = vertex_id
            self.id_to_vertex[vertex_id] = vertex
            self.adjacency_list.append([])

        return self.vertex_to_id[vertex]

    def add_edge(self, u, v):
        """
        Add an undirected edge between vertices u and v.

        Both vertices are created if they do not already exist.
        The edge is recorded in both adjacency lists and n_edges is
        incremented.

        :param u: hashable
            label of the first endpoint
        :param v: hashable
            label of the second endpoint
        """
        u_id = self.add_vertex(u)
        v_id = self.add_vertex(v)

        self.adjacency_list[u_id].append(v_id)
        self.adjacency_list[v_id].append(u_id)
        self.n_edges += 1


class CC:
    """
    Connected-components decomposition of an undirected Graph.

    Computed once at construction via iterative DFS.  Results are cached
    and accessed through vertices(), edges(), and components().

    Attributes:
        n_cc (int): number of connected components
        id (list[int]): component ID for each vertex ID; id[v] is the
            component that vertex v belongs to
        vertices_by_component (list[list[int]]): vertex ID lists,
            one per component
        edges_by_component (list[list[tuple[int,int]]]): edge lists as
            (u_id, v_id) pairs with u_id < v_id, one per component
    """
    def __init__(self, graph):
        """
        Compute the connected components of *graph*.

        Runs an iterative DFS from each unvisited vertex.  Time and space
        complexity are O(V + E).

        :param graph: Graph
            the graph to decompose
        """
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
        """
        Return the vertex labels belonging to a connected component.

        :param graph: Graph
            the graph that was decomposed
        :param component_id: int
            component index in [0, n_cc)
        :return: list
            vertex labels of the component in DFS discovery order
        :raises ValueError:
            if component_id is out of range
        """
        if not 0 <= component_id < self.n_cc:
            raise ValueError(f"Invalid component ID: {component_id}")

        return [
            graph.id_to_vertex[vertex_id]
            for vertex_id in self.vertices_by_component[component_id]
        ]

    def edges(self, graph, component_id):
        """
        Return the edges belonging to a connected component.

        Each undirected edge is returned exactly once as a pair of vertex
        labels (u, v).

        :param graph: Graph
            the graph that was decomposed
        :param component_id: int
            component index in [0, n_cc)
        :return: list[tuple]
            edge pairs (u_label, v_label) of the component
        :raises ValueError:
            if component_id is out of range
        """
        if not 0 <= component_id < self.n_cc:
            raise ValueError(f"Invalid component ID: {component_id}")

        return [
            (graph.id_to_vertex[u], graph.id_to_vertex[v])
            for u, v in self.edges_by_component[component_id]
        ]

    def components(self, graph):
        """
        Return all connected components as a dictionary.

        :param graph: Graph
            the graph that was decomposed
        :return: dict[int, list]
            mapping from component ID to list of vertex labels in that
            component, in DFS discovery order
        """
        return {
            component_id: [
                graph.id_to_vertex[vertex_id]
                for vertex_id in vertex_ids
            ]
            for component_id, vertex_ids in enumerate(self.vertices_by_component)
        }
