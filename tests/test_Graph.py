"""
tests/test_graph.py – unit tests for Graph.pyx
"""

import unittest

from Altendorf_Jeulin_Model.utils.Graph import Graph, CC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_graph(*edges, extra_vertices=()):
    """Build a Graph from an iterable of (u, v) pairs."""
    g = Graph()
    for u, v in edges:
        g.add_edge(u, v)
    for v in extra_vertices:
        g.add_vertex(v)
    return g


# ---------------------------------------------------------------------------
# Graph.add_vertex
# ---------------------------------------------------------------------------

class TestAddVertex(unittest.TestCase):

    def test_returns_stable_id(self):
        g = Graph()
        id1 = g.add_vertex("a")
        id2 = g.add_vertex("a")
        self.assertEqual(id1, id2)

    def test_different_vertices_different_ids(self):
        g = Graph()
        id_a = g.add_vertex("a")
        id_b = g.add_vertex("b")
        self.assertNotEqual(id_a, id_b)

    def test_ids_are_contiguous(self):
        g = Graph()
        ids = [g.add_vertex(i) for i in range(5)]
        self.assertEqual(sorted(ids), list(range(5)))

    def test_adjacency_list_entry_created(self):
        g = Graph()
        g.add_vertex("x")
        self.assertEqual(len(g.adjacency_list), 1)
        self.assertEqual(g.adjacency_list[0], [])

    def test_roundtrip_id_to_vertex(self):
        g = Graph()
        vid = g.add_vertex("hello")
        self.assertEqual(g.id_to_vertex[vid], "hello")


# ---------------------------------------------------------------------------
# Graph.add_edge
# ---------------------------------------------------------------------------

class TestAddEdge(unittest.TestCase):

    def test_n_edges_incremented(self):
        g = make_graph(("a", "b"), ("b", "c"))
        self.assertEqual(g.n_edges, 2)

    def test_adjacency_symmetric(self):
        g = make_graph(("a", "b"))
        a_id = g.vertex_to_id["a"]
        b_id = g.vertex_to_id["b"]
        self.assertIn(b_id, g.adjacency_list[a_id])
        self.assertIn(a_id, g.adjacency_list[b_id])

    def test_new_vertices_created_implicitly(self):
        g = make_graph(("x", "y"))
        self.assertIn("x", g.vertex_to_id)
        self.assertIn("y", g.vertex_to_id)

    def test_integer_vertices(self):
        g = make_graph((0, 1), (1, 2), (2, 3))
        self.assertEqual(g.n_edges, 3)


# ---------------------------------------------------------------------------
# CC – single connected component
# ---------------------------------------------------------------------------

class TestCCSingleComponent(unittest.TestCase):

    def setUp(self):
        # Triangle: 0-1-2-0
        self.g = make_graph((0, 1), (1, 2), (0, 2))
        self.cc = CC(self.g)

    def test_one_component(self):
        self.assertEqual(self.cc.n_cc, 1)

    def test_all_vertices_assigned(self):
        self.assertTrue(all(v == 0 for v in self.cc.id))

    def test_vertices_returns_all(self):
        verts = set(self.cc.vertices(self.g, 0))
        self.assertEqual(verts, {0, 1, 2})

    def test_edges_count(self):
        # Triangle has 3 edges
        edges = self.cc.edges(self.g, 0)
        self.assertEqual(len(edges), 3)

    def test_edges_are_pairs_of_original_vertices(self):
        edges = self.cc.edges(self.g, 0)
        for u, v in edges:
            self.assertIn(u, {0, 1, 2})
            self.assertIn(v, {0, 1, 2})

    def test_no_duplicate_edges(self):
        edges = self.cc.edges(self.g, 0)
        # Normalise to frozensets to check for duplicates
        normalised = [frozenset((u, v)) for u, v in edges]
        self.assertEqual(len(normalised), len(set(normalised)))


# ---------------------------------------------------------------------------
# CC – multiple connected components
# ---------------------------------------------------------------------------

class TestCCMultipleComponents(unittest.TestCase):

    def setUp(self):
        # Component A: 0-1-2 (path)
        # Component B: 3-4 (single edge)
        # Component C: 5 (isolated vertex)
        self.g = make_graph((0, 1), (1, 2), (3, 4), extra_vertices=(5,))
        self.cc = CC(self.g)

    def test_three_components(self):
        self.assertEqual(self.cc.n_cc, 3)

    def test_component_sizes(self):
        sizes = sorted(
            len(self.cc.vertices(self.g, cid))
            for cid in range(self.cc.n_cc)
        )
        self.assertEqual(sizes, [1, 2, 3])

    def test_isolated_vertex_has_no_edges(self):
        # Find the component containing vertex 5
        cid = next(
            cid for cid in range(self.cc.n_cc)
            if 5 in self.cc.vertices(self.g, cid)
        )
        self.assertEqual(self.cc.edges(self.g, cid), [])

    def test_path_component_has_two_edges(self):
        cid = next(
            cid for cid in range(self.cc.n_cc)
            if set(self.cc.vertices(self.g, cid)) == {0, 1, 2}
        )
        self.assertEqual(len(self.cc.edges(self.g, cid)), 2)

    def test_all_vertices_assigned_to_exactly_one_component(self):
        seen = set()
        for cid in range(self.cc.n_cc):
            verts = set(self.cc.vertices(self.g, cid))
            self.assertTrue(verts.isdisjoint(seen),
                            "vertex appears in multiple components")
            seen |= verts
        self.assertEqual(seen, {0, 1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------
# CC – empty graph
# ---------------------------------------------------------------------------

class TestCCEmpty(unittest.TestCase):

    def test_empty_graph_no_components(self):
        g = Graph()
        cc = CC(g)
        self.assertEqual(cc.n_cc, 0)

    def test_components_returns_empty_dict(self):
        g = Graph()
        cc = CC(g)
        self.assertEqual(cc.components(g), {})


# ---------------------------------------------------------------------------
# CC.vertices / CC.edges – invalid component id
# ---------------------------------------------------------------------------

class TestCCInvalidId(unittest.TestCase):

    def setUp(self):
        self.g = make_graph((0, 1))
        self.cc = CC(self.g)

    def test_vertices_negative_id_raises(self):
        with self.assertRaises(ValueError):
            self.cc.vertices(self.g, -1)

    def test_vertices_too_large_id_raises(self):
        with self.assertRaises(ValueError):
            self.cc.vertices(self.g, self.cc.n_cc)

    def test_edges_negative_id_raises(self):
        with self.assertRaises(ValueError):
            self.cc.edges(self.g, -1)

    def test_edges_too_large_id_raises(self):
        with self.assertRaises(ValueError):
            self.cc.edges(self.g, self.cc.n_cc)


# ---------------------------------------------------------------------------
# CC.components
# ---------------------------------------------------------------------------

class TestCCComponents(unittest.TestCase):

    def test_components_keys_are_component_ids(self):
        g = make_graph((0, 1), extra_vertices=(2,))
        cc = CC(g)
        result = cc.components(g)
        self.assertEqual(set(result.keys()), set(range(cc.n_cc)))

    def test_components_values_cover_all_vertices(self):
        g = make_graph((0, 1), (2, 3), extra_vertices=(4,))
        cc = CC(g)
        all_verts = set()
        for verts in cc.components(g).values():
            all_verts |= set(verts)
        self.assertEqual(all_verts, {0, 1, 2, 3, 4})

    def test_components_no_overlap(self):
        g = make_graph((0, 1), (2, 3))
        cc = CC(g)
        seen = []
        for verts in cc.components(g).values():
            for v in verts:
                self.assertNotIn(v, seen)
                seen.append(v)


# ---------------------------------------------------------------------------
# Regression tests for known bugs
# ---------------------------------------------------------------------------

class TestEdgeDuplication(unittest.TestCase):
    """Regression for Bug 1: CC must not record each edge more than once."""

    def test_path_graph_edge_count(self):
        # 0-1-2-3: exactly 3 edges
        g = make_graph((0, 1), (1, 2), (2, 3))
        cc = CC(g)
        edges = cc.edges(g, 0)
        normalised = [frozenset(e) for e in edges]
        self.assertEqual(len(normalised), len(set(normalised)),
                         "duplicate edges found in component")
        self.assertEqual(len(edges), 3)

    def test_complete_graph_k4_edge_count(self):
        # K4: 6 edges
        vertices = [0, 1, 2, 3]
        g = Graph()
        for i in vertices:
            for j in vertices:
                if i < j:
                    g.add_edge(i, j)
        cc = CC(g)
        edges = cc.edges(g, 0)
        normalised = [frozenset(e) for e in edges]
        self.assertEqual(len(normalised), len(set(normalised)),
                         "duplicate edges found in K4")
        self.assertEqual(len(edges), 6)


if __name__ == "__main__":
    unittest.main()