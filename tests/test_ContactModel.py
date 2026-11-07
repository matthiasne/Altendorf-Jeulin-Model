"""
tests/test_contact_model.py – unit tests for ContactModel.pyx
"""

import unittest
import numpy as np

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.ContactModel import (
    find_contact_pairs,
    find_contact_areas,
    find_contact_candidates,
    test_in_contact,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_ball(coord, radius=1.0, fiber_label=0, ball_label=0):
    return Ball(np.array(coord, dtype=np.float64), radius,
                fiber_label=fiber_label, ball_label=ball_label)


def make_fiber(*coords, radius=1.0, fiber_label=0):
    """Build a Fiber from a list of coordinate triples."""
    balls = [
        make_ball(c, radius=radius, fiber_label=fiber_label, ball_label=i)
        for i, c in enumerate(coords)
    ]
    f = Fiber(balls[0])
    for b in balls[1:]:
        f.add_ball(b)
    return f


IMAGE = np.array([50, 50, 50])


# ---------------------------------------------------------------------------
# test_in_contact
# ---------------------------------------------------------------------------

class TestTestInContact(unittest.TestCase):

    def test_touching_balls_different_fibers_in_contact(self):
        # Two balls of radius 1 whose centers are exactly 2 apart → surface distance = 0
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([2, 0, 0], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, dist = test_in_contact(b1, b2, IMAGE, True, contact_distance=0.0)
        self.assertTrue(is_contact)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_overlapping_balls_in_contact(self):
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([1, 0, 0], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, dist = test_in_contact(b1, b2, IMAGE, True, contact_distance=0.0)
        self.assertTrue(is_contact)
        self.assertLess(dist, 0.0)

    def test_separated_balls_not_in_contact(self):
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([5, 0, 0], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, _ = test_in_contact(b1, b2, IMAGE, True, contact_distance=0.0)
        self.assertFalse(is_contact)

    def test_contact_distance_extends_range(self):
        # Surface distance = 1.0; contact_distance = 1.5 → in contact
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([3, 0, 0], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, _ = test_in_contact(b1, b2, IMAGE, True, contact_distance=1.5)
        self.assertTrue(is_contact)

    def test_same_fiber_not_in_contact(self):
        # Balls on the same fiber are never counted as contacts
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([2, 0, 0], radius=1.0, fiber_label=0, ball_label=1)
        is_contact, weight = test_in_contact(b1, b2, IMAGE, True, contact_distance=0.0)
        self.assertFalse(is_contact)
        # Regression for Bug 1: weight must be a valid float, not garbage
        self.assertIsInstance(weight, float)
        self.assertTrue(np.isfinite(weight))

    def test_periodic_wrapping(self):
        # Ball near x=0 and ball near x=50 should be close periodically
        b1 = make_ball([1, 25, 25], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([49, 25, 25], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, dist = test_in_contact(b1, b2, IMAGE, True, contact_distance=1.0)
        self.assertTrue(is_contact)

    def test_nonperiodic_no_wrapping(self):
        # Same pair as above but non-periodic → far apart → not in contact
        b1 = make_ball([1, 25, 25], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([49, 25, 25], radius=1.0, fiber_label=1, ball_label=0)
        is_contact, _ = test_in_contact(b1, b2, IMAGE, False, contact_distance=1.0)
        self.assertFalse(is_contact)

    def test_nonperiodic_same_fiber_returns_finite_weight(self):
        # Regression for Bug 1 (nonperiodic path returns -1 explicitly)
        b1 = make_ball([0, 0, 0], radius=1.0, fiber_label=0, ball_label=0)
        b2 = make_ball([2, 0, 0], radius=1.0, fiber_label=0, ball_label=1)
        is_contact, weight = test_in_contact(b1, b2, IMAGE, False, contact_distance=0.0)
        self.assertFalse(is_contact)
        self.assertEqual(weight, -1.0)


# ---------------------------------------------------------------------------
# find_contact_pairs
# ---------------------------------------------------------------------------

class TestFindContactPairs(unittest.TestCase):

    def test_touching_fibers_produce_pair(self):
        f0 = make_fiber([0, 25, 25], [1, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([3, 25, 25], [4, 25, 25], radius=1.0, fiber_label=1)
        pairs = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                   contact_distance=0.5)
        self.assertGreater(len(pairs), 0)

    def test_distant_fibers_no_pairs(self):
        f0 = make_fiber([0, 0, 0], [1, 0, 0], radius=1.0, fiber_label=0)
        f1 = make_fiber([20, 20, 20], [21, 20, 20], radius=1.0, fiber_label=1)
        pairs = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                   contact_distance=0.0)
        self.assertEqual(len(pairs), 0)

    def test_same_fiber_balls_excluded(self):
        # A single fiber's own balls must never appear as contact pairs
        f0 = make_fiber([0, 25, 25], [2, 25, 25], [4, 25, 25], radius=1.0, fiber_label=0)
        pairs = find_contact_pairs([f0], IMAGE, is_periodic=False,
                                   contact_distance=10.0)
        self.assertEqual(len(pairs), 0)

    def test_pair_format_tuples(self):
        f0 = make_fiber([0, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([2, 25, 25], radius=1.0, fiber_label=1)
        pairs = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                   contact_distance=0.5)
        for pair in pairs:
            self.assertEqual(len(pair), 2)
            self.assertEqual(len(pair[0]), 2)   # (fiber_label, ball_label)
            self.assertEqual(len(pair[1]), 2)

    def test_weighted_pairs_have_three_elements(self):
        f0 = make_fiber([0, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([2, 25, 25], radius=1.0, fiber_label=1)
        pairs = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                   contact_distance=0.5, is_weighted=True)
        for pair in pairs:
            self.assertEqual(len(pair), 3)
            self.assertIsInstance(pair[2], float)

    def test_periodic_finds_wrapped_pair(self):
        f0 = make_fiber([1, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([49, 25, 25], radius=1.0, fiber_label=1)
        pairs_periodic = find_contact_pairs([f0, f1], IMAGE, is_periodic=True,
                                            contact_distance=1.0)
        pairs_nonperiodic = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                               contact_distance=1.0)
        self.assertGreater(len(pairs_periodic), len(pairs_nonperiodic))

    def test_no_duplicate_pairs(self):
        f0 = make_fiber([0, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([2, 25, 25], radius=1.0, fiber_label=1)
        pairs = find_contact_pairs([f0, f1], IMAGE, is_periodic=False,
                                   contact_distance=0.5)
        # Each pair should appear at most once
        self.assertEqual(len(pairs), len(set(pairs)))


# ---------------------------------------------------------------------------
# find_contact_areas
# ---------------------------------------------------------------------------

class TestFindContactAreas(unittest.TestCase):

    def _two_touching_fibers(self):
        f0 = make_fiber([0, 25, 25], [1, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([3, 25, 25], [4, 25, 25], radius=1.0, fiber_label=1)
        return [f0, f1]

    def test_returns_four_values(self):
        fs = self._two_touching_fibers()
        result = find_contact_areas(fs, IMAGE, is_periodic=False, contact_distance=0.5)
        self.assertEqual(len(result), 4)

    def test_no_contacts_all_zero(self):
        f0 = make_fiber([0, 0, 0], radius=1.0, fiber_label=0)
        f1 = make_fiber([20, 20, 20], radius=1.0, fiber_label=1)
        n_cc, n_clots, n_areas, surface = find_contact_areas(
            [f0, f1], IMAGE, is_periodic=False, contact_distance=0.0
        )
        self.assertEqual(n_areas, 0)
        self.assertEqual(n_clots, 0)
        self.assertEqual(surface, 0)

    def test_connected_components_at_least_one(self):
        fs = self._two_touching_fibers()
        n_cc, _, _, _ = find_contact_areas(
            fs, IMAGE, is_periodic=False, contact_distance=0.5
        )
        self.assertGreaterEqual(n_cc, 1)

    def test_contact_surface_positive(self):
        fs = self._two_touching_fibers()
        _, _, _, surface = find_contact_areas(
            fs, IMAGE, is_periodic=False, contact_distance=0.5
        )
        self.assertGreater(surface, 0)


# ---------------------------------------------------------------------------
# find_contact_candidates
# ---------------------------------------------------------------------------

class TestFindContactCandidates(unittest.TestCase):

    def test_returns_edges(self):
        f0 = make_fiber([0, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([2, 25, 25], radius=1.0, fiber_label=1)
        edges = find_contact_candidates([f0, f1], IMAGE,
                                        is_periodic=False,
                                        interaction_distance=0.5)
        # Should return an edge view or iterable
        self.assertIsNotNone(edges)

    def test_distant_fibers_no_candidates(self):
        f0 = make_fiber([0, 0, 0], radius=1.0, fiber_label=0)
        f1 = make_fiber([30, 30, 30], radius=1.0, fiber_label=1)
        edges = list(find_contact_candidates([f0, f1], IMAGE,
                                             is_periodic=False,
                                             interaction_distance=0.5))
        self.assertEqual(len(edges), 0)

    def test_no_self_edges(self):
        """No ball should be a contact candidate with itself."""
        f0 = make_fiber([5, 25, 25], [7, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([9, 25, 25], [11, 25, 25], radius=1.0, fiber_label=1)
        edges = list(find_contact_candidates([f0, f1], IMAGE,
                                             is_periodic=False,
                                             interaction_distance=1.0))
        for u, v in edges:
            self.assertNotEqual(u, v)

    def test_isolated_node_does_not_raise(self):
        """Regression for Bug 6: isolated node must not raise ValueError."""
        f0 = make_fiber([5, 25, 25], radius=1.0, fiber_label=0)
        f1 = make_fiber([25, 25, 25], radius=1.0, fiber_label=1)  # far away
        f2 = make_fiber([7, 25, 25], radius=1.0, fiber_label=2)
        try:
            find_contact_candidates([f0, f1, f2], IMAGE,
                                    is_periodic=False,
                                    interaction_distance=0.5)
        except ValueError as e:
            self.fail(f"find_contact_candidates raised ValueError: {e}")


if __name__ == "__main__":
    unittest.main()