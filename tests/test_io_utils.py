import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils.io_utils import (
    find_next_node,
    print_contact_stats,
    print_fiber_positions_to_file,
    print_stats,
    print_stats_row,
    read_gad,
    save_fibers_as_graph,
    save_fibers_as_image,
    save_fibers_as_small_graph,
    save_lines_as_image,
    write_gad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ball(x, y, z, radius=5.0, fiber_label=0, ball_label=0):
    return Ball(np.array([x, y, z], dtype=float), radius, fiber_label, ball_label)


def _straight_fiber(n=5, radius=5.0, axis="x", offset=50.0, length=100.0):
    """Fiber with n balls equally spaced along one axis."""
    fiber = None
    for i, t in enumerate(np.linspace(0, length, n)):
        if axis == "x":
            coord = [offset + t, 50.0, 50.0]
        elif axis == "y":
            coord = [50.0, offset + t, 50.0]
        else:
            coord = [50.0, 50.0, offset + t]
        ball = _make_ball(*coord, radius=radius, fiber_label=0, ball_label=i)
        if fiber is None:
            fiber = Fiber(ball)
        else:
            fiber.add_ball(ball)
    return fiber


def _curved_fiber(n=10, radius=5.0):
    """Fiber that curves gently in the x-y plane."""
    fiber = None
    for i in range(n):
        angle = i * np.pi / (2 * (n - 1))
        ball = _make_ball(50 + 20 * np.cos(angle), 50 + 20 * np.sin(angle), 50.0,
                          radius=radius, fiber_label=0, ball_label=i)
        if fiber is None:
            fiber = Fiber(ball)
        else:
            fiber.add_ball(ball)
    return fiber


# ---------------------------------------------------------------------------
# find_next_node
# ---------------------------------------------------------------------------

class TestFindNextNode(unittest.TestCase):

    def test_straight_fiber_advances_by_one(self):
        # perfectly straight fiber: curvature is 0, so i advances each call
        fiber = _straight_fiber(n=10)
        result = find_next_node(fiber, 0, len(fiber.balls) - 1)
        # curvature stays 0 < 0.2, so loop runs until i+2 > i_end
        self.assertGreater(result, 0)

    def test_returns_i_start_plus_one_for_two_ball_segment(self):
        # when only one step ahead is available (i+2 == i_end), must return i+1
        fiber = _straight_fiber(n=3)
        result = find_next_node(fiber, 0, 2)
        self.assertEqual(result, 1)

    def test_curved_fiber_stops_earlier_than_straight(self):
        straight = _straight_fiber(n=10)
        curved = _curved_fiber(n=10)
        i_end = 9
        straight_result = find_next_node(straight, 0, i_end)
        curved_result = find_next_node(curved, 0, i_end)
        self.assertLessEqual(curved_result, straight_result)

    def test_antiparallel_does_not_produce_nan(self):
        # two balls pointing back — dot product near -1, arccos domain risk
        fiber = Fiber(_make_ball(50, 50, 50))
        fiber.add_ball(_make_ball(60, 50, 50))
        fiber.add_ball(_make_ball(50, 50, 50))  # reverses direction
        result = find_next_node(fiber, 0, 2)
        self.assertFalse(np.isnan(result))
        self.assertIsInstance(result, int)

    def test_result_always_greater_than_i_start(self):
        for n in [3, 5, 10]:
            fiber = _straight_fiber(n=n)
            result = find_next_node(fiber, 0, n - 1)
            self.assertGreater(result, 0)

    def test_result_never_exceeds_i_end(self):
        fiber = _straight_fiber(n=8)
        i_end = len(fiber.balls) - 1
        result = find_next_node(fiber, 0, i_end)
        self.assertLessEqual(result, i_end)


# ---------------------------------------------------------------------------
# print_fiber_positions_to_file
# ---------------------------------------------------------------------------

class TestPrintFiberPositionsToFile(unittest.TestCase):

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "fibers.txt")
            fs = [_straight_fiber(n=3)]
            print_fiber_positions_to_file(fs, path)
            self.assertTrue(Path(path).exists())

    def test_one_line_per_fiber(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "fibers.txt")
            fs = [_straight_fiber(n=3), _straight_fiber(n=4)]
            print_fiber_positions_to_file(fs, path)
            lines = Path(path).read_text().splitlines()
            self.assertEqual(len(lines), 2)

    def test_empty_fiber_system(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "fibers.txt")
            print_fiber_positions_to_file([], path)
            self.assertEqual(Path(path).read_text(), "")


# ---------------------------------------------------------------------------
# print_stats / print_contact_stats
# ---------------------------------------------------------------------------

class TestPrintStats(unittest.TestCase):

    def test_header_and_rows_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "stats.csv")
            rows = [[0, 10, 0.12, 0.05, 150.0, 0.3, 0.01, 0.0, 0.001],
                    [1, 10, 0.13, 0.04, 150.0, 0.2, 0.00, 0.0, 0.0005]]
            print_stats(path, rows)
            with open(path, newline="") as f:
                reader = list(csv.reader(f))
            self.assertEqual(len(reader), 3)  # header + 2 rows
            self.assertIn("Step", reader[0])
            self.assertEqual(reader[1][0], "0")
            self.assertEqual(reader[2][0], "1")

    def test_expected_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "stats.csv")
            print_stats(path, [])
            with open(path, newline="") as f:
                header = next(csv.reader(f))
            expected = ["Step", "#Fibers", "VF", "MAE", "meanLength",
                        "KL", "MaxOverlap", "ContactDistances", "change"]
            self.assertEqual(header, expected)


class TestPrintContactStats(unittest.TestCase):

    def test_header_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "contact.csv")
            rows = [[3, 1, 5, 12.4]]
            print_contact_stats(path, rows)
            with open(path, newline="") as f:
                reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)
            self.assertIn("#connected_components", reader[0])
            self.assertEqual(reader[1][0], "3")


# ---------------------------------------------------------------------------
# save_fibers_as_image
# ---------------------------------------------------------------------------

class TestSaveFibersAsImage(unittest.TestCase):

    def test_saves_tif(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.tif")
            fs = [_straight_fiber(n=5, radius=5, offset=45, length=10)]
            save_fibers_as_image(fs, (100, 100, 100), path=path, is_periodic=True)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_tif_has_correct_shape(self):
        import tifffile
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.tif")
            fs = [_straight_fiber(n=5, radius=5, offset=45, length=10)]
            save_fibers_as_image(fs, (50, 60, 70), path=path, is_periodic=True)
            img = tifffile.imread(path)
            # after transpose (2,1,0): written as (z,y,x) = (70,60,50)
            self.assertEqual(img.shape, (70, 60, 50))

    def test_nonperiodic_saves(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.tif")
            fs = [_straight_fiber(n=5, radius=5, offset=45, length=10)]
            save_fibers_as_image(fs, (100, 100, 100), path=path, is_periodic=False)
            self.assertTrue(Path(path).exists())

    def test_unsupported_format_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.png")
            fs = [_straight_fiber(n=3)]
            with self.assertRaises(ValueError):
                save_fibers_as_image(fs, (100, 100, 100), path=path)

    def test_empty_fiber_system(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.tif")
            save_fibers_as_image([], (50, 50, 50), path=path)
            self.assertTrue(Path(path).exists())

    def test_scale_affects_output_shape(self):
        import tifffile
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.tif")
            fs = [_straight_fiber(n=3, radius=2, offset=10, length=5)]
            save_fibers_as_image(fs, (100, 100, 100), path=path, scale=2)
            img = tifffile.imread(path)
            # domain/scale = 50 voxels per axis; after transpose: (50,50,50)
            self.assertEqual(img.shape, (50, 50, 50))


# ---------------------------------------------------------------------------
# save_lines_as_image
# ---------------------------------------------------------------------------

class TestSaveLinesAsImage(unittest.TestCase):

    def _line_system(self):
        start = np.array([10.0, 50.0, 50.0])
        end = np.array([90.0, 50.0, 50.0])
        return [(start, end, 5.0)]

    def test_saves_tif(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "lines.tif")
            save_lines_as_image(self._line_system(), (100, 100, 100), path=path)
            self.assertTrue(Path(path).exists())

    def test_unsupported_format_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "lines.png")
            with self.assertRaises(ValueError):
                save_lines_as_image(self._line_system(), (100, 100, 100), path=path)

    def test_empty_line_system_does_not_crash(self):
        # exposes bug 4: max() on empty sequence
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "lines.tif")
            # once bug 4 is fixed this should produce an all-zero image
            try:
                save_lines_as_image([], (50, 50, 50), path=path)
                self.assertTrue(Path(path).exists())
            except ValueError:
                self.fail("save_lines_as_image raised ValueError on empty line_system")


# ---------------------------------------------------------------------------
# save_fibers_as_graph
# ---------------------------------------------------------------------------

class TestSaveFibersAsGraph(unittest.TestCase):

    def test_creates_all_output_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            fs = [_straight_fiber(n=5)]
            save_fibers_as_graph(base, fs)
            for suffix in [".fft.nod", ".fft.elm", ".fft.threads",
                           ".fft.ndn", ".fft.eln", ".fft.threads.number"]:
                self.assertTrue(Path(base + suffix).exists(), f"missing {suffix}")

    def test_node_count_matches_ndn_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            fs = [_straight_fiber(n=5)]
            save_fibers_as_graph(base, fs)
            ndn = int(Path(base + ".fft.ndn").read_text().strip())
            nod_lines = Path(base + ".fft.nod").read_text().splitlines()
            self.assertEqual(ndn, len(nod_lines))

    def test_edge_count_matches_eln_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            fs = [_straight_fiber(n=5)]
            save_fibers_as_graph(base, fs)
            eln = int(Path(base + ".fft.eln").read_text().strip())
            elm_lines = Path(base + ".fft.elm").read_text().splitlines()
            self.assertEqual(eln, len(elm_lines))

    def test_thread_count_matches_number_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            fs = [_straight_fiber(n=5), _straight_fiber(n=4)]
            save_fibers_as_graph(base, fs)
            n = int(Path(base + ".fft.threads.number").read_text().strip())
            threads = Path(base + ".fft.threads").read_text().splitlines()
            self.assertEqual(n, len(fs))
            self.assertEqual(len(threads), len(fs))

    def test_fibers_shorter_than_3_balls_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            # one fiber with 2 balls (too short) and one with 4
            short = _straight_fiber(n=2)
            long = _straight_fiber(n=4)
            save_fibers_as_graph(base, [short, long])
            threads = Path(base + ".fft.threads").read_text().splitlines()
            # only the 4-ball fiber should appear
            self.assertEqual(len(threads), 1)

    def test_multiple_fibers(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "fiber")
            fs = [_straight_fiber(n=5), _straight_fiber(n=6, axis="y")]
            save_fibers_as_graph(base, fs)
            ndn = int(Path(base + ".fft.ndn").read_text().strip())
            self.assertEqual(ndn, 5 + 6)


# ---------------------------------------------------------------------------
# save_fibers_as_small_graph
# ---------------------------------------------------------------------------

class TestSaveFibersAsSmallGraph(unittest.TestCase):

    def test_creates_all_output_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "small")
            fs = [_straight_fiber(n=10)]
            save_fibers_as_small_graph(base, fs)
            for suffix in [".fft.nod", ".fft.elm", ".fft.threads",
                           ".fft.ndn", ".fft.eln", ".fft.threads.number"]:
                self.assertTrue(Path(base + suffix).exists(), f"missing {suffix}")

    def test_fewer_nodes_than_full_graph_for_straight_fiber(self):
        # a straight fiber should be heavily simplified
        with tempfile.TemporaryDirectory() as d:
            base_full = str(Path(d) / "full")
            base_small = str(Path(d) / "small")
            fs = [_straight_fiber(n=20)]
            save_fibers_as_graph(base_full, fs)
            save_fibers_as_small_graph(base_small, fs)
            ndn_full = int(Path(base_full + ".fft.ndn").read_text().strip())
            ndn_small = int(Path(base_small + ".fft.ndn").read_text().strip())
            self.assertLessEqual(ndn_small, ndn_full)

    def test_node_count_matches_ndn_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "small")
            fs = [_curved_fiber(n=10)]
            save_fibers_as_small_graph(base, fs)
            ndn = int(Path(base + ".fft.ndn").read_text().strip())
            nod_lines = [l for l in Path(base + ".fft.nod").read_text().splitlines() if l]
            self.assertEqual(ndn, len(nod_lines))

    def test_fibers_shorter_than_3_balls_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            base = str(Path(d) / "small")
            save_fibers_as_small_graph(base, [_straight_fiber(n=2)])
            threads = Path(base + ".fft.threads").read_text().strip()
            self.assertEqual(threads, "")


# ---------------------------------------------------------------------------
# write_gad / read_gad (round-trip)
# ---------------------------------------------------------------------------

class TestGadRoundTrip(unittest.TestCase):

    def _build_fiber_system(self):
        """Two straight fibers with known geometry."""
        f1 = _straight_fiber(n=4, radius=5.0, offset=100, length=200)
        f2 = _straight_fiber(n=3, radius=8.0, axis="y", offset=200, length=300)
        return [f1, f2]

    def test_write_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            fs = self._build_fiber_system()
            write_gad(fs, path, (612, 612, 612))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("NumberOfObjects", data)
            self.assertEqual(data["NumberOfObjects"], 2)

    def test_roundtrip_number_of_fibers(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            fs = self._build_fiber_system()
            write_gad(fs, path, (612, 612, 612))
            fs_back, _ = read_gad(path)
            self.assertEqual(len(fs_back), len(fs))

    def test_roundtrip_number_of_balls(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            fs = self._build_fiber_system()
            write_gad(fs, path, (612, 612, 612))
            fs_back, _ = read_gad(path)
            for orig, back in zip(fs, fs_back):
                self.assertEqual(orig.get_number_of_balls(), back.get_number_of_balls())

    def test_roundtrip_radius(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            fs = self._build_fiber_system()
            write_gad(fs, path, (612, 612, 612))
            fs_back, _ = read_gad(path)
            for orig, back in zip(fs, fs_back):
                self.assertAlmostEqual(
                    orig.get_mean_radius(), back.get_mean_radius(), places=3
                )

    def test_roundtrip_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            fs = self._build_fiber_system()
            write_gad(fs, path, (612, 612, 612))
            fs_back, _ = read_gad(path)
            for orig, back in zip(fs, fs_back):
                for b_orig, b_back in zip(orig.balls, back.balls):
                    np.testing.assert_allclose(
                        b_orig.coordinate, b_back.coordinate, rtol=1e-5
                    )

    def test_roundtrip_domain_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            shape = (400, 500, 600)
            write_gad(self._build_fiber_system(), path, shape)
            _, shape_back = read_gad(path)
            self.assertEqual(shape_back, shape)

    def test_periodic_flag_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            write_gad(self._build_fiber_system(), path, (100, 100, 100),
                      is_periodic=False)
            with open(path) as f:
                data = json.load(f)
            self.assertFalse(data["Domain"]["PeriodicX"])

    def test_domain_lengths_in_metres(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "out.gad")
            shape = (612, 612, 612)
            write_gad(self._build_fiber_system(), path, shape)
            with open(path) as f:
                data = json.load(f)
            # 612 µm / 1e6 = 6.12e-4 m
            self.assertAlmostEqual(data["Domain"]["LengthX"][0], 612 / 1e6, places=10)


if __name__ == "__main__":
    unittest.main()