"""Interval math checks distinguish actual overlap from a forward envelope."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project2_trace_audit import intersection_duration, merged


class TraceIntervalTests(unittest.TestCase):
    def test_union_does_not_double_count_nested_intervals(self):
        self.assertEqual(merged([(3, 8), (1, 5), (2, 4), (8, 9), (10, 10)]), [(1, 9)])
        self.assertEqual(intersection_duration([(1, 5), (2, 4)], [(2, 3), (2.5, 4)]), 2)

    def test_kernel_gap_is_not_overlap(self):
        self.assertEqual(intersection_duration([(3, 4)], [(1, 2), (5, 6)]), 0)
        self.assertEqual(intersection_duration([(3, 4)], [(1, 6)]), 1)

    def test_touching_boundaries_and_empty_sets(self):
        self.assertEqual(intersection_duration([(0, 1)], [(1, 2)]), 0)
        self.assertEqual(intersection_duration([], [(1, 2)]), 0)
        self.assertEqual(intersection_duration([(1, 2)], []), 0)


if __name__ == "__main__":
    unittest.main()
