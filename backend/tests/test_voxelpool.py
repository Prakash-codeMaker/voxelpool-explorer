import unittest
import numpy as np

from app.workloads import generate_workload, occupancy_summary
from app.voxelpool import python_reference, numpy_add_at, sort_segmented, run_experiment


class WorkloadTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        a = generate_workload(50, 8, 8, "hotspot", 123)
        b = generate_workload(50, 8, 8, "hotspot", 123)
        np.testing.assert_array_equal(a.voxel_ids, b.voxel_ids)
        np.testing.assert_array_equal(a.features, b.features)

    def test_unique_pattern_has_no_collisions(self):
        workload = generate_workload(100, 4, 8, "unique", 42)
        stats = occupancy_summary(workload)
        self.assertEqual(stats["collision_points"], 0)
        self.assertEqual(stats["occupied_voxels"], 100)

    def test_unique_rejects_too_many_points(self):
        with self.assertRaises(ValueError):
            generate_workload(513, 4, 8, "unique", 42)

    def test_all_candidates_match_reference(self):
        for pattern in ("uniform", "hotspot", "unique"):
            workload = generate_workload(120, 7, 8, pattern, 7)
            reference = python_reference(workload)
            for fn in (numpy_add_at, sort_segmented):
                actual = fn(workload)
                self.assertTrue(np.allclose(actual, reference, rtol=1e-4, atol=1e-4))

    def test_experiment_has_correctness_and_timing(self):
        workload = generate_workload(32, 4, 4, "uniform", 99)
        result = run_experiment(workload, repeats=2)
        self.assertEqual(result["measurement_backend"], "portable_cpu_numpy")
        self.assertTrue(result["correctness"]["all_candidates_passed"])
        self.assertEqual(len(result["strategies"]), 3)
        for strategy in result["strategies"]:
            self.assertTrue(strategy["correct"])
            self.assertIsNotNone(strategy["timing"]["median_ms"])
            self.assertEqual(len(strategy["timing"]["samples_ms"]), 2)
            self.assertIn("coefficient_of_variation_percent", strategy["timing"])
        self.assertEqual(len(result["methodology"]["measurement_order"]), 2)
        # The randomized/interleaved schedule is seeded and therefore reproducible.
        second = run_experiment(workload, repeats=2)
        self.assertEqual(result["methodology"]["measurement_order"], second["methodology"]["measurement_order"])


if __name__ == "__main__":
    unittest.main()
