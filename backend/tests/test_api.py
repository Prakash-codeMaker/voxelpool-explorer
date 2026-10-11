import unittest
from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_preview_returns_occupancy_without_timing(self):
        request = {
            "points": 48,
            "channels": 4,
            "grid_size": 4,
            "pattern": "uniform",
            "repeats": 3,
            "seed": 12,
        }
        response = self.client.post("/api/preview", json=request)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["preview_only"])
        self.assertNotIn("strategies", payload)
        self.assertEqual(len(payload["occupancy"]["xy_projection"]), 4)
        self.assertEqual(payload["workload"]["seed"], 12)

    def test_preview_is_deterministic_for_same_seed(self):
        request = {"points": 30, "channels": 4, "grid_size": 8, "pattern": "hotspot", "seed": 17}
        first = self.client.post("/api/preview", json=request).json()
        second = self.client.post("/api/preview", json=request).json()
        self.assertEqual(first["occupancy"], second["occupancy"])

    def test_experiment_includes_environment_and_correctness(self):
        request = {"points": 20, "channels": 4, "grid_size": 4, "pattern": "unique", "repeats": 2, "seed": 3}
        response = self.client.post("/api/experiment", json=request)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["correctness"]["all_candidates_passed"])
        self.assertTrue(payload["environment"]["python_version"])
        self.assertTrue(payload["environment"]["numpy_version"])
        self.assertEqual(len(payload["strategies"]), 3)

    def test_memory_guardrail_rejects_excessive_dense_output(self):
        response = self.client.post("/api/preview", json={
            "points": 64, "channels": 128, "grid_size": 64, "pattern": "uniform", "seed": 1
        })
        self.assertEqual(response.status_code, 422)

    def test_unique_pattern_rejects_more_points_than_voxels(self):
        response = self.client.post("/api/preview", json={
            "points": 65, "channels": 4, "grid_size": 4, "pattern": "unique", "seed": 1
        })
        self.assertEqual(response.status_code, 422)

    def test_sweep_runs_controlled_cases_and_returns_progress(self):
        response = self.client.post("/api/sweeps", json={
            "points": [8, 12], "channels": [2, 4], "grid_sizes": [4],
            "patterns": ["uniform"], "seeds": [21], "repeats": 1,
        })
        self.assertEqual(response.status_code, 200)
        started = response.json()
        self.assertEqual(started["total_cases"], 4)
        poll = self.client.get(f"/api/sweeps/{started['sweep_id']}")
        self.assertEqual(poll.status_code, 200)
        payload = poll.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed_cases"], 4)
        self.assertEqual(len(payload["cases"]), 4)
        self.assertTrue(all(case["result"]["correctness"]["all_candidates_passed"] for case in payload["cases"]))

    def test_sweep_rejects_too_many_combinations(self):
        response = self.client.post("/api/sweeps", json={
            "points": [8, 12], "channels": [2, 4], "grid_sizes": [4, 8],
            "patterns": ["uniform", "hotspot", "unique"], "seeds": [1, 2, 3], "repeats": 1,
        })
        self.assertEqual(response.status_code, 422)

    def test_selector_evaluation_uses_distinct_configurations(self):
        response = self.client.post("/api/sweeps", json={
            "points": [8, 12, 16], "channels": [2, 4], "grid_sizes": [4],
            "patterns": ["uniform"], "seeds": [21, 22], "repeats": 1,
        })
        self.assertEqual(response.status_code, 200)
        sweep_id = response.json()["sweep_id"]
        sweep = self.client.get(f"/api/sweeps/{sweep_id}").json()
        evaluation = self.client.post("/api/selector/evaluate", json={"cases": sweep["cases"]})
        self.assertEqual(evaluation.status_code, 200, evaluation.text)
        result = evaluation.json()
        self.assertEqual(result["status"], "evaluated")
        self.assertGreaterEqual(result["training_configurations"], 2)
        self.assertGreaterEqual(result["heldout_configurations"], 1)
        self.assertEqual(result["input_configurations"], 6)
        self.assertGreaterEqual(result["selector_mean_regret_percent"], -0.001)
        self.assertEqual(len(result["evaluation_cases"]), result["heldout_configurations"])


if __name__ == "__main__":
    unittest.main()
