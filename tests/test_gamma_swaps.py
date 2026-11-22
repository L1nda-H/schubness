import math
import os
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERING_DIR = os.path.join(REPOSITORY_ROOT, "Scripts", "python", "Clustering")
sys.path.insert(0, CLUSTERING_DIR)

from gamma_connectivity import (  # noqa: E402
    build_candidate_pool,
    enumerate_graph_events,
    neighbors_at_gamma,
)
from gamma_leiden import leiden_fixed_k  # noqa: E402
from gamma_swaps import (  # noqa: E402
    fragility_table,
    population_profile,
    purity_trajectory,
    swap_table,
)


def _toy():
    # Cell 0 has candidates 1 (near, dense) and 2 (farther, sparse).
    candidates = np.array([[1, 2], [0, 2], [0, 1]], dtype=int)
    distances = np.array([[1.0, 1.2], [1.0, 2.0], [1.0, 2.0]], dtype=float)
    scales = np.array([1.0, 1.0, 4.0], dtype=float)
    return candidates, distances, scales


class SwapTableTests(unittest.TestCase):
    def test_single_swap_matches_closed_form_gamma(self):
        candidates, distances, scales = _toy()
        labels = np.array(["a", "a", "b"])
        initial = neighbors_at_gamma(candidates, distances, scales, 1, 0.0)
        events = enumerate_graph_events(candidates, distances, scales, 1)
        swaps = swap_table(events, candidates, distances, scales, initial, labels, 1)
        self.assertEqual(len(swaps), 1)
        row = swaps.iloc[0]
        expected = 2 * math.log(1.2 / 1.0) / math.log(4.0 / 1.0)
        self.assertAlmostEqual(row["gamma_star"], expected)
        self.assertAlmostEqual(row["gamma_star_formula"], expected)
        self.assertEqual((row["query"], row["removed"], row["added"]), (0, 1, 2))
        self.assertEqual(row["transition"], "same->cross")
        self.assertEqual(row["delta_same_label"], -1)
        self.assertEqual(row["added_pop_size"], 1)
        self.assertEqual(row["removed_pop_size"], 2)
        self.assertGreater(row["log_sigma_ratio"], 0)

    def test_swap_table_accepts_csv_style_events(self):
        candidates, distances, scales = _toy()
        labels = np.array(["a", "a", "b"])
        initial = neighbors_at_gamma(candidates, distances, scales, 1, 0.0)
        events = enumerate_graph_events(candidates, distances, scales, 1)
        frame = pd.DataFrame(
            [
                {
                    "gamma": e["gamma"],
                    "query": e["query"],
                    "removed": str(list(e["removed"])),
                    "added": str(list(e["added"])),
                }
                for e in events
            ]
        )
        a = swap_table(events, candidates, distances, scales, initial, labels, 1)
        b = swap_table(frame, candidates, distances, scales, initial, labels, 1)
        pd.testing.assert_frame_equal(a, b)

    def test_replayed_swaps_reproduce_final_graph(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(0, 1, (40, 8)), rng.normal(3, 2, (40, 8))])
        labels = np.array(["a"] * 40 + ["b"] * 40)
        c, d, s = build_candidate_pool(X, candidate_pool_size=20, local_scale_rank=5)
        k = 5
        start = neighbors_at_gamma(c, d, s, k, 0.0)
        end = neighbors_at_gamma(c, d, s, k, 1.0)
        swaps = swap_table(
            enumerate_graph_events(c, d, s, k), c, d, s, start, labels, k
        )
        graph = [set(row) for row in start]
        for row in swaps.itertuples():
            graph[row.query].remove(row.removed)
            graph[row.query].add(row.added)
        self.assertEqual(graph, [set(row) for row in end])
        np.testing.assert_allclose(
            swaps["gamma_star"], swaps["gamma_star_formula"], atol=1e-9
        )
        # Net same-label change equals the change in directed purity.
        purity = lambda n: np.sum(labels[n] == labels[:, None])  # noqa: E731
        self.assertEqual(swaps["delta_same_label"].sum(), purity(end) - purity(start))

        trajectory = purity_trajectory(swaps, labels, gamma_grid=[0.0, 1.0])
        total = trajectory[trajectory["source_label"] == "__all__"]
        self.assertEqual(total["cumulative_delta_same"].iloc[0], 0)
        self.assertEqual(
            total["cumulative_delta_same"].iloc[-1], swaps["delta_same_label"].sum()
        )
        frag = fragility_table(swaps)
        self.assertEqual(frag["swaps"].sum(), len(swaps))
        profile = population_profile(swaps, labels, s, start)
        self.assertEqual(set(profile["label"]), {"a", "b"})


class FixedKLeidenTests(unittest.TestCase):
    def test_recovers_two_separated_blobs(self):
        rng = np.random.RandomState(1)
        X = np.vstack([rng.normal(0, 1, (60, 5)), rng.normal(10, 1, (60, 5))])
        labels = np.array([0] * 60 + [1] * 60)
        c, d, s = build_candidate_pool(X, candidate_pool_size=20, local_scale_rank=5)
        nbrs = neighbors_at_gamma(c, d, s, 10, 0.0)
        rows = leiden_fixed_k(nbrs, labels, seeds=(0,))
        self.assertTrue(rows[0]["hit_target"])
        self.assertAlmostEqual(rows[0]["ari"], 1.0)


if __name__ == "__main__":
    unittest.main()
