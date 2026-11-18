import math
import os
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERING_DIR = os.path.join(REPOSITORY_ROOT, "Scripts", "python", "Clustering")
sys.path.insert(0, CLUSTERING_DIR)

from gamma_connectivity import (  # noqa: E402
    annotate_graph_events,
    build_candidate_pool,
    enumerate_graph_events,
    graph_summary,
    neighbors_at_gamma,
    population_conductance_table,
)


class GammaConnectivityTests(unittest.TestCase):
    def setUp(self):
        # Every row excludes its own query cell and contains unique targets.
        self.candidates = np.array([[1, 2], [0, 2], [0, 1]], dtype=int)
        self.distances = np.array(
            [[1.0, 1.2], [1.0, 2.0], [1.0, 2.0]], dtype=float
        )
        self.scales = np.array([1.0, 1.0, 4.0], dtype=float)

    def test_candidate_pool_excludes_self_and_uses_ranked_scale(self):
        X = np.array([[0.0], [1.0], [3.0], [10.0]])
        candidates, distances, scales = build_candidate_pool(
            X, candidate_pool_size=2, local_scale_rank=2
        )
        self.assertEqual(candidates.shape, (4, 2))
        self.assertEqual(distances.shape, (4, 2))
        for query, row in enumerate(candidates):
            self.assertNotIn(query, row)
        self.assertAlmostEqual(scales[0], 3.0)
        self.assertAlmostEqual(scales[1], 2.0)

    def test_known_switch_threshold_changes_top_one(self):
        expected = 2.0 * math.log(1.2) / math.log(4.0)
        before = neighbors_at_gamma(
            self.candidates, self.distances, self.scales, k=1, gamma=0.0
        )
        after = neighbors_at_gamma(
            self.candidates, self.distances, self.scales, k=1, gamma=1.0
        )
        self.assertEqual(before[0, 0], 1)
        self.assertEqual(after[0, 0], 2)

        events = enumerate_graph_events(
            self.candidates, self.distances, self.scales, k=1
        )
        query_zero = [event for event in events if event["query"] == 0]
        self.assertEqual(len(query_zero), 1)
        self.assertAlmostEqual(query_zero[0]["gamma"], expected)
        self.assertEqual(query_zero[0]["removed"], (1,))
        self.assertEqual(query_zero[0]["added"], (2,))

    def test_crossover_below_top_k_is_not_a_graph_event(self):
        candidates = np.array(
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=int
        )
        distances = np.array(
            [
                [0.1, 1.0, 1.2],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        scales = np.array([1.0, 1.0, 1.0, 4.0])
        events = enumerate_graph_events(candidates, distances, scales, k=1)
        self.assertEqual([event for event in events if event["query"] == 0], [])

    def test_annotations_capture_label_and_undirected_effects(self):
        initial = neighbors_at_gamma(
            self.candidates, self.distances, self.scales, k=1, gamma=0.0
        )
        events = enumerate_graph_events(
            self.candidates, self.distances, self.scales, k=1
        )
        table = annotate_graph_events(
            events,
            initial,
            labels=np.array(["A", "A", "B"]),
            local_scales=self.scales,
            k=1,
        )
        row = table.loc[table["query"] == 0].iloc[0]
        self.assertEqual(row["delta_same_label"], -1)
        self.assertEqual(row["removed_same_label"], 1)
        self.assertEqual(row["added_cross_label"], 1)
        # Cell 2 already points to cell 0, so adding 0 -> 2 does not create a
        # new edge in the symmetrized graph.
        self.assertEqual(row["added_effective_undirected"], 0)

    def test_graph_summary(self):
        neighbours = np.array([[1], [0], [0]])
        summary = graph_summary(neighbours, labels=np.array(["A", "A", "B"]))
        self.assertAlmostEqual(summary["directed_same_label_fraction"], 2.0 / 3.0)
        self.assertEqual(summary["undirected_edge_count"], 2)
        self.assertAlmostEqual(summary["anti_hub_fraction"], 1.0 / 3.0)

    def test_population_conductance(self):
        neighbours = np.array([[1], [0], [1], [2]])
        labels = np.array(["A", "A", "B", "B"])
        table = population_conductance_table(neighbours, labels)
        values = dict(zip(table["label"], table["conductance"]))
        self.assertAlmostEqual(values["A"], 1.0 / 3.0)
        self.assertAlmostEqual(values["B"], 1.0 / 3.0)

    def test_event_sequence_reaches_direct_gamma_result(self):
        rng = np.random.RandomState(7)
        n_cells = 8
        pool_size = 5
        candidates = np.empty((n_cells, pool_size), dtype=int)
        distances = np.empty((n_cells, pool_size), dtype=float)
        for query in range(n_cells):
            available = np.delete(np.arange(n_cells), query)
            candidates[query] = rng.choice(available, size=pool_size, replace=False)
            distances[query] = rng.uniform(0.2, 3.0, size=pool_size)
        scales = rng.uniform(0.3, 4.0, size=n_cells)
        k = 3
        initial = neighbors_at_gamma(
            candidates, distances, scales, k=k, gamma=0.0
        )
        expected = neighbors_at_gamma(
            candidates, distances, scales, k=k, gamma=1.0
        )
        events = enumerate_graph_events(candidates, distances, scales, k=k)
        reconstructed = [set(row) for row in initial]
        for event in events:
            reconstructed[event["query"]].difference_update(event["removed"])
            reconstructed[event["query"]].update(event["added"])
        self.assertEqual(
            reconstructed,
            [set(row) for row in expected],
        )


if __name__ == "__main__":
    unittest.main()
