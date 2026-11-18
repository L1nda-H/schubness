"""Exact-within-pool analysis of adjustable local-scaling kNN graphs.

For a query cell ``i`` and candidate ``j`` the ranking score is

    D_gamma(i, j) = d(i, j)^2 / (sigma_i sigma_j)^gamma.

``sigma_i`` is constant for a fixed query, so rankings are determined by the
line ``2 log d(i, j) - gamma log sigma_j``.  This module enumerates pairwise
line crossings and retains only batches that change the top-k candidate set.

The event sequence is exact for the supplied candidate pool.  If the pool is
restricted to (say) the 100 Euclidean nearest neighbours, callers should
separately validate that restriction against all-cell reranking.
"""

from __future__ import division

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def build_candidate_pool(
    X, candidate_pool_size=100, local_scale_rank=10, metric="euclidean"
):
    """Compute Euclidean candidates and local scales.

    Parameters
    ----------
    X : array-like, shape (n_cells, n_features)
        Fixed cell representation, normally PCA coordinates.
    candidate_pool_size : int or None
        Number of Euclidean candidates retained per query. ``None`` uses all
        other cells.
    local_scale_rank : int
        The r in sigma_i = distance from cell i to its r-th neighbour.
    metric : str
        Metric accepted by :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    candidate_indices, candidate_distances, local_scales : numpy.ndarray
        Candidate arrays have shape ``(n_cells, pool_size)``. Self-neighbours
        are excluded. Local scales have shape ``(n_cells,)``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n_cells = X.shape[0]
    if n_cells < 2:
        raise ValueError("at least two cells are required")
    if not 1 <= local_scale_rank < n_cells:
        raise ValueError("local_scale_rank must be between 1 and n_cells - 1")

    if candidate_pool_size is None:
        pool_size = n_cells - 1
    else:
        pool_size = int(candidate_pool_size)
        if not 1 <= pool_size < n_cells:
            raise ValueError(
                "candidate_pool_size must be between 1 and n_cells - 1"
            )

    query_width = max(pool_size, local_scale_rank) + 1
    model = NearestNeighbors(n_neighbors=query_width, metric=metric)
    model.fit(X)
    raw_distances, raw_indices = model.kneighbors(X, return_distance=True)

    candidates = np.empty((n_cells, pool_size), dtype=np.int64)
    distances = np.empty((n_cells, pool_size), dtype=float)
    local_scales = np.empty(n_cells, dtype=float)

    for i in range(n_cells):
        keep = raw_indices[i] != i
        row_indices = raw_indices[i][keep]
        row_distances = raw_distances[i][keep]
        if row_indices.size < max(pool_size, local_scale_rank):
            raise RuntimeError("nearest-neighbour query did not return enough cells")
        candidates[i] = row_indices[:pool_size]
        distances[i] = row_distances[:pool_size]
        local_scales[i] = row_distances[local_scale_rank - 1]

    _validate_candidate_arrays(candidates, distances, local_scales)
    return candidates, distances, local_scales


def _validate_candidate_arrays(candidate_indices, candidate_distances, local_scales):
    candidate_indices = np.asarray(candidate_indices)
    candidate_distances = np.asarray(candidate_distances)
    local_scales = np.asarray(local_scales)
    if candidate_indices.ndim != 2:
        raise ValueError("candidate_indices must be two-dimensional")
    if candidate_indices.shape != candidate_distances.shape:
        raise ValueError("candidate indices and distances must have equal shape")
    if local_scales.shape != (candidate_indices.shape[0],):
        raise ValueError("local_scales must contain one value per cell")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= len(local_scales)):
        raise ValueError("candidate index is out of range")
    if np.any(candidate_distances < 0) or not np.all(np.isfinite(candidate_distances)):
        raise ValueError("candidate distances must be finite and nonnegative")
    if np.any(local_scales < 0) or not np.all(np.isfinite(local_scales)):
        raise ValueError("local scales must be finite and nonnegative")
    for i, row in enumerate(candidate_indices):
        if i in row:
            raise ValueError("candidate rows must exclude the query cell itself")
        if len(np.unique(row)) != len(row):
            raise ValueError("candidate rows cannot contain duplicate cells")


def _log_parameters(candidate_indices, candidate_distances, local_scales, epsilon):
    safe_distances = np.maximum(np.asarray(candidate_distances, dtype=float), epsilon)
    safe_scales = np.maximum(np.asarray(local_scales, dtype=float), epsilon)
    intercepts = 2.0 * np.log(safe_distances)
    slopes = np.log(safe_scales[np.asarray(candidate_indices, dtype=np.int64)])
    return intercepts, slopes


def neighbors_at_gamma(
    candidate_indices,
    candidate_distances,
    local_scales,
    k,
    gamma,
    epsilon=1e-12,
):
    """Return directed top-k neighbours at one correction strength."""
    _validate_candidate_arrays(candidate_indices, candidate_distances, local_scales)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if not 1 <= k <= candidate_indices.shape[1]:
        raise ValueError("k must be between 1 and the candidate-pool size")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    intercepts, slopes = _log_parameters(
        candidate_indices, candidate_distances, local_scales, epsilon
    )
    result = np.empty((candidate_indices.shape[0], k), dtype=np.int64)
    for i in range(candidate_indices.shape[0]):
        scores = intercepts[i] - gamma * slopes[i]
        # Cell id is the deterministic tie-breaker.
        order = np.lexsort((candidate_indices[i], scores))
        result[i] = candidate_indices[i, order[:k]]
    return result


def enumerate_graph_events(
    candidate_indices,
    candidate_distances,
    local_scales,
    k,
    gamma_min=0.0,
    gamma_max=1.0,
    tolerance=1e-12,
    epsilon=1e-12,
):
    """Enumerate crossover batches that change a directed top-k set.

    Returns a list of dictionaries. Each record may contain multiple removed
    and added targets when several score lines cross at the same gamma. Events
    exactly at the endpoints are excluded because the graph is otherwise
    defined on open intervals between crossings.
    """
    _validate_candidate_arrays(candidate_indices, candidate_distances, local_scales)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if not 1 <= k <= candidate_indices.shape[1]:
        raise ValueError("k must be between 1 and the candidate-pool size")
    if not gamma_min < gamma_max:
        raise ValueError("gamma_min must be smaller than gamma_max")

    intercepts, slopes = _log_parameters(
        candidate_indices, candidate_distances, local_scales, epsilon
    )
    events = []
    pool_size = candidate_indices.shape[1]

    for query in range(candidate_indices.shape[0]):
        targets = candidate_indices[query]
        a = intercepts[query]
        b = slopes[query]
        initial_scores = a - gamma_min * b
        order = list(np.lexsort((targets, initial_scores)))
        positions = np.empty(pool_size, dtype=np.int64)
        for position, candidate_position in enumerate(order):
            positions[candidate_position] = position

        crossings = []
        for left in range(pool_size - 1):
            for right in range(left + 1, pool_size):
                denominator = b[left] - b[right]
                if abs(denominator) <= tolerance:
                    continue
                gamma = (a[left] - a[right]) / denominator
                boundary_tolerance = tolerance * max(1.0, abs(gamma))
                if (
                    gamma > gamma_min + boundary_tolerance
                    and gamma < gamma_max - boundary_tolerance
                ):
                    crossings.append((float(gamma), left, right))

        crossings.sort(key=lambda value: value[0])
        cursor = 0
        while cursor < len(crossings):
            gamma = crossings[cursor][0]
            batch = []
            while cursor < len(crossings) and math.isclose(
                crossings[cursor][0], gamma, rel_tol=tolerance, abs_tol=tolerance
            ):
                batch.append(crossings[cursor])
                cursor += 1

            before = set(targets[np.asarray(order[:k], dtype=np.int64)])

            # Connected components identify score lines tied at this gamma.
            # Lines that are identical for every gamma have no pairwise
            # crossing, so each component is expanded to include all lines at
            # the same score before its order is updated.
            adjacency = {}
            for _, left, right in batch:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
            unseen = set(adjacency)
            components = []
            while unseen:
                root = unseen.pop()
                stack = [root]
                component = {root}
                while stack:
                    node = stack.pop()
                    for neighbour in adjacency[node]:
                        if neighbour not in component:
                            component.add(neighbour)
                            unseen.discard(neighbour)
                            stack.append(neighbour)
                components.append(component)

            scores_at_crossing = a - gamma * b
            expanded_components = []
            assigned = set()
            for component in components:
                if component <= assigned:
                    continue
                representative = next(iter(component))
                crossing_score = scores_at_crossing[representative]
                expanded = {
                    node
                    for node in range(pool_size)
                    if math.isclose(
                        scores_at_crossing[node],
                        crossing_score,
                        rel_tol=tolerance,
                        abs_tol=tolerance,
                    )
                }
                assigned.update(expanded)
                expanded_components.append(expanded)

            for component in expanded_components:
                component_positions = sorted(positions[node] for node in component)
                # Immediately after the crossing, larger log(sigma) wins.
                after_component = sorted(
                    component, key=lambda node: (-b[node], int(targets[node]))
                )
                for position, node in zip(component_positions, after_component):
                    order[position] = node
                    positions[node] = position

            after = set(targets[np.asarray(order[:k], dtype=np.int64)])
            removed = tuple(map(int, sorted(before - after)))
            added = tuple(map(int, sorted(after - before)))
            if removed or added:
                if len(removed) != len(added):
                    raise RuntimeError("top-k event did not preserve neighbour count")
                events.append(
                    {
                        "gamma": gamma,
                        "query": int(query),
                        "removed": removed,
                        "added": added,
                        "n_swaps": len(removed),
                    }
                )

    events.sort(key=lambda event: (event["gamma"], event["query"]))
    return events


def _undirected_edges(neighbours):
    edges = set()
    for source, targets in enumerate(neighbours):
        for target in targets:
            target = int(target)
            edges.add((min(source, target), max(source, target)))
    return edges


def annotate_graph_events(
    events,
    initial_neighbours,
    labels,
    local_scales,
    k,
    hub_multiplier=2.0,
    tolerance=1e-12,
):
    """Annotate events using labels, dynamic in-degree, and graph effects.

    All events at the same gamma are applied simultaneously. Hub status and
    in-degree refer to the graph immediately before that event batch.
    """
    labels = np.asarray(labels)
    local_scales = np.asarray(local_scales, dtype=float)
    neighbours = [set(map(int, row)) for row in np.asarray(initial_neighbours)]
    n_cells = len(neighbours)
    if labels.shape != (n_cells,) or local_scales.shape != (n_cells,):
        raise ValueError("labels and local_scales must contain one value per cell")
    if any(len(row) != k for row in neighbours):
        raise ValueError("each initial neighbour row must contain exactly k cells")

    indegree = np.zeros(n_cells, dtype=np.int64)
    for row in neighbours:
        for target in row:
            indegree[target] += 1
    population_sizes = Counter(labels.tolist())
    rows = []
    cursor = 0

    while cursor < len(events):
        gamma = events[cursor]["gamma"]
        batch = []
        while cursor < len(events) and math.isclose(
            events[cursor]["gamma"], gamma, rel_tol=tolerance, abs_tol=tolerance
        ):
            batch.append(events[cursor])
            cursor += 1

        # Copy only rows changed by this batch. Rebuilding the full graph for
        # every distinct gamma would make annotation quadratic in the number
        # of events.
        after_neighbours = list(neighbours)
        for event in batch:
            query = event["query"]
            after_neighbours[query] = set(neighbours[query])
            after_neighbours[query].difference_update(event["removed"])
            after_neighbours[query].update(event["added"])

        for event in batch:
            query = event["query"]
            removed = tuple(event["removed"])
            added = tuple(event["added"])
            source_label = labels[query]
            removed_same = sum(labels[target] == source_label for target in removed)
            added_same = sum(labels[target] == source_label for target in added)
            removed_effective = 0
            for target in removed:
                loses_undirected_edge = query not in after_neighbours[target]
                reverse_existed_before = query in neighbours[target]
                if loses_undirected_edge and (
                    not reverse_existed_before or query < target
                ):
                    removed_effective += 1
            added_effective = 0
            for target in added:
                creates_undirected_edge = query not in neighbours[target]
                reverse_exists_after = query in after_neighbours[target]
                if creates_undirected_edge and (
                    not reverse_exists_after or query < target
                ):
                    added_effective += 1
            rows.append(
                {
                    "gamma": gamma,
                    "query": query,
                    "source_label": source_label,
                    "source_population_size": population_sizes[source_label],
                    "source_sigma": local_scales[query],
                    "removed": json.dumps(list(removed)),
                    "added": json.dumps(list(added)),
                    "n_swaps": len(removed),
                    "removed_same_label": removed_same,
                    "added_same_label": added_same,
                    "delta_same_label": added_same - removed_same,
                    "removed_cross_label": len(removed) - removed_same,
                    "added_cross_label": len(added) - added_same,
                    "removed_mean_sigma": _safe_mean(local_scales[list(removed)]),
                    "added_mean_sigma": _safe_mean(local_scales[list(added)]),
                    "removed_mean_indegree_before": _safe_mean(indegree[list(removed)]),
                    "added_mean_indegree_before": _safe_mean(indegree[list(added)]),
                    "removed_hub_before": any(
                        indegree[target] >= hub_multiplier * k for target in removed
                    ),
                    "added_hub_before": any(
                        indegree[target] >= hub_multiplier * k for target in added
                    ),
                    "removed_effective_undirected": removed_effective,
                    "added_effective_undirected": added_effective,
                }
            )

        for event in batch:
            for target in event["removed"]:
                indegree[target] -= 1
            for target in event["added"]:
                indegree[target] += 1
        neighbours = after_neighbours

    return pd.DataFrame(rows)


def _safe_mean(values):
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else float("nan")


def graph_summary(neighbours, labels=None, hub_multiplier=2.0):
    """Return degree, label-purity, and graph-change summary statistics."""
    neighbours = np.asarray(neighbours, dtype=np.int64)
    if neighbours.ndim != 2:
        raise ValueError("neighbours must be a two-dimensional array")
    n_cells, k = neighbours.shape
    indegree = np.bincount(neighbours.ravel(), minlength=n_cells).astype(float)
    mean_degree = float(k)
    centered = indegree - mean_degree
    variance = float(np.mean(centered ** 2))
    skewness = (
        float(np.mean(centered ** 3) / variance ** 1.5) if variance > 0 else 0.0
    )
    sorted_degree = np.sort(indegree)
    index = np.arange(1, n_cells + 1, dtype=float)
    degree_sum = float(np.sum(sorted_degree))
    gini = (
        float(
            np.sum((2.0 * index - n_cells - 1.0) * sorted_degree)
            / (n_cells * degree_sum)
        )
        if degree_sum > 0
        else 0.0
    )
    robin_hood = (
        float(0.5 * np.sum(np.abs(indegree - mean_degree)) / degree_sum)
        if degree_sum > 0
        else 0.0
    )
    result = {
        "n_cells": n_cells,
        "k": k,
        "in_degree_skewness": skewness,
        "gini": gini,
        "robin_hood": robin_hood,
        "anti_hub_fraction": float(np.mean(indegree == 0)),
        "hub_occurrence": float(
            np.sum(indegree[indegree >= hub_multiplier * k]) / degree_sum
        ),
        "undirected_edge_count": len(_undirected_edges(neighbours)),
    }
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (n_cells,):
            raise ValueError("labels must contain one value per cell")
        same = 0
        for source, targets in enumerate(neighbours):
            same += int(np.sum(labels[targets] == labels[source]))
        result["directed_same_label_fraction"] = float(same / (n_cells * k))
        result["directed_cross_label_fraction"] = 1.0 - result[
            "directed_same_label_fraction"
        ]
    return result


def summarize_gammas(
    candidate_indices,
    candidate_distances,
    local_scales,
    k,
    gammas,
    labels=None,
):
    """Compute graph summaries at selected gamma checkpoints."""
    rows = []
    for gamma in gammas:
        neighbours = neighbors_at_gamma(
            candidate_indices,
            candidate_distances,
            local_scales,
            k,
            float(gamma),
        )
        row = graph_summary(neighbours, labels=labels)
        row["gamma"] = float(gamma)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


def population_edge_table(neighbours, labels, gamma=None):
    """Count directed edges between every ordered pair of populations."""
    neighbours = np.asarray(neighbours, dtype=np.int64)
    labels = np.asarray(labels)
    counts = Counter()
    for source, targets in enumerate(neighbours):
        for target in targets:
            counts[(labels[source], labels[target])] += 1
    rows = []
    source_totals = Counter()
    for (source_label, _), count in counts.items():
        source_totals[source_label] += count
    for (source_label, target_label), count in sorted(
        counts.items(), key=lambda value: (str(value[0][0]), str(value[0][1]))
    ):
        row = {
            "source_label": source_label,
            "target_label": target_label,
            "edge_count": count,
            "source_fraction": count / source_totals[source_label],
        }
        if gamma is not None:
            row["gamma"] = gamma
        rows.append(row)
    return pd.DataFrame(rows)


def population_conductance_table(neighbours, labels, gamma=None):
    """Compute conductance of each reference population on the union graph."""
    neighbours = np.asarray(neighbours, dtype=np.int64)
    labels = np.asarray(labels)
    if neighbours.ndim != 2 or labels.shape != (neighbours.shape[0],):
        raise ValueError("neighbours and labels have incompatible shapes")
    edges = _undirected_edges(neighbours)
    degree = np.zeros(neighbours.shape[0], dtype=np.int64)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    total_volume = int(np.sum(degree))
    rows = []
    for label in sorted(np.unique(labels), key=str):
        in_population = labels == label
        volume = int(np.sum(degree[in_population]))
        cut_edges = sum(in_population[left] != in_population[right] for left, right in edges)
        denominator = min(volume, total_volume - volume)
        row = {
            "label": label,
            "population_size": int(np.sum(in_population)),
            "cut_edges": int(cut_edges),
            "volume": volume,
            "conductance": (
                float(cut_edges / denominator) if denominator > 0 else float("nan")
            ),
        }
        if gamma is not None:
            row["gamma"] = gamma
        rows.append(row)
    return pd.DataFrame(rows)


def run_event_analysis(
    X,
    labels,
    k=15,
    candidate_pool_size=100,
    local_scale_rank=10,
    gamma_min=0.0,
    gamma_max=1.0,
):
    """Convenience wrapper returning candidates, events, and annotations."""
    candidates, distances, scales = build_candidate_pool(
        X,
        candidate_pool_size=candidate_pool_size,
        local_scale_rank=local_scale_rank,
    )
    initial = neighbors_at_gamma(
        candidates, distances, scales, k=k, gamma=gamma_min
    )
    events = enumerate_graph_events(
        candidates,
        distances,
        scales,
        k=k,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    annotated = annotate_graph_events(
        events, initial, labels=labels, local_scales=scales, k=k
    )
    return {
        "candidate_indices": candidates,
        "candidate_distances": distances,
        "local_scales": scales,
        "initial_neighbours": initial,
        "events": events,
        "event_table": annotated,
    }
