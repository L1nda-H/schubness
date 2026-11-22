"""Per-swap analysis of adjustable local scaling (roadmap "next steps").

``gamma_connectivity.enumerate_graph_events`` returns event *batches*. This
module expands them into one row per swap ``i: j -> l`` (cell ``i`` drops
``j`` and gains ``l`` at ``gamma*``) and annotates each swap with what the
roadmap asks for:

* same-label status of the removed and the added edge;
* hub status (in-degree >= 2k, in the graph just before the swap) and
  in-degree of both targets;
* the local scale sigma of all three cells, also as a within-dataset
  percentile (dense = low, sparse = high);
* the sizes of the populations of all three cells;
* whether each directed change alters the undirected graph used by Leiden.

It also recomputes ``gamma* = 2 log(d_il / d_ij) / log(sigma_l / sigma_j)``
from the candidate distances, as an independent check of the event sequence.

Summaries built on the swap table:

* ``purity_trajectory``  exact (grid-free) cumulative change in same-label
  out-edges per cell, per population, as a step function of gamma;
* ``fragility_table``   swap composition by gamma* bin;
* ``edge_flow_table``   where the edges each population loses are re-routed.
"""

from __future__ import division

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

TRANSITIONS = ("same->same", "same->cross", "cross->same", "cross->cross")


def _as_tuple(value):
    if isinstance(value, str):
        return tuple(json.loads(value))
    return tuple(value)


def _event_records(events):
    """Accept the list from enumerate_graph_events or an edge_events table."""
    if isinstance(events, pd.DataFrame):
        records = []
        for row in events.itertuples(index=False):
            records.append(
                {
                    "gamma": float(row.gamma),
                    "query": int(row.query),
                    "removed": _as_tuple(row.removed),
                    "added": _as_tuple(row.added),
                }
            )
        return records
    return [
        {
            "gamma": float(e["gamma"]),
            "query": int(e["query"]),
            "removed": tuple(e["removed"]),
            "added": tuple(e["added"]),
        }
        for e in events
    ]


def swap_table(
    events,
    candidate_indices,
    candidate_distances,
    local_scales,
    initial_neighbours,
    labels,
    k,
    hub_multiplier=2.0,
    tolerance=1e-12,
):
    """Expand event batches into one annotated row per swap.

    Within a multi-swap batch, removed and added targets are paired by
    ascending log-scale (the pairing that is exact for chains of crossings);
    such rows carry ``multi_swap=True``. Hub status and in-degree refer to the
    graph immediately before the batch, matching annotate_graph_events.
    """
    records = _event_records(events)
    labels = np.asarray(labels)
    scales = np.asarray(local_scales, dtype=float)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_distances = np.asarray(candidate_distances, dtype=float)
    n_cells = len(labels)
    neighbours = [set(map(int, row)) for row in np.asarray(initial_neighbours)]
    indegree = np.zeros(n_cells, dtype=np.int64)
    for row in neighbours:
        for target in row:
            indegree[target] += 1

    sizes = Counter(labels.tolist())
    sigma_pct = (rankdata(scales) - 0.5) / n_cells
    log_scales = np.log(np.maximum(scales, 1e-12))
    position = [
        {int(c): p for p, c in enumerate(candidate_indices[i])} for i in range(n_cells)
    ]

    rows = []
    cursor = 0
    while cursor < len(records):
        gamma = records[cursor]["gamma"]
        batch = []
        while cursor < len(records) and math.isclose(
            records[cursor]["gamma"], gamma, rel_tol=tolerance, abs_tol=tolerance
        ):
            batch.append(records[cursor])
            cursor += 1

        after = list(neighbours)
        for event in batch:
            q = event["query"]
            after[q] = (set(neighbours[q]) - set(event["removed"])) | set(event["added"])

        for event in batch:
            i = event["query"]
            removed = sorted(event["removed"], key=lambda c: (log_scales[c], c))
            added = sorted(event["added"], key=lambda c: (log_scales[c], c))
            multi = len(removed) > 1
            for j, l in zip(removed, added):
                d_ij = candidate_distances[i, position[i][j]]
                d_il = candidate_distances[i, position[i][l]]
                denom = log_scales[l] - log_scales[j]
                gamma_formula = (
                    2.0 * math.log(max(d_il, 1e-12) / max(d_ij, 1e-12)) / denom
                    if abs(denom) > tolerance
                    else float("nan")
                )
                same_removed = bool(labels[j] == labels[i])
                same_added = bool(labels[l] == labels[i])
                transition = "{}->{}".format(
                    "same" if same_removed else "cross",
                    "same" if same_added else "cross",
                )
                # An undirected edge {i, j} disappears only if j does not
                # point back to i after the batch; {i, l} is new only if l did
                # not already point to i before it.
                rows.append(
                    {
                        "gamma_star": gamma,
                        "gamma_star_formula": gamma_formula,
                        "query": i,
                        "removed": int(j),
                        "added": int(l),
                        "multi_swap": multi,
                        "source_label": labels[i],
                        "removed_label": labels[j],
                        "added_label": labels[l],
                        "source_pop_size": sizes[labels[i]],
                        "removed_pop_size": sizes[labels[j]],
                        "added_pop_size": sizes[labels[l]],
                        "removed_same_label": same_removed,
                        "added_same_label": same_added,
                        "delta_same_label": int(same_added) - int(same_removed),
                        "transition": transition,
                        "d_ij": d_ij,
                        "d_il": d_il,
                        "log_distance_ratio": math.log(
                            max(d_il, 1e-12) / max(d_ij, 1e-12)
                        ),
                        "log_sigma_ratio": float(denom),
                        "sigma_source": scales[i],
                        "sigma_removed": scales[j],
                        "sigma_added": scales[l],
                        "sigma_pct_source": sigma_pct[i],
                        "sigma_pct_removed": sigma_pct[j],
                        "sigma_pct_added": sigma_pct[l],
                        "indegree_removed_before": int(indegree[j]),
                        "indegree_added_before": int(indegree[l]),
                        "removed_is_hub": bool(indegree[j] >= hub_multiplier * k),
                        "added_is_hub": bool(indegree[l] >= hub_multiplier * k),
                        "removed_changes_undirected": i not in after[j],
                        "added_changes_undirected": i not in neighbours[l],
                    }
                )

        for event in batch:
            for target in event["removed"]:
                indegree[target] -= 1
            for target in event["added"]:
                indegree[target] += 1
        neighbours = after

    return pd.DataFrame(rows)


def purity_trajectory(swaps, labels, gamma_grid=None):
    """Cumulative same-label out-edge change per cell, by source population.

    Returned at every distinct gamma* (exact step function) unless
    ``gamma_grid`` is given, in which case it is evaluated on that grid.
    Includes an ``__all__`` row group for the whole dataset.
    """
    labels = np.asarray(labels)
    sizes = Counter(labels.tolist())
    sizes["__all__"] = len(labels)
    frames = []
    groups = [("__all__", swaps)] + list(swaps.groupby("source_label"))
    for label, group in groups:
        group = group.sort_values("gamma_star")
        gammas = np.concatenate([[0.0], group["gamma_star"].to_numpy(), [1.0]])
        cumulative = np.concatenate(
            [[0], np.cumsum(group["delta_same_label"].to_numpy()), [0]]
        )
        cumulative[-1] = cumulative[-2]
        per_cell = cumulative / sizes[label]
        if gamma_grid is not None:
            idx = np.searchsorted(gammas, gamma_grid, side="right") - 1
            gammas, per_cell, cumulative = (
                np.asarray(gamma_grid),
                per_cell[idx],
                cumulative[idx],
            )
        frames.append(
            pd.DataFrame(
                {
                    "source_label": label,
                    "population_size": sizes[label],
                    "gamma": gammas,
                    "cumulative_delta_same": cumulative,
                    "delta_same_per_cell": per_cell,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def fragility_table(swaps, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Composition of swaps by gamma* bin (low gamma* = most fragile)."""
    swaps = swaps.copy()
    swaps["gamma_bin"] = pd.cut(swaps["gamma_star"], bins=list(bins), include_lowest=True)
    rows = []
    for interval, group in swaps.groupby("gamma_bin", observed=True):
        n = len(group)
        counts = group["transition"].value_counts()
        row = {
            "gamma_bin": str(interval),
            "gamma_lo": interval.left,
            "gamma_hi": interval.right,
            "swaps": n,
            "median_log_distance_ratio": group["log_distance_ratio"].median(),
            "median_log_sigma_ratio": group["log_sigma_ratio"].median(),
            "median_sigma_pct_removed": group["sigma_pct_removed"].median(),
            "median_sigma_pct_added": group["sigma_pct_added"].median(),
            "removed_hub_fraction": group["removed_is_hub"].mean(),
            "added_hub_fraction": group["added_is_hub"].mean(),
            "net_delta_same": int(group["delta_same_label"].sum()),
            "undirected_change_fraction": group["removed_changes_undirected"].mean(),
        }
        for t in TRANSITIONS:
            row["frac_" + t] = counts.get(t, 0) / n
        rows.append(row)
    return pd.DataFrame(rows)


def edge_flow_table(swaps):
    """For each source population: where are lost/gained edges re-routed?

    Counts swaps by (source, removed target population, added target
    population), restricted to label-changing swaps.
    """
    changing = swaps[swaps["transition"].isin(["same->cross", "cross->same"])]
    return (
        changing.groupby(["source_label", "transition", "removed_label", "added_label"])
        .size()
        .rename("swaps")
        .reset_index()
        .sort_values(["source_label", "swaps"], ascending=[True, False])
    )


def population_profile(swaps, labels, local_scales, neighbours0, gamma_hi=1.0):
    """Per-population density/size profile with the net purity change.

    ``neighbour_sigma_ratio`` compares the median sigma of the *cross-label*
    cells in a population's gamma=0 neighbourhoods with the population's own
    median sigma: > 1 means the adjacent populations are sparser, so local
    scaling (which favours large-sigma candidates) pulls edges out of it.
    """
    labels = np.asarray(labels)
    scales = np.asarray(local_scales, dtype=float)
    neighbours0 = np.asarray(neighbours0)
    n = len(labels)
    rows = []
    for label in sorted(np.unique(labels), key=str):
        members = np.flatnonzero(labels == label)
        targets = neighbours0[members].ravel()
        cross = targets[labels[targets] != label]
        # Candidate cross-label cells near this population: its members'
        # kNN at gamma=0 plus cells that point into it.
        incoming = np.flatnonzero(np.isin(neighbours0, members).any(axis=1))
        incoming = incoming[labels[incoming] != label]
        nearby = np.union1d(cross, incoming)
        own_sigma = float(np.median(scales[members]))
        group = swaps[swaps["source_label"] == label]
        rows.append(
            {
                "label": label,
                "population_size": len(members),
                "population_fraction": len(members) / n,
                "median_sigma": own_sigma,
                "median_sigma_pct": float(
                    np.median((rankdata(scales)[members] - 0.5) / n)
                ),
                "cross_edge_fraction_gamma0": len(cross) / max(len(targets), 1),
                "neighbour_sigma_ratio": (
                    float(np.median(scales[nearby])) / own_sigma
                    if len(nearby)
                    else float("nan")
                ),
                "swaps": len(group),
                "same_lost": int((group["transition"] == "same->cross").sum()),
                "same_gained": int((group["transition"] == "cross->same").sum()),
                "delta_same_per_cell": group["delta_same_label"].sum() / len(members),
            }
        )
    return pd.DataFrame(rows)
