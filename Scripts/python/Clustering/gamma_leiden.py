"""Leiden clustering helpers for the gamma experiments.

``leiden_scores`` in run_h5ad_gamma_experiment.py uses a fixed resolution of
1.0, so a change in ARI across gamma can come from the number of clusters
changing rather than from which cells are grouped together. ``leiden_fixed_k``
removes that confound the way the DuoClustering2018 benchmark does: for each
seed it bisects the resolution until Leiden returns the true number of
populations, then scores that partition.
"""

import igraph
import leidenalg
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def union_graph(neighbours):
    neighbours = np.asarray(neighbours)
    n, k = neighbours.shape
    src = np.repeat(np.arange(n), k)
    dst = neighbours.ravel()
    pairs = np.unique(np.sort(np.stack([src, dst], axis=1), axis=1), axis=0)
    return igraph.Graph(n=n, edges=pairs.tolist(), directed=False)


def _partition(graph, resolution, seed):
    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=int(seed),
    )
    return np.asarray(part.membership)


def population_f1(labels, membership):
    """Best-match F1 of each reference population against the clusters."""
    labels = np.asarray(labels)
    out = {}
    for lab in np.unique(labels):
        in_pop = labels == lab
        best = 0.0
        for cluster in np.unique(membership[in_pop]):
            in_cluster = membership == cluster
            tp = np.sum(in_pop & in_cluster)
            best = max(best, 2.0 * tp / (in_pop.sum() + in_cluster.sum()))
        out[lab] = best
    return out


def leiden_fixed_k(neighbours, labels, seeds=(0, 1, 2, 3, 4), n_clusters=None,
                   low=1e-3, high=10.0, max_iter=40, per_population=False):
    """Return per-seed rows with ARI/AMI at the resolution giving n_clusters.

    If no resolution hits the target exactly, the closest partition found is
    used and ``hit_target`` is False. With ``per_population`` each row also
    carries ``f1_<label>``, the best-match F1 of every reference population
    (sensitive to small populations, unlike ARI).
    """
    labels = np.asarray(labels)
    target = len(np.unique(labels)) if n_clusters is None else int(n_clusters)
    graph = union_graph(neighbours)
    rows = []
    for seed in seeds:
        lo, hi = np.log(low), np.log(high)
        best = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            membership = _partition(graph, float(np.exp(mid)), seed)
            found = len(np.unique(membership))
            gap = abs(found - target)
            if best is None or gap < best[0]:
                best = (gap, membership, float(np.exp(mid)))
            if found == target:
                break
            if found > target:
                hi = mid
            else:
                lo = mid
        _, membership, resolution = best
        extra = {}
        if per_population:
            extra = {
                "f1_" + str(lab): value
                for lab, value in population_f1(labels, membership).items()
            }
        rows.append(
            dict(extra, **{
                "seed": int(seed),
                "resolution": resolution,
                "n_clusters": int(len(np.unique(membership))),
                "hit_target": bool(best[0] == 0),
                "ari": adjusted_rand_score(labels, membership),
                "ami": adjusted_mutual_info_score(
                    labels, membership, average_method="arithmetic"
                ),
            })
        )
    return rows
