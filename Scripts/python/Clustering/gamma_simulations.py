"""Controlled simulations for adjustable local scaling.

Gaussian populations in a 50-dimensional space (high enough for kNN hubness
to appear) with independently controlled size, spread (density) and
separation. Each configuration runs the same pipeline as the real data
(Euclidean candidate pool of 100, sigma = distance to 10th neighbour, k = 15,
union graph, Leiden RBConfiguration at resolution 1, five seeds) on a gamma
grid, plus the exact per-swap event analysis for one replicate.

Scenarios
---------
pair      Two populations, factorial in size ratio and spread ratio of the
          focal population A relative to B. Isolates which property makes a
          population lose same-label edges under correction.
imbalance A slightly denser than B (spread 0.9x) and overlapping it, at
          equal size (Zhengmix4eq-like) or 1:10 (Zhengmix4uneq-like). The
          small dense population is then the confused side of the boundary.
gradient  One population with a dense core and a sparse halo around the same
          centre plus a second population. Tests whether full correction
          splits a population along its density gradient (Kumar pattern).

Usage
-----
python gamma_simulations.py <out_dir> [--replicates 3] [--scenarios ...] [--jobs 1]
"""

import argparse
import itertools
import os
from multiprocessing import Pool
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_connectivity import (  # noqa: E402
    build_candidate_pool,
    enumerate_graph_events,
    graph_summary,
    neighbors_at_gamma,
    population_conductance_table,
)
from gamma_swaps import fragility_table, population_profile, swap_table  # noqa: E402
from gamma_leiden import leiden_fixed_k  # noqa: E402
from run_h5ad_gamma_experiment import leiden_scores  # noqa: E402

GAMMA_GRID = tuple(np.round(np.linspace(0, 1, 11), 2))
DIM = 50


def gaussian_population(rng, n, centre, spread):
    return centre + spread * rng.standard_normal((n, len(centre)))


def simulate_pair(size_a, size_b, spread_a, spread_b, separation, seed):
    """Two populations whose centres are ``separation`` apart."""
    rng = np.random.RandomState(seed)
    centre_b = np.zeros(DIM)
    centre_a = np.zeros(DIM)
    centre_a[0] = separation
    X = np.vstack(
        [
            gaussian_population(rng, size_a, centre_a, spread_a),
            gaussian_population(rng, size_b, centre_b, spread_b),
        ]
    )
    labels = np.array(["A"] * size_a + ["B"] * size_b)
    return X, labels


def simulate_gradient(n_core, n_halo, n_other, halo_spread, separation, seed):
    """Population P = dense core + sparse halo (same centre); population Q."""
    rng = np.random.RandomState(seed)
    centre_q = np.zeros(DIM)
    centre_q[0] = separation
    X = np.vstack(
        [
            gaussian_population(rng, n_core, np.zeros(DIM), 1.0),
            gaussian_population(rng, n_halo, np.zeros(DIM), halo_spread),
            gaussian_population(rng, n_other, centre_q, 1.0),
        ]
    )
    labels = np.array(["P"] * (n_core + n_halo) + ["Q"] * n_other)
    compartment = np.array(
        ["P_core"] * n_core + ["P_halo"] * n_halo + ["Q"] * n_other
    )
    return X, labels, compartment


def per_population_purity(neighbours, labels):
    rows = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rows[label] = float(np.mean(labels[neighbours[members]] == label))
    return rows


def run_configuration(
    X,
    labels,
    meta,
    seeds=(0, 1, 2, 3, 4),
    k=15,
    pool=100,
    scale_rank=10,
    with_events=False,
    compartment=None,
):
    candidates, distances, scales = build_candidate_pool(
        X, candidate_pool_size=pool, local_scale_rank=scale_rank
    )
    grid_rows, cond_rows = [], []
    for gamma in GAMMA_GRID:
        nbrs = neighbors_at_gamma(candidates, distances, scales, k, gamma)
        row = graph_summary(nbrs, labels=labels)
        scores = pd.DataFrame(leiden_scores(nbrs, labels, gamma, seeds))
        fixed = pd.DataFrame(leiden_fixed_k(nbrs, labels, seeds, per_population=True))
        row.update(
            ari_fixed_k=fixed["ari"].mean(),
            ari_fixed_k_sd=fixed["ari"].std(),
            fixed_k_hit_rate=fixed["hit_target"].mean(),
            ari_mean=scores["ari"].mean(),
            ari_sd=scores["ari"].std(),
            ami_mean=scores["ami"].mean(),
            clusters_mean=scores["n_clusters"].mean(),
            gamma=float(gamma),
            **meta
        )
        for label, purity in per_population_purity(nbrs, labels).items():
            row["purity_" + str(label)] = purity
            row["f1_" + str(label)] = fixed["f1_" + str(label)].mean()
        if compartment is not None:
            row.update(_compartment_split(nbrs, labels, compartment, seeds))
        grid_rows.append(row)
        cond = population_conductance_table(nbrs, labels, gamma=gamma)
        for key, value in meta.items():
            cond[key] = value
        cond_rows.append(cond)
    out = {
        "grid": pd.DataFrame(grid_rows),
        "conductance": pd.concat(cond_rows, ignore_index=True),
    }
    if with_events:
        nbrs0 = neighbors_at_gamma(candidates, distances, scales, k, 0.0)
        events = enumerate_graph_events(candidates, distances, scales, k)
        swaps = swap_table(events, candidates, distances, scales, nbrs0, labels, k)
        frag = fragility_table(swaps)
        prof = population_profile(swaps, labels, scales, nbrs0)
        for table in (swaps, frag, prof):
            for key, value in meta.items():
                table[key] = value
        out.update(swaps=swaps, fragility=frag, profile=prof)
    return out


def _compartment_split(neighbours, labels, compartment, seeds):
    """How strongly does Leiden separate P_core from P_halo?

    Reports the ARI between Leiden clusters and the core/halo compartments
    restricted to population P (0 = not split along density, 1 = split
    exactly along it), and the fraction of P's out-edges that stay inside
    the same compartment.
    """
    from sklearn.metrics import adjusted_rand_score
    import igraph
    import leidenalg
    from run_h5ad_gamma_experiment import undirected_edges

    graph = igraph.Graph(
        n=neighbours.shape[0], edges=undirected_edges(neighbours), directed=False
    )
    in_p = labels == "P"
    split = []
    for seed in seeds:
        part = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=1.0,
            seed=int(seed),
        )
        membership = np.asarray(part.membership)
        split.append(adjusted_rand_score(compartment[in_p], membership[in_p]))
    members = np.flatnonzero(in_p)
    targets = neighbours[members]
    same_comp = compartment[targets] == compartment[members][:, None]
    p_targets = labels[targets] == "P"
    return {
        "core_halo_split_ari": float(np.mean(split)),
        "within_compartment_fraction_P": float(
            np.sum(same_comp & p_targets) / np.sum(p_targets)
        ),
    }


def pair_scenario(replicates):
    """Factorial: focal population A vs reference B (500 cells, spread 1)."""
    size_ratios = (1.0, 0.25)
    spread_ratios = (0.75, 1.0, 1.33)
    for size_ratio, spread_ratio, rep in itertools.product(
        size_ratios, spread_ratios, range(replicates)
    ):
        size_a = int(round(500 * size_ratio))
        meta = {
            "scenario": "pair",
            "size_ratio": size_ratio,
            "spread_ratio": spread_ratio,
            "replicate": rep,
            "config": "A size x{:g}, spread x{:g}".format(size_ratio, spread_ratio),
        }
        # Separation scales with the mean spread so all configurations have a
        # comparable amount of overlap at gamma = 0.
        # 6 x mean spread gives partial overlap at gamma = 0 (kNN purity of the
        # sparser population ~0.5-0.9) without merging the populations.
        separation = 6.0 * (1.0 + spread_ratio) / 2.0
        X, labels = simulate_pair(size_a, 500, spread_ratio, 1.0, separation, seed=rep)
        yield X, labels, meta, None


def imbalance_scenario(replicates):
    """A slightly denser than B and overlapping; balanced vs 1:10."""
    for (size_a, size_b), rep in itertools.product(
        ((550, 550), (100, 1000)), range(replicates)
    ):
        meta = {
            "scenario": "imbalance",
            "size_a": size_a,
            "size_b": size_b,
            "replicate": rep,
            "config": "A:B = {}:{}".format(size_a, size_b),
        }
        X, labels = simulate_pair(size_a, size_b, 0.9, 1.0, 3.5, seed=200 + rep)
        yield X, labels, meta, None


def gradient_scenario(replicates):
    for halo_spread, rep in itertools.product((1.0, 1.3, 1.6), range(replicates)):
        meta = {
            "scenario": "gradient",
            "halo_spread": halo_spread,
            "replicate": rep,
            "config": "halo spread x{:g}".format(halo_spread),
        }
        X, labels, compartment = simulate_gradient(
            300, 300, 400, halo_spread, separation=6.0, seed=100 + rep
        )
        yield X, labels, meta, compartment


def _run(task):
    X, labels, meta, compartment = task
    print(meta["scenario"], meta["config"], "rep", meta["replicate"], flush=True)
    return run_configuration(
        X,
        labels,
        meta,
        with_events=meta["replicate"] == 0,
        compartment=compartment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--scenarios", nargs="+", default=["pair", "imbalance", "gradient"]
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "pair": pair_scenario,
        "imbalance": imbalance_scenario,
        "gradient": gradient_scenario,
    }
    tasks = [
        task
        for name in args.scenarios
        for task in generators[name](args.replicates)
    ]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(_run, tasks, chunksize=1)
    else:
        results = [_run(task) for task in tasks]
    collected = {}
    for result in results:
        for key, table in result.items():
            collected.setdefault(key, []).append(table)
    for key, tables in collected.items():
        pd.concat(tables, ignore_index=True).to_csv(
            args.out_dir / "sim_{}.csv".format(key), index=False
        )


if __name__ == "__main__":
    main()
