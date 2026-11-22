"""Roadmap next-steps analysis on top of run_h5ad_gamma_experiment.py output.

For every dataset directory produced by run_h5ad_gamma_experiment.py
(``pca_input.npz``, ``candidate_pool.npz``, ``edge_events.csv``) this writes:

swaps.csv.gz              one annotated row per swap i: j -> l at gamma*
fragility.csv             swap composition by gamma* bin
purity_trajectory.csv     exact per-population same-label change vs gamma
population_profile.csv    size / density / net purity change per population
edge_flow.csv             where label-changing swaps re-route edges
fixed_k_scores.csv        Leiden ARI with resolution tuned to the true K
population_recovery.csv   best-match F1 of each population in those partitions
population_split.csv      which reference populations Leiden splits, and
                          whether the split follows local scale sigma
pool_sensitivity.csv      (n <= --exhaustive-max cells) all-cell vs 100-pool

and dataset-level aggregates ``all_*.csv`` plus ``dataset_summary.csv`` in
the output root.

Usage:
python run_gamma_next_steps.py <experiment_root> <out_root> [--jobs 6]
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_connectivity import (  # noqa: E402
    build_candidate_pool,
    graph_summary,
    neighbors_at_gamma,
)
from gamma_leiden import _partition, leiden_fixed_k, union_graph  # noqa: E402
from gamma_swaps import (  # noqa: E402
    edge_flow_table,
    fragility_table,
    population_profile,
    purity_trajectory,
    swap_table,
)
from run_h5ad_gamma_experiment import leiden_scores  # noqa: E402

K = 15
GRID = tuple(np.round(np.linspace(0, 1, 11), 2))
FINE_GRID = np.round(np.linspace(0, 1, 101), 2)
SEEDS = (0, 1, 2, 3, 4)


def balanced_purity(neighbours, labels):
    """Mean over populations of the within-population kNN purity."""
    values = [
        np.mean(labels[neighbours[labels == lab]] == lab) for lab in np.unique(labels)
    ]
    return float(np.mean(values)), float(np.min(values))


def population_split(neighbours, labels, scales, gamma, seeds=SEEDS, min_share=0.1):
    """For each population: is it split across Leiden clusters (resolution 1)?

    A population counts as split when at least two clusters each hold
    >= ``min_share`` of it. For split populations, compares the local-scale
    percentile of the two largest parts (Mann-Whitney U).
    """
    graph = union_graph(neighbours)
    pct = (pd.Series(scales).rank().to_numpy() - 0.5) / len(scales)
    rows = []
    for seed in seeds:
        membership = _partition(graph, 1.0, seed)
        for lab in np.unique(labels):
            members = np.flatnonzero(labels == lab)
            counts = pd.Series(membership[members]).value_counts()
            parts = counts[counts >= min_share * len(members)]
            row = {
                "gamma": gamma,
                "seed": seed,
                "label": lab,
                "population_size": len(members),
                "n_parts": int(len(parts)),
                "split": bool(len(parts) >= 2),
                "largest_part_share": float(counts.iloc[0] / len(members)),
            }
            if len(parts) >= 2:
                a = members[membership[members] == parts.index[0]]
                b = members[membership[members] == parts.index[1]]
                row.update(
                    sigma_pct_part1=float(np.median(pct[a])),
                    sigma_pct_part2=float(np.median(pct[b])),
                    sigma_split_p=float(mannwhitneyu(pct[a], pct[b]).pvalue),
                )
            rows.append(row)
    return rows


def pool_sensitivity(X, labels, pool_cands, pool_dists, pool_scales):
    full = build_candidate_pool(X, candidate_pool_size=None, local_scale_rank=10)
    rows = []
    for gamma in GRID:
        for name, (c, d, s) in (
            ("pool100", (pool_cands, pool_dists, pool_scales)),
            ("all_cells", full),
        ):
            nbrs = neighbors_at_gamma(c, d, s, K, gamma)
            res1 = pd.DataFrame(leiden_scores(nbrs, labels, gamma, SEEDS))
            fk = pd.DataFrame(leiden_fixed_k(nbrs, labels, SEEDS))
            summary = graph_summary(nbrs, labels=labels)
            rows.append(
                {
                    "gamma": gamma,
                    "pool": name,
                    "ari_res1": res1["ari"].mean(),
                    "ari_fixed_k": fk["ari"].mean(),
                    "same_label_fraction": summary["directed_same_label_fraction"],
                    "in_degree_skewness": summary["in_degree_skewness"],
                }
            )
        a = neighbors_at_gamma(pool_cands, pool_dists, pool_scales, K, gamma)
        b = neighbors_at_gamma(*full, K, gamma)
        recall = np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)])
        rows[-1]["neighbour_recall_vs_pool"] = recall
        rows[-2]["neighbour_recall_vs_pool"] = recall
    return rows


def analyse(args):
    directory, out_root, exhaustive_max = args
    name = directory.name
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(directory / "pca_input.npz", allow_pickle=True)
    X, labels = data["X_pca"], data["labels"].astype(str)
    pool = np.load(directory / "candidate_pool.npz")
    cands, dists, scales = (
        pool["candidate_indices"],
        pool["candidate_distances"],
        pool["local_scales"],
    )
    events = pd.read_csv(directory / "edge_events.csv")
    nbrs0 = neighbors_at_gamma(cands, dists, scales, K, 0.0)

    swaps = swap_table(events, cands, dists, scales, nbrs0, labels, K)
    swaps.insert(0, "dataset", name)
    swaps.to_csv(out / "swaps.csv.gz", index=False)

    tables = {
        "fragility": fragility_table(swaps),
        "purity_trajectory": purity_trajectory(swaps, labels, gamma_grid=FINE_GRID),
        "population_profile": population_profile(swaps, labels, scales, nbrs0),
        "edge_flow": edge_flow_table(swaps),
    }

    fixed, splits, grid_rows = [], [], []
    for gamma in GRID:
        nbrs = neighbors_at_gamma(cands, dists, scales, K, gamma)
        for row in leiden_fixed_k(nbrs, labels, SEEDS, per_population=True):
            row["gamma"] = gamma
            fixed.append(row)
        splits.extend(population_split(nbrs, labels, scales, gamma))
        bal, worst = balanced_purity(nbrs, labels)
        grid_rows.append(
            {"gamma": gamma, "balanced_purity": bal, "worst_population_purity": worst}
        )
    fixed = pd.DataFrame(fixed)
    f1_cols = [c for c in fixed.columns if c.startswith("f1_")]
    recovery = fixed.melt(
        id_vars=["gamma", "seed"], value_vars=f1_cols, var_name="label", value_name="f1"
    )
    recovery["label"] = recovery["label"].str[3:]
    sizes = pd.Series(labels).value_counts()
    recovery["population_size"] = recovery["label"].map(sizes)
    tables["population_recovery"] = recovery
    tables["fixed_k_scores"] = fixed.drop(columns=f1_cols)
    tables["population_split"] = pd.DataFrame(splits)
    tables["balanced_purity"] = pd.DataFrame(grid_rows)

    if X.shape[0] <= exhaustive_max:
        tables["pool_sensitivity"] = pd.DataFrame(
            pool_sensitivity(X, labels, cands, dists, scales)
        )

    rho_d = spearmanr(swaps["gamma_star"], swaps["log_distance_ratio"]).correlation
    rho_s = spearmanr(swaps["gamma_star"], swaps["log_sigma_ratio"]).correlation
    tables["swap_correlations"] = pd.DataFrame(
        [
            {
                "spearman_gamma_vs_log_distance_ratio": rho_d,
                "spearman_gamma_vs_log_sigma_ratio": rho_s,
                "max_abs_formula_error": float(
                    np.nanmax(np.abs(swaps["gamma_star"] - swaps["gamma_star_formula"]))
                ),
                "multi_swap_rows": int(swaps["multi_swap"].sum()),
            }
        ]
    )
    for key, table in tables.items():
        table.insert(0, "dataset", name)
        table.to_csv(out / (key + ".csv"), index=False)
    print(name, "done", flush=True)
    return name


def dataset_summary(exp_root, out_root):
    rows = []
    for d in sorted(Path(out_root).glob("*/fixed_k_scores.csv")):
        name = d.parent.name
        g = pd.read_csv(Path(exp_root) / name / "gamma_summary.csv").set_index("gamma")
        fk = pd.read_csv(d).groupby("gamma")["ari"].mean()
        bal = pd.read_csv(d.parent / "balanced_purity.csv").set_index("gamma")
        prof = pd.read_csv(d.parent / "population_profile.csv")
        worst = prof.loc[prof["delta_same_per_cell"].idxmin()]
        rows.append(
            {
                "dataset": name,
                "cells": int(g["n_cells"].iloc[0]),
                "populations": len(prof),
                "ari_res1_g0": g.loc[0.0, "ari_mean"],
                "ari_res1_g1": g.loc[1.0, "ari_mean"],
                "best_gamma_res1": float(g["ari_mean"].idxmax()),
                "ari_fixed_k_g0": fk.loc[0.0],
                "ari_fixed_k_g1": fk.loc[1.0],
                "best_gamma_fixed_k": float(fk.idxmax()),
                "clusters_res1_g0": g.loc[0.0, "clusters_mean"],
                "clusters_res1_g1": g.loc[1.0, "clusters_mean"],
                "skewness_g0": g.loc[0.0, "in_degree_skewness"],
                "skewness_g1": g.loc[1.0, "in_degree_skewness"],
                "global_purity_g0": g.loc[0.0, "directed_same_label_fraction"],
                "global_purity_g1": g.loc[1.0, "directed_same_label_fraction"],
                "balanced_purity_g0": bal.loc[0.0, "balanced_purity"],
                "balanced_purity_g1": bal.loc[1.0, "balanced_purity"],
                "worst_population_purity_g0": bal.loc[0.0, "worst_population_purity"],
                "worst_population_purity_g1": bal.loc[1.0, "worst_population_purity"],
                "most_hurt_population": worst["label"],
                "most_hurt_delta_same_per_cell": worst["delta_same_per_cell"],
                "most_hurt_population_fraction": worst["population_fraction"],
                "most_hurt_neighbour_sigma_ratio": worst["neighbour_sigma_ratio"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--exhaustive-max", type=int, default=600)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    dirs = sorted(
        p.parent for p in args.experiment_root.glob("*/complete.json")
    )
    if args.datasets:
        dirs = [d for d in dirs if d.name in args.datasets]
    # Largest first so the long jobs start early.
    dirs.sort(key=lambda d: -(d / "edge_events.csv").stat().st_size)
    with Pool(args.jobs) as pool:
        pool.map(
            analyse,
            [(d, args.out_root, args.exhaustive_max) for d in dirs],
            chunksize=1,
        )
    for name in (
        "fragility",
        "purity_trajectory",
        "population_profile",
        "edge_flow",
        "fixed_k_scores",
        "population_recovery",
        "population_split",
        "balanced_purity",
        "pool_sensitivity",
        "swap_correlations",
    ):
        tables = [pd.read_csv(p) for p in sorted(args.out_root.glob("*/" + name + ".csv"))]
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(
                args.out_root / ("all_" + name + ".csv"), index=False
            )
    dataset_summary(args.experiment_root, args.out_root).to_csv(
        args.out_root / "dataset_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
