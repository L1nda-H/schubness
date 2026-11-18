"""Run the gamma-connectivity experiment on converted Duo H5AD datasets.

The script reproduces the exploratory preprocessing from the counts layer:
total-count normalization to 10,000, log1p, and 50-component PCA. It then
writes exact-within-pool edge events, graph summaries, Leiden scores,
population edge flows, conductance, and a sampled candidate-pool validation.
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "schubness_numba"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "schubness_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "schubness_cache"))

import anndata
import igraph
import leidenalg
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import pairwise_distances

from gamma_connectivity import (
    build_candidate_pool,
    enumerate_graph_events,
    annotate_graph_events,
    graph_summary,
    neighbors_at_gamma,
    population_conductance_table,
    population_edge_table,
)


DEFAULT_GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def dataset_name(path):
    name = Path(path).name
    prefix = "sce_filteredHVG10_"
    suffix = "_gpu.h5ad"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return Path(path).stem


def prepare_pca(path, n_comps=50, seed=0):
    backed = anndata.read_h5ad(path, backed="r")
    if "counts" not in backed.layers or "phenoid" not in backed.obs.columns:
        backed.file.close()
        raise ValueError("H5AD must contain layers['counts'] and obs['phenoid']")
    counts = np.asarray(backed.layers["counts"], dtype=np.float32)
    labels = backed.obs["phenoid"].astype(str).to_numpy()
    obs_names = np.asarray(backed.obs_names.astype(str))
    backed.file.close()

    adata = anndata.AnnData(X=sp.csr_matrix(counts))
    del counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    components = min(int(n_comps), adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(
        adata,
        n_comps=components,
        random_state=seed,
        svd_solver="arpack",
    )
    X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if not np.all(np.isfinite(X_pca)):
        raise ValueError("PCA produced non-finite coordinates")
    return X_pca, labels, obs_names


def undirected_edges(neighbours):
    edges = set()
    for source, targets in enumerate(neighbours):
        for target in targets:
            target = int(target)
            edges.add((min(source, target), max(source, target)))
    return sorted(edges)


def leiden_scores(neighbours, labels, gamma, seeds, resolution=1.0):
    graph = igraph.Graph(
        n=neighbours.shape[0], edges=undirected_edges(neighbours), directed=False
    )
    rows = []
    for seed in seeds:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=int(seed),
        )
        membership = np.asarray(partition.membership)
        rows.append(
            {
                "gamma": float(gamma),
                "seed": int(seed),
                "ari": adjusted_rand_score(labels, membership),
                "ami": adjusted_mutual_info_score(
                    labels, membership, average_method="arithmetic"
                ),
                "n_clusters": len(np.unique(membership)),
            }
        )
    return rows


def validate_candidate_pool(
    X,
    candidate_indices,
    candidate_distances,
    local_scales,
    k,
    gammas,
    sample_size=200,
    exhaustive_threshold=600,
    seed=0,
):
    rng = np.random.RandomState(seed)
    if X.shape[0] <= exhaustive_threshold:
        sample = np.arange(X.shape[0])
        query_scope = "all"
    else:
        sample = np.sort(
            rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
        )
        query_scope = "sample"
    full_distances = pairwise_distances(X[sample], X, metric="euclidean")
    rows = []
    safe_scales = np.maximum(local_scales, 1e-12)
    for gamma in gammas:
        pooled = neighbors_at_gamma(
            candidate_indices,
            candidate_distances,
            local_scales,
            k=k,
            gamma=gamma,
        )
        recalls = []
        exact = []
        for row, query in enumerate(sample):
            scores = np.square(full_distances[row]) / np.power(
                safe_scales[query] * safe_scales, gamma
            )
            scores[query] = np.inf
            # Stable full sort makes ties deterministic by cell index.
            exact_neighbours = np.argsort(scores, kind="mergesort")[:k]
            overlap = len(set(pooled[query]).intersection(exact_neighbours))
            recalls.append(overlap / k)
            exact.append(overlap == k)
        rows.append(
            {
                "gamma": float(gamma),
                "query_scope": query_scope,
                "sampled_queries": len(sample),
                "mean_neighbor_recall": float(np.mean(recalls)),
                "exact_set_fraction": float(np.mean(exact)),
                "minimum_neighbor_recall": float(np.min(recalls)),
            }
        )
    return pd.DataFrame(rows)


def _event_summary_row(table):
    if len(table) == 0:
        return {
            "event_batches": 0,
            "swap_count": 0,
            "delta_same_label": 0,
            "improving_event_fraction": float("nan"),
            "worsening_event_fraction": float("nan"),
            "neutral_event_fraction": float("nan"),
            "removed_hub_event_fraction": float("nan"),
            "added_hub_event_fraction": float("nan"),
            "effective_undirected_removed": 0,
            "effective_undirected_added": 0,
            "median_gamma": float("nan"),
        }
    delta = table["delta_same_label"]
    return {
        "event_batches": len(table),
        "swap_count": int(table["n_swaps"].sum()),
        "delta_same_label": int(delta.sum()),
        "improving_event_fraction": float(np.mean(delta > 0)),
        "worsening_event_fraction": float(np.mean(delta < 0)),
        "neutral_event_fraction": float(np.mean(delta == 0)),
        "removed_hub_event_fraction": float(table["removed_hub_before"].mean()),
        "added_hub_event_fraction": float(table["added_hub_before"].mean()),
        "effective_undirected_removed": int(
            table["removed_effective_undirected"].sum()
        ),
        "effective_undirected_added": int(
            table["added_effective_undirected"].sum()
        ),
        "median_gamma": float(table["gamma"].median()),
    }


def write_event_summaries(event_table, output):
    dataset = (
        str(event_table["dataset"].iloc[0]) if len(event_table) else output.name
    )
    overall = _event_summary_row(event_table)
    overall["dataset"] = dataset
    pd.DataFrame([overall]).to_csv(output / "event_summary.csv", index=False)
    rows = []
    for label, group in event_table.groupby("source_label", dropna=False):
        row = _event_summary_row(group)
        row.update({"dataset": dataset, "source_label": label})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output / "event_summary_by_population.csv", index=False
    )


def run_dataset(
    path,
    out_root,
    gammas=DEFAULT_GAMMAS,
    n_comps=50,
    k=15,
    scale_rank=10,
    pool_size=100,
    seeds=(0, 1, 2, 3, 4),
    overwrite=False,
):
    name = dataset_name(path)
    output = Path(out_root) / name
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "complete.json"
    if completion.exists() and not overwrite:
        print("skip {}: complete".format(name), flush=True)
        return

    pca_path = output / "pca_input.npz"
    if pca_path.exists() and not overwrite:
        saved = np.load(pca_path, allow_pickle=True)
        X_pca = saved["X_pca"]
        labels = saved["labels"]
        obs_names = saved["obs_names"]
    else:
        print("{}: preprocessing".format(name), flush=True)
        X_pca, labels, obs_names = prepare_pca(path, n_comps=n_comps, seed=0)
        np.savez_compressed(
            pca_path, X_pca=X_pca, labels=labels, obs_names=obs_names
        )
    print("{}: {} cells, {} PCs".format(name, *X_pca.shape), flush=True)

    candidates, distances, scales = build_candidate_pool(
        X_pca,
        candidate_pool_size=min(pool_size, X_pca.shape[0] - 1),
        local_scale_rank=scale_rank,
    )
    np.savez_compressed(
        output / "candidate_pool.npz",
        candidate_indices=candidates,
        candidate_distances=distances,
        local_scales=scales,
    )
    initial = neighbors_at_gamma(candidates, distances, scales, k=k, gamma=0.0)
    print("{}: enumerating graph events".format(name), flush=True)
    events = enumerate_graph_events(candidates, distances, scales, k=k)
    event_table = annotate_graph_events(
        events, initial, labels=labels, local_scales=scales, k=k
    )
    event_table.insert(0, "dataset", name)
    event_table.to_csv(output / "edge_events.csv", index=False)
    write_event_summaries(event_table, output)

    summaries = []
    scores = []
    flows = []
    conductance = []
    for gamma in gammas:
        print("{}: gamma={:.2f}".format(name, gamma), flush=True)
        neighbours = neighbors_at_gamma(
            candidates, distances, scales, k=k, gamma=gamma
        )
        summary = graph_summary(neighbours, labels=labels)
        summary.update({"dataset": name, "gamma": float(gamma)})
        summaries.append(summary)
        scores.extend(leiden_scores(neighbours, labels, gamma, seeds))
        flow = population_edge_table(neighbours, labels, gamma=gamma)
        flow.insert(0, "dataset", name)
        flows.append(flow)
        population = population_conductance_table(neighbours, labels, gamma=gamma)
        population.insert(0, "dataset", name)
        conductance.append(population)

    score_table = pd.DataFrame(scores)
    score_table.insert(0, "dataset", name)
    summary_table = pd.DataFrame(summaries)
    means = score_table.groupby("gamma").agg(
        ari_mean=("ari", "mean"),
        ari_sd=("ari", "std"),
        ami_mean=("ami", "mean"),
        ami_sd=("ami", "std"),
        clusters_mean=("n_clusters", "mean"),
    ).reset_index()
    summary_table = summary_table.merge(means, on="gamma", how="left")
    summary_table.to_csv(output / "gamma_summary.csv", index=False)
    score_table.to_csv(output / "leiden_scores.csv", index=False)
    pd.concat(flows, ignore_index=True).to_csv(
        output / "population_edges.csv", index=False
    )
    pd.concat(conductance, ignore_index=True).to_csv(
        output / "population_conductance.csv", index=False
    )
    validation = validate_candidate_pool(
        X_pca,
        candidates,
        distances,
        scales,
        k=k,
        gammas=gammas,
    )
    validation.insert(0, "dataset", name)
    validation.to_csv(output / "candidate_pool_validation.csv", index=False)

    metadata = {
        "dataset": name,
        "cells": int(X_pca.shape[0]),
        "pca_components": int(X_pca.shape[1]),
        "k": int(k),
        "local_scale_rank": int(scale_rank),
        "candidate_pool_size": int(candidates.shape[1]),
        "event_batches": len(events),
        "gammas": list(map(float, gammas)),
        "leiden_seeds": list(map(int, seeds)),
    }
    with completion.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print("{}: complete ({} events)".format(name, len(events)), flush=True)
    del X_pca, labels, obs_names, candidates, distances, scales
    gc.collect()


def aggregate_results(out_root):
    out_root = Path(out_root)
    # Backfill summaries for completed datasets created by earlier versions of
    # this runner without repeating PCA or event enumeration.
    for event_path in sorted(out_root.glob("*/edge_events.csv")):
        output = event_path.parent
        if not (output / "event_summary.csv").exists():
            write_event_summaries(pd.read_csv(event_path), output)
    mappings = {
        "all_gamma_summaries.csv": "gamma_summary.csv",
        "all_leiden_scores.csv": "leiden_scores.csv",
        "all_candidate_pool_validations.csv": "candidate_pool_validation.csv",
        "all_population_edges.csv": "population_edges.csv",
        "all_population_conductance.csv": "population_conductance.csv",
        "all_event_summaries.csv": "event_summary.csv",
        "all_event_summaries_by_population.csv": "event_summary_by_population.csv",
    }
    for destination, source in mappings.items():
        tables = [pd.read_csv(path) for path in sorted(out_root.glob("*/" + source))]
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(
                out_root / destination, index=False
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/gamma_connectivity")
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--scale-rank", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("sce_filteredHVG10_*_gpu.h5ad"))
    if args.datasets:
        requested = set(args.datasets)
        files = [path for path in files if dataset_name(path) in requested]
    if not files:
        raise FileNotFoundError("no matching *_gpu.h5ad datasets found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        run_dataset(
            path,
            args.out_dir,
            gammas=tuple(args.gammas),
            n_comps=args.n_comps,
            k=args.k,
            scale_rank=args.scale_rank,
            pool_size=args.pool_size,
            seeds=tuple(args.seeds),
            overwrite=args.overwrite,
        )
    aggregate_results(args.out_dir)


if __name__ == "__main__":
    main()
