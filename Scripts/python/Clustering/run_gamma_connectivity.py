"""Command-line entry point for adjustable local-scaling event analysis.

Input is an ``.npz`` file containing a two-dimensional ``X`` array and a
one-dimensional ``labels`` array. The representation should already be the
fixed PCA matrix used for the clustering experiment.
"""

import argparse
import os

import numpy as np
import pandas as pd

from gamma_connectivity import (
    neighbors_at_gamma,
    population_edge_table,
    run_event_analysis,
    summarize_gammas,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NPZ file containing X and labels arrays")
    parser.add_argument("output_dir")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--scale-rank", type=int, default=10)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    X = data["X"]
    labels = data["labels"]
    os.makedirs(args.output_dir, exist_ok=True)

    result = run_event_analysis(
        X,
        labels,
        k=args.k,
        candidate_pool_size=args.pool_size,
        local_scale_rank=args.scale_rank,
    )
    result["event_table"].to_csv(
        os.path.join(args.output_dir, "gamma_edge_events.csv"), index=False
    )
    summaries = summarize_gammas(
        result["candidate_indices"],
        result["candidate_distances"],
        result["local_scales"],
        args.k,
        args.gammas,
        labels=labels,
    )
    summaries.to_csv(
        os.path.join(args.output_dir, "gamma_graph_summaries.csv"), index=False
    )

    flow_tables = []
    for gamma in args.gammas:
        neighbours = neighbors_at_gamma(
            result["candidate_indices"],
            result["candidate_distances"],
            result["local_scales"],
            args.k,
            gamma,
        )
        flow_tables.append(population_edge_table(neighbours, labels, gamma=gamma))
    pd.concat(flow_tables, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "gamma_population_edges.csv"), index=False
    )


if __name__ == "__main__":
    main()
