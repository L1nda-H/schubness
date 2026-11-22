# Gamma connectivity analysis

`Scripts/python/Clustering/gamma_connectivity.py` implements the final-project
analysis of adjustable local scaling. It separates three objects that should
not be conflated:

1. pairwise candidate-score crossovers;
2. crossovers that change a directed top-k neighbour set; and
3. directed changes that add or remove an edge in the undirected graph used
   for clustering.

The event sequence is exact within the supplied Euclidean candidate pool.
Use `candidate_pool_size=None` for all-cell reranking on datasets where that is
computationally feasible. On larger datasets, compare pool sizes and measure
agreement with all-cell reranking on a representative subset of query cells.

## Notebook usage

```python
from gamma_connectivity import run_event_analysis, summarize_gammas

result = run_event_analysis(
    X_pca,
    reference_labels,
    k=15,
    candidate_pool_size=100,
    local_scale_rank=10,
)

event_table = result["event_table"]
summary = summarize_gammas(
    result["candidate_indices"],
    result["candidate_distances"],
    result["local_scales"],
    k=15,
    gammas=[0, 0.25, 0.5, 0.75, 1],
    labels=reference_labels,
)
```

Each event-table row reports the correction strength, source cell, removed and
added targets, change in same-label edges, endpoint scales and pre-event
in-degrees, pre-event hub involvement, and the number of changes that actually
alter the undirected graph.

## Command line

Save the fixed PCA representation and labels:

```python
np.savez("dataset_pca.npz", X=X_pca, labels=reference_labels)
```

Then run:

```bash
python Scripts/python/Clustering/run_gamma_connectivity.py \
  dataset_pca.npz results/dataset_name
```

This writes edge events, graph summaries, and population-to-population edge
counts as CSV files.

## Tests

From the repository root, with the `schubness-cs6850` environment active:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Next steps: per-swap analysis, simulations and figures

These scripts build on the files `run_h5ad_gamma_experiment.py` writes for
each dataset (`pca_input.npz`, `candidate_pool.npz`, `edge_events.csv`).

| Script | Purpose |
|---|---|
| `Scripts/R/export_duo_hvg10.R` | Exports the ExperimentHub HVG10 RDS files (EH1502-EH1546) to MatrixMarket using base R and `Matrix` only, without SingleCellExperiment |
| `Scripts/python/Clustering/mtx_to_h5ad.py` | Converts that export into `data/sce_filteredHVG10_<name>_gpu.h5ad` |
| `Scripts/python/Clustering/gamma_swaps.py` | One annotated row per swap `i: j -> l`, plus fragility, purity-trajectory, edge-flow and population-profile tables |
| `Scripts/python/Clustering/gamma_leiden.py` | Fixed-K Leiden (resolution bisected to the true K) and per-population F1 |
| `Scripts/python/Clustering/run_gamma_next_steps.py` | Runs all of the above on every dataset |
| `Scripts/python/Clustering/gamma_simulations.py` | Controlled 50-D Gaussian simulations (size x density factorial; density-gradient population) |
| `Scripts/python/Clustering/gamma_umap_cpu.py` | CPU UMAP of each D_gamma kNN graph (replaces the CUDA-only module) |
| `Scripts/python/Clustering/make_gamma_figures.py` | All figures in `results/gamma_next_steps/figures` |

Full reproduction, from the repository root:

```bash
for id in 1502 1506 1510 1514 1518 1522 1526 1530 1534 1538 1542 1546; do
  curl -L -o rds/EH$id.rds https://experimenthub.bioconductor.org/fetch/$id
done
Rscript Scripts/R/export_duo_hvg10.R rds export
python Scripts/python/Clustering/mtx_to_h5ad.py export data
python Scripts/python/Clustering/run_h5ad_gamma_experiment.py \
  --data-dir data --out-dir results/gamma_next_steps/experiment
python Scripts/python/Clustering/run_gamma_next_steps.py \
  results/gamma_next_steps/experiment results/gamma_next_steps/real
python Scripts/python/Clustering/gamma_simulations.py results/gamma_next_steps/simulations
python Scripts/python/Clustering/gamma_umap_cpu.py results/gamma_next_steps/experiment \
  results/gamma_next_steps/umap --datasets Kumar Zhengmix4eq Zhengmix4uneq
python Scripts/python/Clustering/make_gamma_figures.py results/gamma_next_steps
```

These were run with Python 3.12 (numpy 2, scanpy 1.12, leidenalg, umap-learn)
rather than the 3.7 conda environment; the gamma-connectivity numbers match
the original run to about three decimals.

**Note on SimKumar:** the earlier `results/gamma_connectivity` tables report
ARI near 0 for the three SimKumar datasets. With labels exported directly
from the RDS files, all three give ARI 1.0 and kNN purity at or above 0.995,
so the earlier H5AD conversion scrambled their labels.
