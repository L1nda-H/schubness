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
