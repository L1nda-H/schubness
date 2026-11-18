# Gamma connectivity experiment summary

## Completed analysis

All 12 DuoClustering2018 filteredHVG10 H5AD datasets were processed from the
`counts` layer using total-count normalization to 10,000, `log1p`, and a fixed
50-component PCA representation. Directed 15-nearest-neighbour graphs used
the 100 closest Euclidean candidates and local-scale rank 10. Exact
top-k-changing crossover batches were enumerated for gamma in (0, 1), and
Leiden clustering was evaluated at gamma 0, 0.25, 0.5, 0.75, and 1 with five
seeds and resolution 1.0.

The result directory contains per-dataset edge events, graph summaries,
Leiden scores, population-to-population edge counts, population conductance,
and candidate-pool validation, together with aggregate CSV files.

## Clustering results

| Dataset | ARI gamma=0 | ARI gamma=1 | Change |
|---|---:|---:|---:|
| Koh | 0.975 | 0.979 | +0.004 |
| KohTCC | 0.975 | 0.966 | -0.009 |
| Kumar | 0.989 | 0.863 | -0.125 |
| KumarTCC | 0.977 | 0.871 | -0.106 |
| SimKumar4easy | -0.001 | -0.000 | +0.001 |
| SimKumar4hard | 0.000 | 0.001 | +0.001 |
| SimKumar8hard | -0.001 | -0.000 | +0.001 |
| Trapnell | 0.264 | 0.288 | +0.024 |
| TrapnellTCC | 0.267 | 0.285 | +0.018 |
| Zhengmix4eq | 0.808 | 0.882 | +0.074 |
| Zhengmix4uneq | 0.688 | 0.584 | -0.104 |
| Zhengmix8eq | 0.508 | 0.561 | +0.053 |

Kumar reproduces the exploratory pattern: ARI is 0.989 at gamma 0, reaches
1.000 at gamma 0.5 and 0.75, and drops to 0.863 at gamma 1 even as degree
inequality continues to fall.

## Main connectivity result

Overall directed same-label edge fraction increases in both Zhengmix4eq and
Zhengmix4uneq. It therefore does not by itself explain their opposing ARI
responses.

The population-level changes reveal the imbalance:

- In Zhengmix4eq, naive-cytotoxic cells lose 592 same-label outgoing edges
  across 789 cells (0.75 per cell), while regulatory-T cells gain 1,359 across
  830 cells (1.64 per cell). Conductance falls from 0.139 to 0.107 for
  naive-cytotoxic cells and from 0.157 to 0.105 for regulatory-T cells.
- In Zhengmix4uneq, the small naive-cytotoxic population loses 577 same-label
  outgoing edges across only 364 cells (1.59 per cell), while the large
  regulatory-T population gains 1,209 across 2,332 cells (0.52 per cell).
  Naive-cytotoxic conductance remains high, changing from 0.303 to 0.279,
  while regulatory-T conductance improves from 0.058 to 0.046.

Thus, correction improves the population-weighted global purity statistic
while disproportionately damaging the outgoing neighbourhoods of the small
naive-cytotoxic population. This is a plausible mechanism for the ARI decline
in Zhengmix4uneq and supports reporting population-balanced metrics rather
than only global edge counts.

Hub-related event annotations support the intended correction mechanism. In
the three Zheng mixtures, 46-49% of event batches remove an edge to a cell
that is a hub immediately before the event, whereas only 7-8% add an edge to
a current hub. The downstream outcome depends on which populations receive
the replacement edges, not just on whether hubs lose edges.

## Candidate-pool validation

The 100-candidate approximation was compared with all-cell reranking at the
five gamma checkpoints. Every query was evaluated for datasets with at most
600 cells; 200 fixed queries were evaluated for each Zheng mixture.

At gamma 1, mean neighbour recall was 0.990-0.995 for the Zheng mixtures and
0.994 for Kumar. It was lower for KohTCC (0.933), TrapnellTCC (0.919), and
KumarTCC (0.963), with a few individual queries having very low recall. The
main Zheng comparison is therefore well supported by the candidate pool, but
the TCC and Koh/Trapnell results should receive a larger-pool or all-cell
sensitivity analysis before strong conclusions are drawn.

## Remaining project components

The real-data connectivity experiment is complete. The controlled Gaussian
mixture simulations described in the roadmap and the GPU UMAP embedding run
are separate remaining components. The missing `modules/gamma_scaling.py`
dependency has been added and the GPU embedding module now points to the
top-level `data` directory, but the local environment does not contain CUDA,
CuPy, cuML, or rapids-singlecell.
