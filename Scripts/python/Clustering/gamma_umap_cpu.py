"""CPU counterpart of modules/gamma_umap_embeddings.py.

Embeds each dataset once per gamma from the D_gamma kNN graph itself (UMAP
does not search for neighbours again), so differences between panels come
only from the graph. Uses the candidate pool saved by
run_h5ad_gamma_experiment.py.

python gamma_umap_cpu.py <experiment_root> <out_dir> --datasets Kumar ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_connectivity import neighbors_at_gamma  # noqa: E402

K = 15


def gamma_knn(cands, dists, scales, gamma):
    nbrs = neighbors_at_gamma(cands, dists, scales, K, gamma)
    n = len(scales)
    pos = [{c: p for p, c in enumerate(row)} for row in cands]
    d = np.array([[dists[i, pos[i][j]] for j in nbrs[i]] for i in range(n)])
    s = np.maximum(scales, 1e-12)
    # sqrt(D_gamma) so the embedding distances keep Euclidean units at gamma=0.
    dg = d / np.power(s[:, None] * s[nbrs], gamma / 2.0)
    idx = np.hstack([np.arange(n)[:, None], nbrs])
    dst = np.hstack([np.zeros((n, 1)), dg])
    order = np.argsort(dst, axis=1, kind="mergesort")
    return np.take_along_axis(idx, order, 1), np.take_along_axis(dst, order, 1)


def main():
    import umap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        root = args.experiment_root / name
        data = np.load(root / "pca_input.npz", allow_pickle=True)
        pool = np.load(root / "candidate_pool.npz")
        cands, dists, scales = (
            pool["candidate_indices"],
            pool["candidate_distances"],
            pool["local_scales"],
        )
        for gamma in args.gammas:
            idx, dst = gamma_knn(cands, dists, scales, gamma)
            model = umap.UMAP(
                n_neighbors=K + 1,
                precomputed_knn=(idx, dst, None),
                random_state=0,
                init="spectral",
            )
            emb = model.fit_transform(data["X_pca"])
            np.savez_compressed(
                args.out_dir / "{}_gamma{:.2f}.npz".format(name, gamma),
                embedding=emb,
                labels=data["labels"],
                local_scales=scales,
            )
            print(name, gamma, flush=True)


if __name__ == "__main__":
    main()
