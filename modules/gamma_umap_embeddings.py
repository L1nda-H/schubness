"""
2D UMAP embeddings of the DuoClustering2018 HVG10 datasets for each gamma.

PCA is computed once per dataset. Each gamma rebuilds the neighbour graph
with D_gamma and embeds every cell from that graph. cuML UMAP does not
search for neighbours again, so different gammas produce different embeddings.

Requires the Colab GPU stack (CuPy, cuML, rapids-singlecell), not the CPU
environment. Upload this file and gamma_scaling.py to the same Drive folder.

    import sys
    sys.path.insert(0, "/content/drive/MyDrive/.../modules")
    import gamma_umap_embeddings as emb
    emb.run(
        "/content/drive/MyDrive/.../data",
        "/content/drive/MyDrive/.../results/gamma_umap",
    )

Each output is a compressed npz with embedding (n_cells, 2), phenoid, and
obs_names, in the same cell order as the input h5ad.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gamma_scaling as gs


GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_DATA = (
    Path(__file__).resolve().parent
    .parent
    / "data"
)


def prepare_gpu(adata, n_comps=50, seed=0):
    """Log-normalize counts and reduce to n_comps principal components.

    The matrix is fixed across gamma values. This is the GPU counterpart of
    run_sweep.prepare.
    """
    import cupy as cp
    import rapids_singlecell as rsc

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    adata.layers.clear()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    n_pc = min(n_comps, adata.n_vars - 1, adata.n_obs - 1)
    if n_pc < 1:
        raise ValueError("PCA needs at least two cells and genes")
    rsc.pp.pca(adata, n_comps=n_pc, key_added="X_pca", mask_var=None, rng=seed)
    return cp.asarray(adata.obsm["X_pca"], dtype=cp.float32)


def candidate_neighbors(X, candidate_k=100, metric="euclidean"):
    """Exact self-free neighbours on GPU. Independent of gamma."""
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    n = X.shape[0]
    pool = min(candidate_k, n - 1)
    model = NearestNeighbors(
        n_neighbors=pool + 1, algorithm="brute", metric=metric, output_type="cupy")
    model.fit(X)
    dist, ind = model.kneighbors(X)
    dist = cp.asarray(dist)
    ind = cp.asarray(ind).astype(cp.int32)
    rows = cp.arange(n, dtype=cp.int32)[:, None]
    keep = cp.argsort(cp.where(ind == rows, cp.inf, dist), axis=1)[:, :pool]
    dist = cp.take_along_axis(dist, keep, axis=1)
    ind = cp.take_along_axis(ind, keep, axis=1)
    if bool(cp.any(ind == rows)):
        raise RuntimeError("self neighbour survived the filter")
    return dist, ind


def gamma_graph(dist, ind, k, gamma, k_sigma):
    """Top-k neighbours under D_gamma, still without the self column."""
    import cupy as cp

    reduced = gs.gamma_scaled(dist, ind, gamma, k_sigma)
    order = cp.argsort(reduced, axis=1)[:, :k]
    chosen = cp.take_along_axis(ind, order, axis=1)
    values = cp.take_along_axis(reduced, order, axis=1)
    return chosen, values


def embed_umap(X, ind, dist, seed=0):
    """Embed every row of X from the gamma graph.

    cuML requires a self neighbour in column 0. The extra column is distance
    0, so the k gamma neighbours are the neighbours that determine the embedding.
    """
    import cupy as cp
    from cuml.manifold import UMAP

    n = ind.shape[0]
    self_ind = cp.arange(n, dtype=cp.int32)[:, None]
    self_dist = cp.zeros((n, 1), dtype=cp.float32)
    knn_ind = cp.concatenate([self_ind, ind.astype(cp.int32)], axis=1)
    knn_dist = cp.concatenate([self_dist, dist.astype(cp.float32)], axis=1)
    knn_ind = np.ascontiguousarray(cp.asnumpy(knn_ind))
    knn_dist = np.ascontiguousarray(cp.asnumpy(knn_dist))
    reducer = UMAP(
        n_neighbors=knn_ind.shape[1],
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        precomputed_knn=(knn_ind, knn_dist),
        output_type="numpy",
    )
    embedding = reducer.fit_transform(X)
    return np.asarray(cp.asnumpy(cp.asarray(embedding)), dtype=np.float32)


def output_path(out_dir, stem, n_comps, k_graph, k_sigma, seed, gamma):
    name = (
        f"{stem}_pca{n_comps}_k{k_graph}_ksigma{k_sigma}"
        f"_seed{seed}_gamma{gamma:.2f}_umap.npz"
    )
    return Path(out_dir) / name


def embed_dataset(path, out_dir, gammas=GAMMAS, n_comps=50, k_graph=15,
                  k_sigma=10, candidate_k=100, seed=0, overwrite=False):
    """Write one embedding per gamma for a converted *_gpu.h5ad file."""
    import anndata
    import cupy as cp

    path = Path(path)
    stem = path.name.removesuffix("_gpu.h5ad").removesuffix(".h5ad")
    pending = [
        g for g in gammas
        if overwrite or not output_path(
            out_dir, stem, n_comps, k_graph, k_sigma, seed, g).exists()
    ]
    if not pending:
        print(f"skip {path.name}: all gammas already saved", flush=True)
        return []

    adata = anndata.read_h5ad(path)
    phenoid = (adata.obs["phenoid"].astype(str).to_numpy()
               if "phenoid" in adata.obs.columns else None)
    obs_names = np.asarray(adata.obs_names.astype(str))
    print(f"{path.name}: {adata.n_obs} cells, {adata.n_vars} genes", flush=True)
    X = prepare_gpu(adata, n_comps=n_comps, seed=seed)
    if X.shape[0] <= k_graph:
        raise ValueError(f"{path.name} has too few cells for k_graph={k_graph}")

    pool = max(k_graph, candidate_k, k_sigma)
    dist, ind = candidate_neighbors(X, candidate_k=pool)
    written = []
    for gamma in pending:
        print(f"  gamma={gamma:.2f}", flush=True)
        chosen, values = gamma_graph(dist, ind, k_graph, gamma, k_sigma)
        embedding = embed_umap(X, chosen, values, seed=seed)
        if embedding.shape != (X.shape[0], 2):
            raise RuntimeError(
                f"expected {(X.shape[0], 2)} embedding, got {embedding.shape}")
        destination = output_path(
            out_dir, stem, n_comps, k_graph, k_sigma, seed, gamma)
        temporary = destination.with_suffix(".tmp.npz")
        payload = dict(embedding=embedding, obs_names=obs_names, gamma=np.float64(gamma))
        if phenoid is not None:
            payload["phenoid"] = phenoid
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
        written.append(destination)
        del chosen, values, embedding
        cp.get_default_memory_pool().free_all_blocks()
    del adata, X, dist, ind
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return written


def run(data_dir, out_dir, gammas=GAMMAS, n_comps=50, k_graph=15, k_sigma=10,
        candidate_k=100, seed=0, limit=None, overwrite=False):
    """Embed every converted HVG10 dataset in data_dir."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    files = sorted(data_dir.glob("*sce*_gpu.h5ad"))
    if not files:
        raise FileNotFoundError(
            f"No *_gpu.h5ad files in {data_dir}. Run convert_rds_gpu.py first.")
    if limit is not None:
        files = files[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in files:
        written.extend(embed_dataset(
            path, out_dir, gammas=gammas, n_comps=n_comps, k_graph=k_graph,
            k_sigma=k_sigma, candidate_k=candidate_k, seed=seed,
            overwrite=overwrite))
    print(f"Wrote {len(written)} embeddings to {out_dir}", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--k-graph", type=int, default=15)
    parser.add_argument("--k-sigma", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N datasets, for a short Colab check.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, gammas=tuple(args.gammas),
        n_comps=args.n_comps, k_graph=args.k_graph, k_sigma=args.k_sigma,
        candidate_k=args.candidate_k, seed=args.seed, limit=args.limit,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
