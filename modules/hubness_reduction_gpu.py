"""CuPy ports of the scikit-hubness 0.21.2 reducers used by the GPU notebook.

Input rows contain sorted, self-free candidate neighbors. Distances returned here
are deliberately unsorted; the caller ranks candidates after reduction.
"""

import cupy as cp
from cupyx.scipy.special import ndtr


def reduce_mp_normal_gpu(dist, ind, X_gpu=None, params=None):
    if params and params.get("method", "normal") != "normal":
        raise ValueError("Only scikit-hubness MP normal is implemented")
    mu = cp.nanmean(dist, axis=1)
    sd = cp.nanstd(dist, axis=1)
    with cp.errstate(divide="ignore", invalid="ignore"):
        p1 = ndtr((mu[:, None] - dist) / sd[:, None])
        p2 = ndtr((mu[ind] - dist) / sd[ind])
    # scipy.stats.norm.sf returns nan for a zero standard deviation.
    return 1 - p1 * p2


def reduce_local_scaling_gpu(dist, ind, X_gpu=None, params=None):
    k = (params or {}).get("k", 5) + 1  # scikit-hubness includes index k
    radius = dist[:, :k][:, -1]
    with cp.errstate(divide="ignore", invalid="ignore"):
        return 1 - cp.exp(-dist ** 2 / (radius[:, None] * radius[ind]))


def reduce_nicdm_gpu(dist, ind, X_gpu=None, params=None):
    k = (params or {}).get("k", 5) + 1
    radius = dist[:, :k].mean(axis=1)
    with cp.errstate(divide="ignore", invalid="ignore"):
        return dist / cp.sqrt(radius[:, None] * radius[ind])


def reduce_dsl_gpu(dist, ind, X_gpu, params=None):
    params = params or {}
    k = min(params.get("k", 5), ind.shape[1])
    X = cp.asarray(X_gpu)
    centroids = X[ind[:, :k]].mean(axis=1)
    train_sq = cp.sum((X - centroids) ** 2, axis=1)
    test_centroids = centroids[ind[:, :k]].mean(axis=1)
    test_sq = cp.sum((X - test_centroids) ** 2, axis=1)
    # The reference recomputes squared Euclidean distances regardless of
    # the distances supplied by the neighbor search.
    norms = cp.sum(X * X, axis=1)
    sqdist = cp.empty(dist.shape, dtype=X.dtype)
    for start in range(0, X.shape[0], 256):
        stop = min(start + 256, X.shape[0])
        targets = X[ind[start:stop]]
        product = cp.einsum("bd,bkd->bk", X[start:stop], targets)
        sqdist[start:stop] = cp.maximum(
            norms[start:stop, None] + norms[ind[start:stop]] - 2 * product, 0)
    reduced = sqdist - test_sq[:, None] - train_sq[ind]
    reduced -= cp.minimum(reduced.min(), 0)
    if not params.get("squared", True):
        reduced = cp.sqrt(reduced)
    return reduced


def reduce_hubness_gpu(dist, ind, X_gpu, hubness, params=None):
    if hubness == "mp":
        return reduce_mp_normal_gpu(dist, ind, X_gpu, params)
    if hubness == "ls":
        if (params or {}).get("method") == "nicdm":
            return reduce_nicdm_gpu(dist, ind, X_gpu, params)
        return reduce_local_scaling_gpu(dist, ind, X_gpu, params)
    if hubness == "dsl":
        return reduce_dsl_gpu(dist, ind, X_gpu, params)
    raise ValueError("Unknown hubness reduction: " + str(hubness))
