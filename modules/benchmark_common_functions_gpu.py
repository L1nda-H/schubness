import scanpy as sc
import anndata
import leidenalg
import igraph
import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew
import sklearn.metrics.cluster
import sklearn.cluster
import gc
import time
import array
import csv
import warnings

warnings.filterwarnings("ignore")


def _gpu_modules():
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import rapids_singlecell as rsc
    return cp, cpx_sparse, rsc


class IncrementalCSRMatrix(object):
    """modified from https://maciejkula.github.io/2015/02/22/incremental-construction-of-sparse-matrices/"""

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = "i"
        elif dtype is np.int64:
            type_flag = "l"
        elif dtype is np.float32:
            type_flag = "f"
        elif dtype is np.float64:
            type_flag = "d"
        else:
            raise Exception("Dtype not supported.")

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array("i")
        self.cols = array.array("i")
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if i >= m or j >= n:
            raise Exception("Index out of bounds")

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocsr(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=self.shape)


def get_csr(f, shape):
    # shape = (n_dims, n_samples)
    mat = IncrementalCSRMatrix(shape, np.float32)

    for row_ind, line in enumerate(f, start=-1):
        print(row_ind, end="\r")
        if row_ind == -1:
            continue

        data = np.array(line[1:], dtype=float)
        idx = data.nonzero()[0]
        for col_ind in idx:
            mat.append(row_ind, col_ind, data[col_ind])

    return mat


def color_sign_df(value):
    """
    Colors elements in a dateframe
    green if positive and red if
    negative. Does not color NaN
    values.
    """

    if value < 0:
        color = "red"
    elif value > 0:
        color = "green"
    else:
        color = "black"

    return "color: %s" % color


def angular_dist(u, v, w=None):
    cosine_dist = scipy.spatial.distance.cosine(u, v, w)
    cosine_sim = -cosine_dist + 1
    return np.arccos(cosine_sim) / np.pi


def get_complexity(data, labels):
    """ Calculate dataset complexity as defined in Abdelaal et al."""

    unique_types = np.unique(labels)

    # calculate avg expression of each cell type
    avg_exprs = np.zeros((len(unique_types), data.shape[1]))
    for i, cell_type in enumerate(unique_types):
        avg_exprs[i, :] = np.mean(data[np.where(labels == cell_type)], axis=0)

    # correlation between avg expression of each cell type
    coeffs = np.corrcoef(avg_exprs)
    np.fill_diagonal(coeffs, 0)
    # mean of max correlation
    complexity = np.mean(np.max(coeffs, axis=1))
    return complexity


def get_scores(true_labels, res_dict, retained_cells_idx, X, metric, seed):
    keys_list = list(res_dict.keys())

    df = pd.concat(
        (
            # supervised
            pd.Series(
                data=[
                    sklearn.metrics.cluster.adjusted_rand_score(
                        true_labels, res_dict[k][retained_cells_idx]
                    )
                    for k in keys_list
                ],
                index=keys_list,
            ),
            pd.Series(
                data=[
                    sklearn.metrics.cluster.adjusted_mutual_info_score(
                        true_labels,
                        res_dict[k][retained_cells_idx],
                        average_method="arithmetic",
                    )
                    for k in keys_list
                ],
                index=keys_list,
            ),
            pd.Series(
                data=[
                    sklearn.metrics.cluster.homogeneity_score(
                        true_labels, res_dict[k][retained_cells_idx]
                    )
                    for k in keys_list
                ],
                index=keys_list,
            ),
            pd.Series(
                data=[
                    sklearn.metrics.cluster.completeness_score(
                        true_labels, res_dict[k][retained_cells_idx]
                    )
                    for k in keys_list
                ],
                index=keys_list,
            ),
            pd.Series(
                data=[
                    sklearn.metrics.cluster.v_measure_score(
                        true_labels, res_dict[k][retained_cells_idx]
                    )
                    for k in keys_list
                ],
                index=keys_list,
            ),
            # unsupervised
            # pd.Series(data=[sklearn.metrics.cluster.davies_bouldin_score(X[retained_cells_idx],res_dict[k][retained_cells_idx]) for k in keys_list],index=keys_list),
            # pd.Series(data=[sklearn.metrics.cluster.calinski_harabasz_score(X[retained_cells_idx],res_dict[k][retained_cells_idx]) for k in keys_list],index=keys_list),
            # pd.Series(data=[sklearn.metrics.cluster.silhouette_score(X,res_dict[k][retained_cells_idx],metric=metric,random_state=seed) for k in keys_list],index=keys_list)
        ),
        axis=1,
    )

    df.columns = [
        "ARI",
        "AMI",
        "Homogeneity",
        "Completeness",
        "V_measure",
    ]  # ,'Davies-Bouldin','Calinski-Harabasz']#,'Silhouette']

    return df


def parse_results_dict(results_dict):
    # extract main info
    hubness_dfs = {}
    clus = {}
    clus_scores = {}
    clus_info = {}
    for k, v in results_dict.items():
        hubness_dfs[k] = v["X_pca"]["hubness_df"]
        clus[k] = v["X_pca"]["clus"]
        clus_scores[k] = v["X_pca"]["clus_scores"]
        clus_info[k] = v["X_pca"]["clus_info"]

    # invert dicts of dicts to regroup by clustering or hubness method : {datasets:{methods}} -> {methods:{datasets}}
    get_key = lambda l, key: {k: v[key] for k, v in l.items()}

    hub_methods_keys = hubness_dfs[list(hubness_dfs.keys())[0]].columns
    clus_methods_keys = clus[list(clus.keys())[0]].keys()

    methods_clus = {}
    methods_clus_scores = {}
    methods_clus_info = {}

    for key in clus_methods_keys:
        methods_clus[key] = get_key(clus, key)
        methods_clus_scores[key] = get_key(clus_scores, key)
        methods_clus_info[key] = get_key(clus_info, key)

    methods_hubness_dfs = {key: get_key(hubness_dfs, key) for key in hub_methods_keys}

    return methods_hubness_dfs, methods_clus, methods_clus_scores, methods_clus_info


def load_rds(path, fname):
    import rpy2.robjects as ro  # optional legacy loader
    readRDS = ro.r["readRDS"]
    raw = readRDS(path + fname + ".rds")
    adata_dict = {k: v for k, v in raw.items()}
    adata = anndata.AnnData(X=adata_dict["counts"])
    adata.uns["Order"] = adata_dict["groups_id"].iloc[:, 1].values
    del raw, adata_dict
    return adata


def load_h5ad(path, fname):
    return anndata.read_h5ad(path + fname + ".h5ad")


# GPU entry points use the shared CPU scoring helper above.
from hubness_reduction_gpu import reduce_hubness_gpu


def _host_labels(labels):
    cp, _, _ = _gpu_modules()
    return cp.asnumpy(labels) if isinstance(labels, cp.ndarray) else np.asarray(labels)


_cpu_get_scores = get_scores


def get_scores(true_labels, res_dict, retained_cells_idx, X, metric, seed):
    host = {name: _host_labels(labels) for name, labels in res_dict.items()}
    return _cpu_get_scores(np.asarray(true_labels), host, retained_cells_idx,
                           None, metric, seed)


def recipe_duo(adata, do_log, renorm):
    cp, _, rsc = _gpu_modules()
    rsc.pp.normalize_total(adata, target_sum=1e4)
    if do_log:
        rsc.pp.log1p(adata)
    keep = cp.asnumpy(cp.argsort(cp.asarray(adata.X.mean(axis=0)).ravel())[::-1][:5000])
    adata._inplace_subset_var(keep)
    if renorm:
        rsc.pp.normalize_total(adata, target_sum=1e4)


def recipe_seurat(adata, do_log, norm_scale):
    _, _, rsc = _gpu_modules()
    rsc.pp.normalize_total(adata, target_sum=1e4)
    if do_log:
        rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, flavor="seurat", min_mean=0.0125,
                                  max_mean=3, min_disp=0.5)
    adata._inplace_subset_var(adata.var.highly_variable.to_numpy())
    if norm_scale:
        rsc.pp.scale(adata, max_value=10)


def _zheng17_gene_mask(X, n_top_genes):
    """Scanpy 1.7.2's raw-input Cell Ranger dispersion ranking on CuPy."""
    cp, cpx_sparse, _ = _gpu_modules()
    n = X.shape[0]
    mean = cp.asarray(X.sum(axis=0)).ravel() / n
    if cpx_sparse.issparse(X):
        second = cp.asarray(X.multiply(X).sum(axis=0)).ravel() / n
    else:
        second = cp.mean(X * X, axis=0)
    var = (second - mean * mean) * (n / (n - 1))
    disp = var / cp.where(mean == 0, 1e-12, mean)
    edges = cp.concatenate((cp.array([-cp.inf]),
                            cp.percentile(mean, cp.arange(10, 105, 5)),
                            cp.array([cp.inf])))
    if bool(cp.any(cp.diff(edges) <= 0)):
        raise ValueError("Cell Ranger mean bins have duplicate edges")
    bins = cp.searchsorted(edges[1:], mean, side="left")
    norm = cp.full(disp.shape, cp.nan, dtype=cp.float64)
    for b in range(len(edges) - 1):
        mask = bins == b
        if not bool(cp.any(mask)):
            continue
        vals = disp[mask]
        median = cp.median(vals)
        mad = cp.median(cp.abs(vals - median)) / 0.6744897501960817
        norm[mask] = cp.abs(vals - median) / mad
    finite = norm[~cp.isnan(norm)].astype(cp.float32)
    if finite.size < n_top_genes:
        raise ValueError("Too few finite Cell Ranger dispersion values")
    cutoff = cp.sort(finite)[-n_top_genes]
    return cp.asnumpy(norm >= cutoff)


def recipe_zheng17(adata, do_log, norm_scale, n_top_genes=1000):
    cp, _, rsc = _gpu_modules()
    adata.obs["n_counts_all"] = cp.asnumpy(cp.asarray(adata.X.sum(axis=1)).ravel())
    rsc.pp.normalize_total(adata)
    mask = _zheng17_gene_mask(adata.X, min(adata.n_vars, n_top_genes))
    adata._inplace_subset_var(mask)
    rsc.pp.normalize_total(adata)
    if do_log:
        rsc.pp.log1p(adata)
    if norm_scale:
        rsc.pp.scale(adata)


def _exact_candidates(X_gpu, metric, candidate_k):
    cp, _, _ = _gpu_modules()
    from cuml.neighbors import NearestNeighbors
    X = cp.asarray(X_gpu, dtype=cp.float32)
    n = X.shape[0]
    k = min(candidate_k, n - 1)
    model = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                             metric=metric, output_type="cupy")
    model.fit(X)
    dist, ind = model.kneighbors(X, n_neighbors=k + 1)
    dist, ind = cp.asarray(dist), cp.asarray(ind)
    rows = cp.arange(n)[:, None]
    order = cp.argsort(cp.where(ind == rows, cp.inf, dist), axis=1)[:, :k]
    dist = cp.take_along_axis(dist, order, axis=1)
    ind = cp.take_along_axis(ind, order, axis=1)
    if bool(cp.any(ind == rows)):
        raise RuntimeError("Could not remove self neighbors")
    return dist, ind


def _hubness_scores(final_ind, n_obs, k):
    cp, _, _ = _gpu_modules()
    from scipy import stats
    occ = cp.bincount(final_ind.ravel(), minlength=n_obs)
    counts = cp.asnumpy(occ).astype(np.int64)
    ordered = np.sort(counts)
    i = np.arange(1, n_obs + 1)
    gini = np.sum((2 * i - n_obs - 1) * ordered) / (n_obs * counts.sum())
    hubs = np.flatnonzero(counts >= 2 * k)
    antihubs = np.flatnonzero(counts == 0)
    std = counts.std(ddof=1)
    trunc = stats.truncnorm(-counts.mean() / std, np.inf).moment(3)
    return dict(k_skewness=float(stats.skew(counts)),
                k_skewness_truncnorm=float(trunc),
                atkinson=float(1 - np.mean(np.sqrt(counts)) ** 2 / counts.mean()),
                gini=float(gini),
                robinhood=float(0.5 * np.abs(counts - counts.mean()).sum() / counts.sum()),
                antihubs=antihubs,
                antihub_occurrence=float(len(antihubs) / n_obs),
                hubs=hubs,
                hub_occurrence=float(counts[hubs].sum() / (k * n_obs)),
                groupie_ratio=float(counts.max() / (k * n_obs)),
                k_occurrence=counts)


def generate_clustering_inputs2(X, metric, n_neighbors, weighted, seed,
                                hubness, hubness_params, candidate_cache=None,
                                stage_timings=None):
    cp, cpx_sparse, _ = _gpu_modules()
    if not 1 <= n_neighbors < X.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_obs - 1")
    if candidate_cache is None:
        candidate_cache = {}
    primary_metric = "euclidean" if hubness == "dsl" else metric
    candidate_k = min(max(n_neighbors, 100), X.shape[0] - 1)
    key = (primary_metric, candidate_k)
    if key not in candidate_cache:
        cp.cuda.Stream.null.synchronize()
        stage_start = time.perf_counter()
        candidate_cache[key] = _exact_candidates(X, primary_metric, candidate_k)
        cp.cuda.Stream.null.synchronize()
        if stage_timings is not None:
            stage_timings["candidate_search_" + primary_metric] = time.perf_counter() - stage_start
    dist, ind = candidate_cache[key]
    cp.cuda.Stream.null.synchronize()
    stage_start = time.perf_counter()
    if hubness is not None:
        reduced = reduce_hubness_gpu(dist, ind, X, hubness, hubness_params)
    else:
        reduced = dist
    order = cp.argsort(reduced, axis=1)[:, :n_neighbors]
    final_ind = cp.take_along_axis(ind, order, axis=1)
    final_dist = cp.take_along_axis(reduced, order, axis=1)
    cp.cuda.Stream.null.synchronize()
    if stage_timings is not None:
        name = hubness if hubness != "ls" else (hubness_params or {}).get("method", "ls")
        stage_timings["reduction_" + str(name)] = time.perf_counter() - stage_start
    scores = _hubness_scores(final_ind, X.shape[0], n_neighbors)
    rows = cp.repeat(cp.arange(X.shape[0]), n_neighbors)
    values = final_dist.ravel() if weighted else cp.ones(rows.size, dtype=cp.float32)
    if weighted and not bool(cp.all(cp.isfinite(values))):
        values = cp.ones(rows.size, dtype=cp.float32)
        weighted = False
    gpu_adj = cpx_sparse.csr_matrix((values, (rows, final_ind.ravel())),
                                     shape=(X.shape[0], X.shape[0]))
    cp.cuda.Stream.null.synchronize()
    stage_start = time.perf_counter()
    adjmat = gpu_adj.get()
    affinity_matrix = 0.5 * (adjmat + adjmat.T)
    coo = adjmat.tocoo()
    G = igraph.Graph(n=X.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())),
                     directed=True)
    G.es["weight"] = coo.data.tolist()
    weights = np.asarray(G.es["weight"], dtype=np.float64) if weighted else None
    if stage_timings is not None:
        stage_timings["graph_transfer_" + str(hubness or "base")] = time.perf_counter() - stage_start
    return adjmat, affinity_matrix, G, weights, scores


def getNclusters2(adata, G, n_clusters, seed, cluster_func, flavor, weights,
                  range_min=0, range_max=3, max_steps=20, neighbors_key=None,
                  scanpy_use_weights=None, scanpy_gpu=True):
    if cluster_func != "leiden":
        raise ValueError("cluster_func must be 'leiden'")
    _, _, rsc = _gpu_modules()
    low, high = float(range_min), float(range_max)
    for _ in range(max_steps):
        resolution = (low + high) / 2
        if flavor == "scanpy":
            if scanpy_gpu:
                rsc.tl.leiden(adata, resolution=resolution, rng=seed,
                              neighbors_key=neighbors_key, use_weights=scanpy_use_weights)
            else:
                sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                             neighbors_key=neighbors_key, use_weights=scanpy_use_weights)
            labels = adata.obs["leiden"].astype(int).to_numpy()
        elif flavor == "base":
            labels = np.asarray(leidenalg.find_partition(
                graph=G, partition_type=leidenalg.RBConfigurationVertexPartition,
                weights=weights, resolution_parameter=resolution,
                seed=seed).membership)
        else:
            raise ValueError("flavor must be scanpy or base")
        count = len(np.unique(labels))
        if count == n_clusters:
            return labels, dict(resolution=resolution, succeeded=True)
        if count > n_clusters:
            high = resolution
        else:
            low = resolution
    return labels, dict(resolution=resolution, succeeded=False)


def clustering_analysis(
    adata, true_labels, do_norm, norm_scale, do_log, do_pca,
    n_neighbors, n_clusters, metric, weighted, seed, n_comps=50,
    hubness_methods=None, retained_cells_idx=None,
):
    """GPU preprocessing and graph building with Leiden clustering."""
    cp, _, rsc = _gpu_modules()
    if hubness_methods is None:
        hubness_methods = {
            "mp_normal": ("mp", {"method": "normal"}),
            "ls": ("ls", None),
            "ls_nicdm": ("ls", {"method": "nicdm"}),
            "dsl": ("dsl", None),
        }
    if retained_cells_idx is None:
        retained_cells_idx = range(adata.n_obs)
    true_labels = np.asarray(true_labels)
    results = {"params": dict(do_norm=do_norm, norm_scale=norm_scale,
                              do_log=do_log, do_pca=do_pca,
                              n_neighbors=n_neighbors, n_clusters=n_clusters,
                              metric=metric, weighted=weighted, seed=seed,
                              n_comps=n_comps)}
    res = results["X_pca"] = {"clus": {}, "clus_info": {}, "clus_scores": {}}
    method_keys = ("scanpy_default_umap", "scanpy_default_gauss",
                   "base_default", "scanpy_umap", "scanpy_gauss", "base")
    for key in method_keys:
        res["clus"][key] = {}
        res["clus_info"][key] = {}
    timings = {}

    def tick(name, since):
        cp.cuda.Stream.null.synchronize()
        timings[name] = time.perf_counter() - since
        return time.perf_counter()

    start = time.perf_counter()
    rsc.get.anndata_to_GPU(adata)
    if isinstance(do_norm, str):
        if do_norm == "duo":
            recipe_duo(adata, do_log, norm_scale)
        elif do_norm == "seurat":
            recipe_seurat(adata, do_log, norm_scale)
        elif do_norm == "zheng17":
            recipe_zheng17(adata, do_log, norm_scale, n_top_genes=5000)
        else:
            raise ValueError("Unknown normalization: " + do_norm)
    elif do_log:
        rsc.pp.log1p(adata)
    if do_pca:
        n_pc = min(adata.n_vars - 1, adata.n_obs - 1, n_comps)
        if n_pc < 1:
            raise ValueError("PCA needs at least two cells and genes")
        rsc.pp.pca(adata, n_comps=n_pc, key_added="X_pca", mask_var=None, rng=seed)
    X_gpu = adata.obsm["X_pca"]
    start = tick("preprocessing_pca", start)

    scanpy_graphs = {}
    for method in ("umap", "gauss"):
        try:
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors + 1, metric=metric,
                             use_rep="X_pca", method=method, algorithm="brute",
                             key_added="neighbors_" + method, rng=seed)
            scanpy_graphs[method] = (adata, True)
        except ValueError:
            # Scanpy's dense knn=False graph is the reference fallback.
            cpu_adata = anndata.AnnData(X=cp.asnumpy(X_gpu))
            cpu_adata.obsm["X_pca"] = cpu_adata.X
            sc.pp.neighbors(cpu_adata, n_neighbors=n_neighbors + 1,
                            metric=metric, use_rep="X_pca", method=method,
                            knn=False, key_added="neighbors_" + method)
            scanpy_graphs[method] = (cpu_adata, False)
    start = tick("scanpy_graphs", start)

    candidate_cache = {}
    adjmat, affinity, G, weights, scores = generate_clustering_inputs2(
        X_gpu, metric, n_neighbors, weighted, seed, None, None,
        candidate_cache, timings)
    res["hubness_df"] = pd.DataFrame.from_dict(scores, orient="index", columns=["base"])
    start = tick("base_graph", start)

    def record_default(method_name, algorithm, labels):
        labels = _host_labels(labels).astype(int)
        res["clus"].setdefault(method_name, {})[algorithm] = labels
        res["clus_info"].setdefault(method_name, {})[algorithm] = {
            "n_clusters": len(np.unique(labels))}

    def record_target(method_name, algorithm, graph, graph_weights,
                      flavor="base", neighbors_key=None, cluster_adata=None,
                      scanpy_gpu=True):
        labels, info = getNclusters2(
            cluster_adata if cluster_adata is not None else adata,
            graph, n_clusters, seed, algorithm, flavor, graph_weights,
            neighbors_key=neighbors_key, scanpy_use_weights=weighted,
            scanpy_gpu=scanpy_gpu)
        res["clus"].setdefault(method_name, {})[algorithm] = _host_labels(labels).astype(int)
        res["clus_info"].setdefault(method_name, {})[algorithm] = info

    def graph_cluster(graph, graph_weights, resolution=1.0):
        part = leidenalg.find_partition(
            graph=graph, partition_type=leidenalg.RBConfigurationVertexPartition,
            weights=graph_weights, resolution_parameter=resolution, seed=seed)
        return np.asarray(part.membership)

    for method in ("umap", "gauss"):
        neighbors_key = "neighbors_" + method
        graph_adata, graph_gpu = scanpy_graphs[method]
        if graph_gpu:
            rsc.tl.leiden(graph_adata, resolution=1.0, rng=seed,
                          neighbors_key=neighbors_key, use_weights=weighted)
        else:
            sc.tl.leiden(graph_adata, resolution=1.0, random_state=seed,
                         neighbors_key=neighbors_key, use_weights=weighted)
        record_default("scanpy_default_" + method, "leiden",
                       graph_adata.obs["leiden"].astype(int).to_numpy())
        record_target("scanpy_" + method, "leiden", G, weights,
                      flavor="scanpy", neighbors_key=neighbors_key,
                      cluster_adata=graph_adata, scanpy_gpu=graph_gpu)
    record_default("base_default", "leiden", graph_cluster(G, weights))
    record_target("base", "leiden", G, weights)
    start = tick("default_and_target_clustering", start)

    for method_name, (hubness, hubness_params) in hubness_methods.items():
        for key in (method_name, method_name + "_default"):
            res["clus"][key] = {}
            res["clus_info"][key] = {}
        _, _, hub_G, hub_weights, hub_scores = generate_clustering_inputs2(
            X_gpu, metric, n_neighbors, weighted, seed, hubness,
            hubness_params, candidate_cache, timings)
        res["hubness_df"] = pd.concat(
            (res["hubness_df"], pd.DataFrame.from_dict(
                hub_scores, orient="index", columns=[method_name])), axis=1)
        start = tick(method_name + "_graph", start)
        record_default(method_name + "_default", "leiden",
                       graph_cluster(hub_G, hub_weights))
        record_target(method_name, "leiden", hub_G, hub_weights)
        start = tick(method_name + "_clustering", start)

    score_keys = list(method_keys)
    for method_name in hubness_methods:
        score_keys.extend((method_name + "_default", method_name))
    for key in score_keys:
        labels_by_algorithm = res["clus"][key]
        res["clus_scores"][key] = get_scores(
            true_labels, labels_by_algorithm, retained_cells_idx, X_gpu,
            metric, seed)
        res["clus_info"][key] = pd.DataFrame(res["clus_info"][key])
    tick("scoring", start)
    print("Stage seconds:", timings)
    print("Peak GPU pool bytes:", cp.get_default_memory_pool().total_bytes())
    return results

