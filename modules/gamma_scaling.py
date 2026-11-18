"""Adjustable-strength local-scaling dissimilarity for CuPy arrays."""

import cupy as cp


def gamma_scaled(distances, indices, gamma, local_scale_rank=10, epsilon=1e-12):
    """Return D_gamma for sorted, self-free candidate-neighbour arrays."""
    distances = cp.asarray(distances)
    indices = cp.asarray(indices)
    if not 1 <= local_scale_rank <= distances.shape[1]:
        raise ValueError("local_scale_rank exceeds the candidate-pool width")
    scales = cp.maximum(distances[:, local_scale_rank - 1], epsilon)
    denominator = cp.power(scales[:, None] * scales[indices], gamma)
    return cp.square(distances) / denominator
