import jax
import jax.numpy as jnp

from cr.sparse.cluster.spectral import (
    SpectralclusteringSolution,
    normalized_symmetric_w,
)
from cr.sparse._src.cluster.kmeans import kmeans
import cr.nimble.svd as lasvd
import cr.nimble as cnb


def normalized_symmetric_fast_k(key, W, k):
    """Normalized symmetric spectral clustering fast implementation"""
    # following is a shortcut to compute D^{-1} W
    W = normalized_symmetric_w(W)
    # convert it into a sparse matrix
    # W = BCOO.fromdense(W)
    p0 = lasvd.lanbpro_random_start(key, W)
    U, S, V, bnd, n_converged, state = lasvd.lansvd_simple(W, k, p0)
    # Choose the last k eigen vectors
    kernel = V[:, :k]
    # normalize the rows of kernel
    kernel = cnb.normalize_l2_rw(kernel)
    result = kmeans(key, kernel, k, iter=100)
    return SpectralclusteringSolution(
        singular_values=S,
        assignment=result.assignment,
        # technically we didn't compute the Laplacian correctly
        laplancian=W,
        num_clusters=k,
        # we didn't compute the connectivity
        connectivity=-1,
    )


def compute_landmark_op(key: jax.Array, affinity_matrix: jax.Array, n_landmark: int):
    """Compute the landmark operator
    Calculates spectral clusters on the kernel, and calculates transition
    probabilities between cluster centers by using transition probabilities
    between samples assigned to each cluster.
    """
    key, subkey = jax.random.split(key)

    # Don't take gradients of cluster positions
    clusters = normalized_symmetric_fast_k(
        subkey, jax.lax.stop_gradient(affinity_matrix), n_landmark
    )
    del subkey

    # transition matrices
    # P(node i -> cluster k) = sum(A_ij for node j in cluster k)
    pmn = jax.ops.segment_sum(affinity_matrix, clusters.assignment, n_landmark)

    # row normalize
    pnm = pmn.transpose()
    pmn /= pmn.sum(axis=1, keepdims=True)
    pnm /= pnm.sum(axis=1, keepdims=True)

    landmark_op = jnp.matmul(pmn, pnm)
    return landmark_op, pnm


def extend_to_graph(data_to_landmarks: jax.Array, diff_potential: jax.Array):
    """Extend the diffusion potential on the landmarks to the full graph.

    This is done by left-multiplication by the (N, K) matrix of node-landmark affinities

    Args:
        data_to_landmarks (jax.Array): (N, K) matrix of node-landmark affinities
        diff_potential (jax.Array): diff potential of landmark graph

    Returns:
        jax.Array: full diffusion potential
    """
    diff_potential = jnp.matmul(data_to_landmarks, diff_potential)
    return diff_potential
