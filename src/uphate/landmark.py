import jax
import jax.numpy as jnp

from cr.sparse.cluster.spectral import normalized_symmetric_fast_k


def compute_landmark_op(key: jax.Array, affinity_matrix: jax.Array, n_landmark: int):
    """Compute the landmark operator
    Calculates spectral clusters on the kernel, and calculates transition
    probabilities between cluster centers by using transition probabilities
    between samples assigned to each cluster.
    """
    key, subkey = jax.random.split(key)
    clusters = normalized_symmetric_fast_k(subkey, affinity_matrix, n_landmark)
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
