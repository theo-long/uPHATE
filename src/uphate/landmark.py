import jax
import jax.numpy as jnp

from cr.sparse.cluster.spectral import normalized_symmetric_fast_k


def build_landmark_op(key: jax.Array, affinity_matrix: jax.Array, n_landmark: int):
    """Build the landmark operator
    Calculates spectral clusters on the kernel, and calculates transition
    probabilities between cluster centers by using transition probabilities
    between samples assigned to each cluster.
    """
    key, subkey = jax.random.split(key)
    clusters = normalized_symmetric_fast_k(subkey, affinity_matrix, n_landmark)
    del subkey

    # transition matrices
    pmn = jnp.array(
        [jnp.sum(affinity_matrix[clusters == i, :], axis=0) for i in range(n_landmark)]
    )

    # row normalize
    pnm = pmn.transpose()
    pmn /= pmn.sum(axis=1, keepdims=True)
    pnm /= pnm.sum(axis=1, keepdims=True)

    # sparsity agnostic matrix multiplication
    landmark_op = pmn.dot(pnm)
    return landmark_op
