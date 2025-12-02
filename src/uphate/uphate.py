"""
Differentiable implementation of PHATE
"""

from typing import Optional

import jax
import jax.numpy as jnp
from tqdm import trange

from uphate.landmark import compute_landmark_op, extend_to_graph
from uphate.utils import pdist_squared, compute_von_neumann_entropy, find_knee_point
from uphate.mds import compute_metric_mds_embedding, compute_classic_mds_embedding


def compute_affinity_matrix(
    X: jax.Array,
    knn: float,
    decay: float,
    thresh=1e-4,
    affinity_weights: Optional[jax.Array] = None,
) -> jax.Array:
    # TODO handle landmarks

    # Note that rather than taking the square root to get distance, we just use squared dist everywhere
    # To account for this we multiply the alpha decay factor by 0.5
    decay *= 0.5
    pairwise_dist = pdist_squared(X)
    if affinity_weights is not None:
        pairwise_dist = (
            pairwise_dist * affinity_weights[:, None] * affinity_weights[None, :]
        )
    knn_low = jnp.floor(knn).astype(jnp.int32)
    knn_high = knn_low + 1
    frac = knn_high - knn_low

    # The bandwidth is given by the k-th nearest neighbor
    # If k is a float, we just interpolate between floor and ceil
    sorted_pairwise_dist = jnp.sort(pairwise_dist, axis=1)
    bandwith = sorted_pairwise_dist[:, knn_low] * frac + sorted_pairwise_dist[
        :, knn_high
    ] * (1 - frac)

    # Note that this is *not* symmetric
    locally_adaptive_pairwise_dist = jnp.power(pairwise_dist / bandwith[:, None], decay)

    affinity = jnp.exp(-1 * locally_adaptive_pairwise_dist)
    affinity = jnp.where(affinity > thresh, affinity, 0.0)

    # Symmetrize
    affinity = (affinity + affinity.T) / 2
    return affinity


def compute_optimal_t(diff_op: jax.Array):
    t, h = compute_von_neumann_entropy(diff_op)
    return find_knee_point(y=h, x=t)


def compute_diff_op(affinity_matrix):
    # Row-normalized diffusion probabilities
    diff_op = affinity_matrix / affinity_matrix.sum(axis=1, keepdims=True)
    return diff_op


def log_diffusion_potential(args):
    P, gamma = args
    eps = 1e-7
    return -1 * jnp.log(P + eps)


def powered_diffusion_potential(args):
    P, gamma = args
    c = (1 - gamma) / 2
    return (P**c) / c


def compute_diffusion_potential(
    diff_op: jax.Array,
    t: float,
    gamma: float,
):
    diff_op_t = jnp.linalg.matrix_power(diff_op, t)
    return jax.lax.cond(
        gamma == 1,
        log_diffusion_potential,
        powered_diffusion_potential,
        operand=(diff_op_t, gamma),
    )


def get_phate_embedding(
    X: jax.Array,
    key: jax.Array,
    *,
    t: float,
    n_components: int = 2,
    knn: float = 5.0,
    decay: float = 40.0,
    n_landmark: Optional[int] = None,
    gamma: float = 1.0,
    weights: Optional[jax.Array] = None,
    affinity_weights: Optional[jax.Array] = None,
):
    """Calculate the PHATE embedding of a dataset X.

    Potential of Heat-diffusion for Affinity-based Trajectory Embedding
    (PHATE) embeds high dimensional single-cell data into two or three
    dimensions for visualization of biological progressions as described
    in Moon et al, 2017 [1]_.

    Args:
        X: jax.Array
            data to generate low-dimensional embedding for

        key: jax.Array
            jax PRNG key.

        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        knn : int, optional, default: 5
            number of nearest neighbors on which to build kernel

        decay : int, optional, default: 40
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        n_landmark : int, optional, default: None
            number of landmarks to use in fast PHATE

        t : int
            power to which the diffusion operator is powered.
            This sets the level of diffusion.

        gamma : float, optional, default: 1
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.

        weights : node weights, used for bootstrap sampling.
    """
    affinity_matrix = compute_affinity_matrix(
        X,
        knn=knn,
        decay=decay,
        affinity_weights=affinity_weights,
    )
    if weights is not None:
        affinity_matrix = affinity_matrix * weights[None, :]

    if n_landmark is None:
        diff_op = compute_diff_op(affinity_matrix)
        diff_potential = compute_diffusion_potential(diff_op, t, gamma)
    else:
        key, subkey = jax.random.split(key)
        diff_op, data_to_landmarks = compute_landmark_op(
            subkey, affinity_matrix, n_landmark
        )
        del subkey
        diff_potential = compute_diffusion_potential(diff_op, t, gamma)
        diff_potential = extend_to_graph(data_to_landmarks, diff_potential)

    init_embedding = compute_classic_mds_embedding(
        pdist_squared(diff_potential), n_components=n_components
    )
    return compute_metric_mds_embedding(init_embedding, diff_potential, key)


get_phate_embedding_jit = jax.jit(
    get_phate_embedding,
    static_argnames=[
        "t",
        "n_components",
        "knn",
        "decay",
        "n_landmark",
        "gamma",
    ],
)


def get_phate_embedding_bootstrap(
    X: jax.Array,
    key: jax.Array,
    *,
    n_samples: int,
    dirichlet_alpha: float = 1.0,
    t: float,
    n_components: int = 2,
    knn: float = 5.0,
    decay: float = 40.0,
    n_landmark: Optional[int] = None,
    gamma: float = 1.0,
    affinity_weighting: bool = False,
):
    """Generate bayesian bootstrap samples of the phate embedding

    Args:
        X: jax.Array
            data to generate low-dimensional embedding for

        key: jax.Array
            jax PRNG key.

        n_samples: int
            number of bootstrap samples

        dirichlet_alpha: float, default: 1
            alpha parameter to use for bootstrap dirichlet distribution

        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        knn : int, optional, default: 5
            number of nearest neighbors on which to build kernel

        decay : int, optional, default: 40
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        n_landmark : int, optional, default: None
            number of landmarks to use in fast PHATE

        t : int
            power to which the diffusion operator is powered.
            This sets the level of diffusion.

        gamma : float, optional, default: 1
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.

        weights : node weights, used for bootstrap sampling.

    Returns:
        jax.Array : bootstrapped embedding samples
    """
    X = jnp.array(X)
    embeddings = []
    key, subkey = jax.random.split(key)
    weights = jax.random.dirichlet(
        subkey, jnp.ones(X.shape[0]) * dirichlet_alpha, shape=(n_samples,)
    )
    del subkey
    for i in trange(n_samples):
        affinity_weight_vector = None
        weight_vector = None
        if affinity_weighting:
            affinity_weight_vector = weights[i]
            affinity_weight_vector *= len(affinity_weight_vector)
        else:
            weight_vector = weights[i]
        key, subkey = jax.random.split(key)
        emb = get_phate_embedding_jit(
            X,
            subkey,
            n_components=n_components,
            knn=knn,
            t=t,
            decay=decay,
            n_landmark=n_landmark,
            gamma=gamma,
            weights=weight_vector,
            affinity_weights=affinity_weight_vector,
        )
        del subkey
        embeddings.append(emb)

    return jnp.stack(embeddings)
