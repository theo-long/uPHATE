"""
Differentiable implementation of PHATE
"""

import jax
import jax.numpy as jnp
import pcax

from .jax_utils import (
    pdist_squared,
    compute_affinity_matrix,
    compute_von_neumann_entropy,
    find_knee_point,
)


def mds_lr_schedule(t_max, eps=0.01):
    eta_max = 1
    eta_min = eps
    lambd = jnp.log(eta_max / eta_min) / (t_max - 1)
    etas = eta_max * jnp.exp(-lambd * jnp.arange(t_max))
    return etas


LR_SCHEDULE_LENGTH: int = 30
DEFAULT_LR_SCHEDULE: jax.Array = mds_lr_schedule(LR_SCHEDULE_LENGTH)


def compute_optimal_t(diff_op: jax.Array):
    t, h = compute_von_neumann_entropy(diff_op)
    return find_knee_point(y=h, x=t)


def compute_classic_mds_embedding(
    squared_dist_matrix: jax.Array,
    n_components: int,
):
    squared_dist_matrix -= squared_dist_matrix.mean(axis=0, keepdims=True)
    squared_dist_matrix -= squared_dist_matrix.mean(axis=1, keepdims=True)
    state = pcax.fit(squared_dist_matrix, n_components=n_components)
    return pcax.transform(state, squared_dist_matrix)


def compute_metric_mds_embedding(
    key: jax.Array,
    diff_potential: jax.Array,
    n_components: int,
):
    squared_dist_matrix = pdist_squared(diff_potential)
    dist_matrix = jnp.sqrt(squared_dist_matrix)
    triu_indices = jnp.stack(jnp.triu_indices_from(dist_matrix, k=1), axis=1)
    # Initialize with classic MDS
    X_transformed = compute_classic_mds_embedding(
        squared_dist_matrix, n_components=n_components
    )
    iters_per_epoch = len(triu_indices)

    def pairwise_sgd_update(i, state):
        X_transformed, triu_indices, lr = state
        index = triu_indices[i]
        X_ij = X_transformed[index[0]] - X_transformed[index[1]]
        transformed_dist = jnp.linalg.norm(X_ij)
        r = (
            X_ij
            * (transformed_dist - dist_matrix[index[0], index[1]])
            / (2 * transformed_dist)
        )
        X_transformed = X_transformed.at[index].add(
            jnp.array([-lr * r, lr * r]),
        )
        return X_transformed, triu_indices, lr

    def sgd_epoch(i, state):
        key, X_transformed = state
        lr = DEFAULT_LR_SCHEDULE[i]
        key, subkey = jax.random.split(key)
        shuffled_triu_indices = jax.random.permutation(subkey, triu_indices)

        inner_state = (X_transformed, shuffled_triu_indices, lr)
        X_transformed, shuffled_triu_indices, lr = jax.lax.fori_loop(
            0, iters_per_epoch, pairwise_sgd_update, inner_state
        )
        return key, X_transformed

    key, X_transformed = jax.lax.fori_loop(
        0, LR_SCHEDULE_LENGTH, sgd_epoch, (key, X_transformed)
    )
    return X_transformed


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
    n_components: int = 2,
    knn: float = 5.0,
    decay: float = 40.0,
    n_landmark: int = 2000,
    t: float,
    gamma: float = 1.0,
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

        n_landmark : int, optional, default: 2000
            number of landmarks to use in fast PHATE

        t : int
            power to which the diffusion operator is powered.
            This sets the level of diffusion.

        gamma : float, optional, default: 1
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.
    """
    affinity_matrix = compute_affinity_matrix(
        X,
        n_landmark=n_landmark,
        knn=knn,
        decay=decay,
    )
    # Row-normalized diffusion probabilities
    diff_op = affinity_matrix / affinity_matrix.sum(axis=1)
    diff_potential = compute_diffusion_potential(diff_op, t, gamma)
    return compute_metric_mds_embedding(key, diff_potential, n_components=n_components)
