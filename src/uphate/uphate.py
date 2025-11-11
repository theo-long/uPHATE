"""
Differentiable implementation of PHATE
"""

from typing import Optional

import jax
import jax.numpy as jnp
import pcax


def mds_lr_schedule(t_max, eps=0.01):
    eta_max = 1
    eta_min = eps
    lambd = jnp.log(eta_max / eta_min) / (t_max - 1)
    etas = eta_max * jnp.exp(-lambd * jnp.arange(t_max))
    return etas


LR_SCHEDULE_LENGTH: int = 30
DEFAULT_LR_SCHEDULE: jax.Array = mds_lr_schedule(LR_SCHEDULE_LENGTH)


def pdist_squared(x):
    return jnp.sum((x[:, None] - x[None, :]) ** 2, axis=-1)


def compute_affinity_matrix(
    X: jax.Array,
    n_landmark: Optional[int],
    knn: float,
    decay: float,
    thresh=1e-4,
) -> jax.Array:
    # TODO handle landmarks

    # Note that rather than taking the square root to get distance, we just use squared dist everywhere
    # To account for this we multiply the alpha decay factor by 0.5
    decay *= 0.5
    pairwise_dist = pdist_squared(X)
    knn_low = jnp.floor(knn)
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


def compute_von_neumann_entropy(data, t_max=100):
    """
    Determines the Von Neumann entropy of data
    at varying matrix powers. The user should select a value of t
    around the "knee" of the entropy curve.

    Parameters
    ----------
    t_max : int, default: 100
        Maximum value of t to test

    Returns
    -------
    entropy : array, shape=[t_max]
        The entropy of the diffusion affinities for each value of t

    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> X = jjnp.eye(10)
    >>> X[0,0] = 5
    >>> X[3,2] = 4
    >>> h = phate.vne.compute_von_neumann_entropy(X)
    >>> phate.vne.find_knee_point(h)
    23

    """
    _, eigenvalues, _ = jnp.linalg.svd(data)
    entropy = []
    eigenvalues_t = jnp.copy(eigenvalues)
    for _ in range(t_max):
        prob = eigenvalues_t / jnp.sum(eigenvalues_t)
        prob = prob + jnp.finfo(float).eps
        entropy.append(-jnp.sum(prob * jnp.log(prob)))
        eigenvalues_t = eigenvalues_t * eigenvalues
    entropy = jnp.array(entropy)

    return jnp.array(entropy)


def find_knee_point(y, x=None):
    """
    Returns the x-location of a (single) knee of curve y=f(x)

    Parameters
    ----------

    y : array, shape=[n]
        data for which to find the knee point

    x : array, optional, shape=[n], default=jnp.arange(len(y))
        indices of the data points of y,
        if these are not in order and evenly spaced

    Returns
    -------
    knee_point : int
    The index (or x value) of the knee point on y

    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> x = jnp.arange(20)
    >>> y = jnp.exp(-x/10)
    >>> phate.vne.find_knee_point(y,x)
    8

    """
    try:
        y.shape
    except AttributeError:
        y = jnp.array(y)

    if len(y) < 3:
        raise ValueError("Cannot find knee point on vector of length 3")
    elif len(y.shape) > 1:
        raise ValueError("y must be 1-dimensional")

    if x is None:
        x = jnp.arange(len(y))
    else:
        try:
            x.shape
        except AttributeError:
            x = jnp.array(x)
        if not x.shape == y.shape:
            raise ValueError("x and y must be the same shape")
        else:
            # ensure x is sorted float
            idx = jnp.argsort(x)
            x = x[idx]
            y = y[idx]

    n = jnp.arange(2, len(y) + 1).astype(jnp.float32)
    # figure out the m and b (in the y=mx+b sense) for the "left-of-knee"
    sigma_xy = jnp.cumsum(x * y)[1:]
    sigma_x = jnp.cumsum(x)[1:]
    sigma_y = jnp.cumsum(y)[1:]
    sigma_xx = jnp.cumsum(x * x)[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mfwd = (n * sigma_xy - sigma_x * sigma_y) / det
    bfwd = -(sigma_x * sigma_xy - sigma_xx * sigma_y) / det

    # figure out the m and b (in the y=mx+b sense) for the "right-of-knee"
    sigma_xy = jnp.cumsum(x[::-1] * y[::-1])[1:]
    sigma_x = jnp.cumsum(x[::-1])[1:]
    sigma_y = jnp.cumsum(y[::-1])[1:]
    sigma_xx = jnp.cumsum(x[::-1] * x[::-1])[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mbck = ((n * sigma_xy - sigma_x * sigma_y) / det)[::-1]
    bbck = (-(sigma_x * sigma_xy - sigma_xx * sigma_y) / det)[::-1]

    # figure out the sum of per-point errors for left- and right- of-knee fits
    error_curve = jnp.full_like(y, jnp.nan)
    for breakpt in jnp.arange(1, len(y) - 1):
        delsfwd = (mfwd[breakpt - 1] * x[: breakpt + 1] + bfwd[breakpt - 1]) - y[
            : breakpt + 1
        ]
        delsbck = (mbck[breakpt - 1] * x[breakpt:] + bbck[breakpt - 1]) - y[breakpt:]

        error_curve[breakpt] = jnp.sum(jnp.abs(delsfwd)) + jnp.sum(jnp.abs(delsbck))

    # find location of the min of the error curve
    loc = jnp.argmin(error_curve[1:-1]) + 1
    knee_point = x[loc]
    return knee_point


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
    diff_op = compute_diff_op(affinity_matrix)
    diff_potential = compute_diffusion_potential(diff_op, t, gamma)
    return compute_metric_mds_embedding(key, diff_potential, n_components=n_components)
