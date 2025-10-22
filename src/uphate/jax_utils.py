import jax
import jax.numpy as jnp


@jax.jit
def squared_dist(x, y):
    return jnp.sum(x[:, None] - y[None, :]) ** 2


@jax.jit
def affinity_matrix(
    X: jnp.ndarray,
    n_landmark: int,
    knn: float,
    decay: float,
    thresh=1e-4,
):
    # TODO handle landmarks

    # Note that rather than taking the square root to get distance, we just use squared dist everywhere
    # To account for this we multiply the alpha decay factor by 0.5
    decay *= 0.5
    pairwise_dist = squared_dist(X, X)
    knn_low, knn_high = jnp.floor(knn), jnp.ceil(knn)
    frac = knn_high - knn_low

    # The bandwidth is given by the k-th nearest neighbor
    # If k is a float, we just interpolate between floor and ceil
    bandwith = (
        jnp.sort(pairwise_dist, axis=1)[:, knn_low:knn_high]
        * jnp.array([frac, 1 - frac])
    ).sum(axis=1)

    affinity = 0.5 * jnp.power(
        pairwise_dist / bandwith[None, :], decay
    ) + 0.5 * jnp.power(pairwise_dist / bandwith[:, None], decay)

    affinity = jnp.where(affinity > thresh, affinity, 0.0)
    return affinity
