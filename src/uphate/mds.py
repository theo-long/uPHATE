from typing import Any
import jax
import jax.numpy as jnp
import jaxopt
import jaxopt.linear_solve
import cr.nimble.svd as lasvd

from uphate.utils import pdist_squared


def mds_lr_schedule(t_max, eps=0.01):
    eta_max = 1
    eta_min = eps
    lambd = jnp.log(eta_max / eta_min) / (t_max - 1)
    etas = eta_max * jnp.exp(-lambd * jnp.arange(t_max))
    return etas


LR_SCHEDULE_LENGTH: int = 30
DEFAULT_LR_SCHEDULE: jax.Array = mds_lr_schedule(LR_SCHEDULE_LENGTH)


def pca_decomposition(key, X, n_components):
    p0 = lasvd.lanbpro_random_start(key, X)
    U, S, V, bnd, n_converged, state = lasvd.lansvd_simple(X, n_components, p0)
    return V


def compute_classic_mds_embedding(
    key,
    squared_dist_matrix: jax.Array,
    n_components: int,
):
    squared_dist_matrix -= squared_dist_matrix.mean(axis=0, keepdims=True)
    squared_dist_matrix -= squared_dist_matrix.mean(axis=1, keepdims=True)
    return pca_decomposition(key, squared_dist_matrix, n_components)


def safe_pdist(x):
    """Compute pairwise distances with numerical stability."""
    eps = 1e-8
    return jnp.sqrt(jnp.triu(pdist_squared(x), k=1) + eps)


def mds_loss(embedding: jax.Array, data: jax.Array, key: Any):
    """
    Loss function for MDS, used in custom derivative for MDS solver.
        We pass the key to match the solver signature. Note that jaxopt expects
        the optimality function (in this case grad(loss) == 0) to have the parameters
        we are solving for be the *first* argument."""
    return ((safe_pdist(data) - safe_pdist(embedding)) ** 2).sum()


@jaxopt.implicit_diff.custom_root(
    jax.checkpoint(jax.grad(mds_loss)),  # pyright: ignore[reportPrivateImportUsage]
    solve=jaxopt.linear_solve.solve_normal_cg,
)
def compute_metric_mds_embedding(
    init_embedding: jax.Array,
    data: jax.Array,
    key: jax.Array,
):
    """Solve metric MDS using pairwise SGD on data indices.

    Args:
        init_embedding (jax.Array): Initialization for embedding
        data (jax.Array): High-dimensional data
        key (jax.Array): jax PRNG key used for SGD

    Returns:
        jax.Array: MDS solution
    """
    squared_dist_matrix = pdist_squared(data)
    dist_matrix = jnp.sqrt(squared_dist_matrix)
    triu_indices = jnp.stack(jnp.triu_indices_from(dist_matrix, k=1), axis=1)
    # Initialize with classic MDS
    X_transformed = init_embedding

    def pairwise_sgd_update(state, index):
        X_transformed, lr = state
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
        return (X_transformed, lr), None

    def sgd_epoch(state, lr):
        key, X_transformed = state
        key, subkey = jax.random.split(key)
        shuffled_triu_indices = jax.random.permutation(subkey, triu_indices)
        (X_transformed, lr), _ = jax.lax.scan(
            pairwise_sgd_update,
            init=(X_transformed, lr),
            xs=shuffled_triu_indices,
        )
        return (key, X_transformed), None

    (key, X_transformed), _ = jax.lax.scan(
        sgd_epoch,
        init=(key, X_transformed),
        xs=DEFAULT_LR_SCHEDULE,
    )
    return X_transformed
