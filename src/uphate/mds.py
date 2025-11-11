import jax
import jax.numpy as jnp
import jaxopt
import pcax

from uphate.utils import pdist_squared


def mds_lr_schedule(t_max, eps=0.01):
    eta_max = 1
    eta_min = eps
    lambd = jnp.log(eta_max / eta_min) / (t_max - 1)
    etas = eta_max * jnp.exp(-lambd * jnp.arange(t_max))
    return etas


LR_SCHEDULE_LENGTH: int = 30
DEFAULT_LR_SCHEDULE: jax.Array = mds_lr_schedule(LR_SCHEDULE_LENGTH)


def compute_classic_mds_embedding(
    squared_dist_matrix: jax.Array,
    n_components: int,
):
    squared_dist_matrix -= squared_dist_matrix.mean(axis=0, keepdims=True)
    squared_dist_matrix -= squared_dist_matrix.mean(axis=1, keepdims=True)
    state = pcax.fit(squared_dist_matrix, n_components=n_components)
    return pcax.transform(state, squared_dist_matrix)


def mds_loss(X, Y):
    """Loss function for MDS, used in custom derivative for MDS solver."""
    triu_indices = jnp.triu_indices_from(X, k=1)
    return (
        (pdist_squared(X)[triu_indices] ** 0.5 - pdist_squared(Y)[triu_indices] ** 0.5)
        ** 2
    ).sum()


@jaxopt.implicit_diff.custom_root(jax.grad(mds_loss))
def compute_metric_mds_embedding(
    key: jax.Array,
    init_embedding: jax.Array,
    diff_potential: jax.Array,
):
    squared_dist_matrix = pdist_squared(diff_potential)
    dist_matrix = jnp.sqrt(squared_dist_matrix)
    triu_indices = jnp.stack(jnp.triu_indices_from(dist_matrix, k=1), axis=1)
    # Initialize with classic MDS
    X_transformed = init_embedding
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
