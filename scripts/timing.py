import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from uphate.uphate import (
    get_phate_embedding,
    get_phate_embedding_jit,
    get_phate_embedding_bootstrap,
)
from uphate.utils import align_embeddings
from phate.tree import gen_dla

# Using smaller dataset for now since full jac is expensive
X, labels = gen_dla(n_branch=10, n_dim=20, branch_length=50)

X_uphate: jax.Array = get_phate_embedding_jit(
    X,
    jax.random.key(20),
    t=5,
)
print("X_uphate.device:", X_uphate.device)

print("Starting JAX jacobian computation...")
start = time.time()
X_uphate_jac = jax.jit(
    jax.jacobian(get_phate_embedding),
    static_argnames=[
        "t",
        "n_components",
        "knn",
        "decay",
        "n_landmark",
        "gamma",
        "threshold",
    ],
)(
    X,
    jax.random.key(20),
    t=5,
)
end = time.time()
X_uphate_jac.block_until_ready()
print(f"JAX jacobian with compilation time: {end - start:.2f} seconds")

start = time.time()
X_uphate_jac = jax.jit(
    jax.jacobian(get_phate_embedding),
    static_argnames=[
        "t",
        "n_components",
        "knn",
        "decay",
        "n_landmark",
        "gamma",
        "threshold",
    ],
)(
    X,
    jax.random.key(20),
    t=5,
)
X_uphate_jac.block_until_ready()
end = time.time()
print(f"JAX jacobian computation time: {end - start:.2f} seconds")