from pathlib import Path
import jax
import jax.numpy as jnp

from matplotlib.collections import EllipseCollection
import matplotlib.pyplot as plt
from uphate.uphate import (
    get_phate_embedding,
    get_phate_embedding_jit,
    get_phate_embedding_bootstrap,
)
from uphate.utils import align_embeddings
from phate.tree import gen_dla

PHATE_PARAMS = {
    "n_components": 2,
    "knn": 5.0,
    "t": 20,
}

DATA_PARAMS = {
    "n_branch": 5,
    "n_dim": 10,
    "branch_length": 10,
}

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)


def get_data():
    X, labels = gen_dla(**DATA_PARAMS)
    return jnp.array(X), labels


def get_base_phate(X, key):
    print("Generating base PHATE embedding...")
    X_uphate: jax.Array = get_phate_embedding_jit(
        X,
        key,
        **PHATE_PARAMS,
    )
    return X_uphate


def base_phate_plot(X_uphate, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        X_uphate[:, 0],
        X_uphate[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.8,
    )
    ax.set_title("Base PHATE Embedding")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    plt.colorbar(scatter, label="Branch Label")
    fig.tight_layout()
    fig.savefig("figures/base_phate_embedding.png", dpi=300)
    return fig, ax


def get_boostrap_embeddings(X, key, n_bootstrap=10):
    print("Generating bootstrap PHATE embeddings...")
    embeddings = get_phate_embedding_bootstrap(
        jnp.array(X),
        key,
        n_samples=n_bootstrap,
        **PHATE_PARAMS,
    )
    return embeddings


def align_bootstrap_embeddings(embeddings, base_phate_embedding):
    print("Aligning bootstrap embeddings...")
    aligned_embeddings = list(
        map(lambda e: align_embeddings(base_phate_embedding, e), embeddings)
    )
    return aligned_embeddings


def bootstrap_phate_plot(aligned_embeddings):
    fig, ax = plt.subplots(figsize=(6, 5))
    for emb in aligned_embeddings:
        ax.scatter(*emb.T, s=1)
    ax.set_title("Bootstrap PHATE Embeddings")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    fig.tight_layout()
    fig.savefig("figures/bootstrap_phate_embeddings.png", dpi=300)
    return fig, ax


def bootstrap_point_plot(aligned_embeddings, index=0):
    fig, ax = plt.subplots(figsize=(6, 5))
    for emb in aligned_embeddings:
        ax.scatter(*emb.T, c="gray", alpha=0.2, s=1)

    for emb in aligned_embeddings:
        ax.scatter(*emb[index].T, s=10, marker="x")

    ax.set_title(f"Bootstrapped PHATE Embeddings for {index}-th Point")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    fig.tight_layout()
    fig.savefig("figures/bootstrap_phate_point_embeddings.png", dpi=300)
    return fig, ax


def phate_gradients(X, key):
    print("Starting JAX jacobian computation...")
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
        key,
        **PHATE_PARAMS,
    )
    return X_uphate_jac


def phate_gradients_plot(X_uphate_jac, X_uphate):
    grad_magnitudes = jnp.linalg.norm(X_uphate_jac, axis=(2, 3))
    fig, ax = plt.subplots()
    ec = EllipseCollection(
        widths=grad_magnitudes[:, 0],
        heights=grad_magnitudes[:, 1],
        angles=jnp.zeros(grad_magnitudes.shape[0]),
        offsets=X_uphate,
        units="xy",
        transOffset=ax.transData,
    )

    ax.add_collection(ec)
    ax.set_xlim(X_uphate[:, 0].min() * 1.05, X_uphate[:, 0].max() * 1.05)
    ax.set_ylim(X_uphate[:, 1].min() * 1.05, X_uphate[:, 1].max() * 1.05)
    fig.tight_layout()
    fig.savefig("figures/phate_gradient_magnitudes.png", dpi=300)
    return fig, ax


def main():
    key = jax.random.PRNGKey(0)
    X, labels = get_data()

    key, base_subkey = jax.random.split(key)
    X_uphate = get_base_phate(X, base_subkey)
    base_phate_plot(X_uphate, labels)

    key, bootstrap_subkey = jax.random.split(key)
    embeddings = get_boostrap_embeddings(X, bootstrap_subkey, n_bootstrap=20)
    aligned_embeddings = align_bootstrap_embeddings(embeddings, X_uphate)
    bootstrap_phate_plot(aligned_embeddings)
    bootstrap_point_plot(aligned_embeddings, index=100)

    X_uphate_jac = phate_gradients(X, base_subkey)
    phate_gradients_plot(X_uphate_jac, X_uphate)


if __name__ == "__main__":
    main()
