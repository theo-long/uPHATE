from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
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
    "n_branch": 10,
    "n_dim": 50,
    "branch_length": 80,
}
PLOT_INDEX = 0
N_BOOTSTRAP = 10
FIGSIZE = (6, 5)

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
    fig, ax = plt.subplots(figsize=FIGSIZE)
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
    ax.set_aspect("equal")
    ax.legend(
        *scatter.legend_elements(),
        title="Branches",
        bbox_to_anchor=(1.05, 1),
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig("figures/base_phate_embedding.png", dpi=300)
    return fig, ax


def get_boostrap_embeddings(X, key, n_bootstrap):
    print("Generating bootstrapped PHATE embeddings...")
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


def bootstrap_phate_plot(aligned_embeddings, base_phate_embedding):
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    colors = ["C0", "C1", "C2", "C3"]
    for ax, emb, color in zip(axs.ravel(), aligned_embeddings, colors):
        ax.scatter(*emb.T, s=1, c=color)
        ax.scatter(*base_phate_embedding.T, c="grey", s=5, alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
    fig.supxlabel("PHATE 1")
    fig.supylabel("PHATE 2")
    fig.suptitle("Bootstrapped PHATE Embeddings vs. Base Embedding")
    fig.tight_layout()
    fig.savefig("figures/bootstrap_phate_embeddings.png", dpi=300)
    return fig, axs


def bootstrap_point_plot(aligned_embeddings, point_index):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for emb in aligned_embeddings:
        ax.scatter(*emb.T, c="gray", alpha=0.2, s=1)

    for emb in aligned_embeddings:
        ax.scatter(*emb[point_index].T, s=10, marker="x")

    ax.set_title("Bootstrapped PHATE Embeddings for Single Point")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("figures/bootstrap_phate_point_embeddings.png", dpi=300)
    return fig, ax


def gradient_magnitudes(X, key):
    X_uphate_jac = jax.jacrev(get_phate_embedding)(
        X,
        key,
        **PHATE_PARAMS,
    )
    grad_magnitudes = jnp.linalg.norm(X_uphate_jac, axis=(2, 3)) / jnp.sqrt(
        X.shape[0] * X_uphate_jac.shape[1]
    )
    return grad_magnitudes


def phate_gradients(X, key):
    print("Starting JAX jacobian computation...")
    grad_magnitudes = jax.jit(gradient_magnitudes)(
        X,
        key,
    )
    return grad_magnitudes


def create_gradient_sprite(size, cmap_name):
    """
    Creates a square RGBA image with a radial gradient circle in the middle.
    Pixels outside the circle are transparent.
    """
    # Create a grid of coordinates from -1 to 1
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Calculate radius
    R = np.sqrt(X**2 + Y**2)

    # Normalize radius for the colormap (0 at center, 1 at edge)
    # You can flip this (1 - R) if you want the "hottest" color in the center
    norm_R = 1 - R

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Apply colormap to the radius values
    # This gives us an (N, N, 4) array (RGBA)
    image = cmap(norm_R)

    # Set Alpha channel:
    # 1. Make pixels outside the circle (R > 1) completely transparent
    image[R > 1, 3] = 0.0

    # Fade the alpha towards the edge for a "glowing" effect
    image[:, :, 3] = np.maximum((1 - R).clip(0, 1) ** 0.5 * (R <= 1), 0.1)

    return image


def plot_ellipses_with_sprites(positions, axis_lengths):
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # 1. Generate the generic gradient image once
    sprite = create_gradient_sprite(size=512, cmap_name="Blues")

    # 2. Plot each ellipse as a stretched image
    for pos, axes in zip(positions, axis_lengths):
        cx, cy = pos
        width, height = axes

        # Calculate the bounding box (extent) for imshow
        # extent = [left, right, bottom, top]
        left = cx - width / 2
        right = cx + width / 2
        bottom = cy - height / 2
        top = cy + height / 2

        # Plot the image stretched to the ellipse bounds
        ax.imshow(
            sprite,
            extent=(left, right, bottom, top),
            aspect="auto",  # Allows stretching
            origin="lower",
            interpolation="bilinear",
        )  # Smooths the gradient

    # 3. Plot the black dots on top
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=1, zorder=10)
    return fig, ax


def phate_gradients_plot(grad_magnitudes, X_uphate):
    fig, ax = plot_ellipses_with_sprites(X_uphate, grad_magnitudes)
    ax.set_aspect("equal")
    ax.set_title("Position Uncertainty via PHATE Gradient Magnitudes")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
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
    embeddings = get_boostrap_embeddings(X, bootstrap_subkey, n_bootstrap=N_BOOTSTRAP)
    aligned_embeddings = align_bootstrap_embeddings(embeddings, X_uphate)
    bootstrap_phate_plot(aligned_embeddings, X_uphate)
    bootstrap_point_plot(aligned_embeddings, PLOT_INDEX)

    gradient_magnitudes = phate_gradients(X, base_subkey)
    phate_gradients_plot(gradient_magnitudes, X_uphate)


if __name__ == "__main__":
    main()
