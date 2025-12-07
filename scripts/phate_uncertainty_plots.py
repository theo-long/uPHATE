import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jax._src.api import _std_basis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
from zadu import zadu

from uphate.uphate import (
    get_phate_embedding,
    get_phate_embedding_jit,
    get_phate_embedding_bootstrap,
)
from uphate.utils import align_embeddings
from phate.tree import gen_dla

DATA_PARAMS = {
    "n_branch": 10,
    "n_dim": 50,
    "branch_length": 80,
}
FIGSIZE = (6, 5)

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PHATE uncertainty plots.")
    parser.add_argument(
        "--n_bootstrap", type=int, default=10, help="Number of bootstrap samples."
    )
    parser.add_argument(
        "--plot_index",
        type=int,
        default=0,
        help="Index of point to plot in bootstrap point plot.",
    )
    parser.add_argument(
        "--knn", type=float, default=5.0, help="Number of nearest neighbors for PHATE."
    )
    parser.add_argument("--t", type=int, default=20, help="Diffusion time for PHATE.")
    parser.add_argument(
        "--dataset", type=str, choices=["dla", "embryoid", "embryoid_pca"], default="dla", help="Dataset to use."
    )
    parser.add_argument(
        "--save", action="store_true", help="Whether to save computed embeddings."
    )
    parser.add_argument("--batch_size", type=int, default=400)
    args = parser.parse_args()
    return args


def get_embryoid(pca=False):
    try:
        if pca:
            X = jnp.load("./data/embryoid_body_preprocessed_pca.npy")
        else:
            X = jnp.load("./data/embryoid_body_preprocessed.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Could not find embryoid data, try running examples/embryoid_body_data.ipynb first."
        ) from e

    labels = jnp.load("./data/embryoid_body_timepoint.npy")
    return X, labels


def get_data():
    X, labels = gen_dla(**DATA_PARAMS)
    return jnp.array(X), labels


def get_base_phate(X, key, phate_params):
    print("Generating base PHATE embedding...")
    X_uphate: jax.Array = get_phate_embedding_jit(
        X,
        key,
        **phate_params,
    )
    return X_uphate


def base_phate_plot(X_uphate, labels):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(
        x=X_uphate[:, 0],
        y=X_uphate[:, 1],
        hue=labels,
        style=labels,
        palette="tab10",
        s=20,
        alpha=0.8,
        legend=True,
        ax=ax,
    )
    ax.set_title("Base PHATE Embedding")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.set_aspect("equal")
    ax.legend(
        title="Branches",
        bbox_to_anchor=(1.05, 1),
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "base_phate_embedding.png", dpi=300)
    return fig, ax


def get_boostrap_embeddings(X, key, n_bootstrap, phate_params):
    print("Generating bootstrapped PHATE embeddings...")
    embeddings = get_phate_embedding_bootstrap(
        jnp.array(X),
        key,
        n_samples=n_bootstrap,
        **phate_params,
    )
    return embeddings


def align_bootstrap_embeddings(embeddings, base_phate_embedding):
    print("Aligning bootstrap embeddings...")
    aligned_embeddings = list(
        map(lambda e: align_embeddings(base_phate_embedding, e), embeddings)
    )
    return jnp.array(aligned_embeddings)


def bootstrap_phate_plot(aligned_embeddings, base_phate_embedding):
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    colors = ["C0", "C1", "C2", "C3"]
    for ax, emb, color in zip(axs.ravel(), aligned_embeddings, colors):
        ax.scatter(*emb.T, s=1, c=color)
        ax.scatter(*base_phate_embedding.T, c="grey", s=5, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
    fig.supxlabel("PHATE 1")
    fig.supylabel("PHATE 2")
    fig.suptitle("Bootstrapped PHATE Embeddings vs. Base Embedding")
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_phate_embeddings.png", dpi=300)
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
    fig.savefig(fig_dir / "bootstrap_phate_point_embeddings.png", dpi=300)
    return fig, ax


def phate_gradients(X, key, phate_params, batch_size):
    print("Starting JAX jacobian computation...")

    def embedding_fun(X):
        return get_phate_embedding(X, key, **phate_params)

    def gradient_magnitudes(X):
        Y, vjp_fun = jax.vjp(embedding_fun, X)
        basis = _std_basis(Y)
        X_uphate_jac = jax.lax.map(vjp_fun, basis, batch_size=batch_size)[0].reshape(
            *Y.shape, *X.shape
        )
        grad_magnitudes = jnp.linalg.norm(X_uphate_jac, axis=(2, 3)) / jnp.sqrt(
            X.shape[0] * X_uphate_jac.shape[1]
        )
        return grad_magnitudes

    compiled_f = jax.jit(gradient_magnitudes).trace(X).lower().compile()
    print("Finished compiling grad function, executing...")
    grad_magnitudes = compiled_f(X)
    return grad_magnitudes.block_until_ready()


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
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 1. Generate the generic gradient image once
    sprite = create_gradient_sprite(size=128, cmap_name="Blues")

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
    print("Plotting gradient magnitudes")
    fig, ax = plot_ellipses_with_sprites(
        X_uphate, jnp.clip(grad_magnitudes * 20, 0.0, 10.0)
    )
    ax.set_aspect("equal")
    ax.set_title("Position Uncertainty via PHATE Gradient Ellipses")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    fig.tight_layout()
    print("Finished generating plot, saving...")
    fig.savefig(fig_dir / "phate_gradient_ellipses.png", dpi=300)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    grad_ellipse_area = jnp.clip(jnp.log(jnp.prod(grad_magnitudes, axis=1)), None, 1)
    ax.scatter(
        *X_uphate.T,
        c=grad_ellipse_area,
        s=(grad_ellipse_area - grad_ellipse_area.min() + 0.2) ** 2,
    )
    ax.set_aspect("equal")
    ax.set_title("Position Uncertainty via PHATE Gradient Magnitudes")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    fig.savefig(fig_dir / "phate_gradient_magnitudes.png", dpi=300)
    return


def phate_uncertainty_comparison(X_uphate, gradient_magnitudes, bootstrap_embeddings):
    print("Plotting uncertainty comparison...")
    coord_std = jnp.std(bootstrap_embeddings, axis=0)
    single_std = jnp.linalg.norm(coord_std, axis=1)
    single_grad = jnp.linalg.norm(gradient_magnitudes, axis=1)
    corr = scipy.stats.spearmanr(single_std, single_grad)
    print(f"Gradient vs. Bootstrap Spearman Correlation: {corr:2f}")

    def size_transform(x):
        return (x + 1) ** 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for i, (name, uncertainty_metric) in enumerate(
        [
            ("Bootstrap Std. Dev.", single_std),
            ("Gradient Magnitude", single_grad),
        ]
    ):
        rank = jnp.argsort(jnp.argsort(uncertainty_metric))
        sns.scatterplot(
            x=X_uphate[:, 0],
            y=X_uphate[:, 1],
            c=rank,
            ax=axes[i],
            s=size_transform(rank),
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("PHATE 1")
        axes[i].set_ylabel("PHATE 2")

    fig.savefig("figures/uncertainty_comparison.png", dpi=300)


def generate_correlation_plots(X, labels, X_uphate, X_bootstrap, gradient_magnitudes):
    coord_std = jnp.std(X_bootstrap, axis=0)
    single_std = jnp.linalg.norm(coord_std, axis=1)
    single_grad = jnp.linalg.norm(gradient_magnitudes, axis=1)
    spec = [
        {
            "id": "tnc",
            "params": {"k": 20},
        },
        {"id": "mrre", "params": {"k": 20}},
        {
            "id": "ca_tnc",
            "params": {"k": 20},
        },
    ]
    global_scores, (local_tnc, local_mrre, local_ca_tnc) = zadu.ZADU(
        spec, X, return_local=True
    ).measure(X_uphate, label=labels)

    data = dict(**local_tnc, **local_ca_tnc, **local_mrre)
    data["Gradient Magnitudes"] = single_grad
    data["Bootstrap Std. Dev."] = single_std

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(
        pd.DataFrame(data)
        .corr("spearman")
        .loc[["Gradient Magnitudes", "Bootstrap Std. Dev."], :],
        ax=ax,
        annot=True,
    )
    ax.set_xticklabels(data.keys(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig("figures/metric_correlations.png", dpi=300)


def main():
    args = parse_args()
    phate_params = {
        "knn": args.knn,
        "t": args.t,
    }
    key = jax.random.PRNGKey(0)
    if args.dataset == "dla":
        X, labels = get_data()
    elif args.dataset == "embryoid":
        X, labels = get_embryoid()
    elif args.dataset == "embryoid_pca":
        X, labels = get_embryoid(pca=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    key, base_subkey = jax.random.split(key)
    X_uphate = get_base_phate(X, base_subkey, phate_params)
    base_phate_plot(X_uphate, labels)

    key, bootstrap_subkey = jax.random.split(key)
    embeddings = get_boostrap_embeddings(
        X, bootstrap_subkey, n_bootstrap=args.n_bootstrap, phate_params=phate_params
    )
    aligned_embeddings = align_bootstrap_embeddings(embeddings, X_uphate)
    bootstrap_phate_plot(aligned_embeddings, X_uphate)
    bootstrap_point_plot(aligned_embeddings, args.plot_index)

    gradient_magnitudes = phate_gradients(X, base_subkey, phate_params, args.batch_size)
    phate_gradients_plot(gradient_magnitudes, X_uphate)

    max_mem_gb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    print(f"Completed with {max_mem_gb:.4f} max GPU memory usage")

    phate_uncertainty_comparison(X_uphate, gradient_magnitudes, aligned_embeddings)

    if args.save:
        jnp.save("X.npy", X)
        jnp.save("X_uphate.npy", X_uphate)
        jnp.save("boostrap_embeddings.npy", aligned_embeddings)
        jnp.save("gradient_magnitudes.npy", gradient_magnitudes)


if __name__ == "__main__":
    main()
