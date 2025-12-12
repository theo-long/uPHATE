import argparse
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from jax._src.api import _std_basis
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
from uphate.utils import align_embeddings, get_embryoid
from uphate.plotting import plot_ellipses_with_sprites
from phate.tree import gen_dla

DATA_PARAMS = {
    "n_branch": 10,
    "n_dim": 50,
    "branch_length": 80,
}
FIGSIZE = (6, 5)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PHATE uncertainty plots.")
    parser.add_argument(
        "--n_bootstrap", type=int, default=10, help="Number of bootstrap samples."
    )
    parser.add_argument("--n_landmark", type=int)
    parser.add_argument(
        "--plot_index",
        type=int,
        default=0,
        help="Index of point to plot in bootstrap point plot.",
    )
    parser.add_argument("--decay", default=20, type=float)
    parser.add_argument(
        "--knn", type=float, default=5.0, help="Number of nearest neighbors for PHATE."
    )
    parser.add_argument("--t", type=int, default=20, help="Diffusion time for PHATE.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dla", "embryoid", "embryoid_pca"],
        default="dla",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Whether to save computed embeddings."
    )
    parser.add_argument("--batch_size", type=int, default=400)
    args = parser.parse_args()
    return args


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


def phate_gradients_plot(grad_magnitudes, X_uphate):
    print("Plotting gradient magnitudes")
    fig, ax = plot_ellipses_with_sprites(
        X_uphate, jnp.clip(grad_magnitudes * 20, 0.0, 10.0), FIGSIZE
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


def main(args):
    phate_params = {
        "knn": args.knn,
        "t": args.t,
        "n_landmark": args.n_landmark,
        "decay": args.decay,
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
    s = time.time()
    X_uphate = get_base_phate(X, base_subkey, phate_params)
    print(f"TIME: {time.time() - s:.1f}")
    base_phate_plot(X_uphate, labels)

    key, bootstrap_subkey = jax.random.split(key)
    embeddings = get_boostrap_embeddings(
        X, bootstrap_subkey, n_bootstrap=args.n_bootstrap, phate_params=phate_params
    )
    aligned_embeddings = align_bootstrap_embeddings(embeddings, X_uphate)
    bootstrap_phate_plot(aligned_embeddings, X_uphate)
    bootstrap_point_plot(aligned_embeddings, args.plot_index)

    gradient_magnitudes = phate_gradients(X, base_subkey, phate_params, args.batch_size)

    if args.save:
        jnp.save("X.npy", X)
        jnp.save("X_uphate.npy", X_uphate)
        jnp.save("boostrap_embeddings.npy", aligned_embeddings)
        jnp.save("gradient_magnitudes.npy", gradient_magnitudes)

    phate_gradients_plot(gradient_magnitudes, X_uphate)

    max_mem_gb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    print(f"Completed with {max_mem_gb:.4f} max GPU memory usage")

    phate_uncertainty_comparison(X_uphate, gradient_magnitudes, aligned_embeddings)


if __name__ == "__main__":
    args = parse_args()
    fig_dir = Path("figures") / args.dataset
    fig_dir.mkdir(exist_ok=True)
    main(args)
