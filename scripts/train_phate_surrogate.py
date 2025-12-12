import argparse
from pathlib import Path
import datetime

import jax
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx
from uphate.nn import train_phate_surrogate, TransformerConfig
from uphate.utils import get_embryoid, standardize
from phate.tree import gen_dla
from phate import PHATE
import jax.numpy as jnp

DATA_PARAMS = {
    "n_branch": 10,
    "n_dim": 50,
    "branch_length": 80,
}
MODEL_SAVE_DIR = Path("./models").resolve()
X_SAVE_DIR = Path("./embeddings").resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PHATE uncertainty plots.")
    parser.add_argument("--n_landmark", type=int, default=2000)
    parser.add_argument("--decay", default=20, type=float)
    parser.add_argument(
        "--knn", type=int, default=5, help="Number of nearest neighbors for PHATE."
    )
    parser.add_argument("--t", type=int, default=20, help="Diffusion time for PHATE.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dla", "embryoid", "embryoid_pca"],
        default="dla",
        help="Dataset to use.",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.dataset == "dla":
        X, labels = gen_dla(**DATA_PARAMS)
    elif args.dataset == "embryoid":
        X, labels = get_embryoid()
    elif args.dataset == "embryoid_pca":
        X, labels = get_embryoid(pca=True)
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset}")

    X_phate = PHATE(
        knn=args.knn, decay=args.decay, n_landmark=args.n_landmark, t=args.t
    ).fit_transform(X)

    # standardize
    X, X_phate = standardize(jnp.array(X)), standardize(jnp.array(X_phate))

    surrogate = train_phate_surrogate(
        X,
        X_phate,
        TransformerConfig(),
        epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    _, state = nnx.split(surrogate)

    # Use *synchronous* checkpointer
    now_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        MODEL_SAVE_DIR / f"surrogate_{args.dataset}_{now_str}.nnx",
        state,
    )

    X_surrogate = surrogate(X)
    save_dir = X_SAVE_DIR / f"{args.dataset}_{now_str}"
    save_dir.mkdir(exist_ok=True)
    jnp.save(
        save_dir / "X_surrogate.npy",
        X_surrogate,
    )
    jnp.save(
        save_dir / "X.npy",
        X,
    )
    jnp.save(
        save_dir / "X_phate.npy",
        X_phate,
    )
    jnp.save(
        save_dir / "labels.npy",
        labels,
    )

    jac_surrogate = jax.jit(jax.jacrev(surrogate))(X)
    jnp.save(
        save_dir / "jac_surrogate.npy",
        jac_surrogate,
    )

    # Save comparison plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax[0].scatter(*X_phate.T, c=labels, cmap="tab10", s=1)
    ax[0].set_title("PHATE")
    ax[1].scatter(*X_surrogate.T, c=labels, cmap="tab10", s=1)
    ax[1].set_title("Surrogate")
    fig.savefig(
        save_dir / "surrogate_comparison.png",
    )

if __name__ == "__main__":
    main()
