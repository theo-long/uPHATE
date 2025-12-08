import argparse
from pathlib import Path
import datetime

import orbax.checkpoint as ocp
from flax import nnx
import jax.experimental.sparse as jaxsp
from uphate.nn import train_phate_surrogate, TransformerConfig
from uphate.utils import get_embryoid
from phate.tree import gen_dla
from phate import PHATE

DATA_PARAMS = {
    "n_branch": 10,
    "n_dim": 50,
    "branch_length": 80,
}
MODEL_SAVE_DIR = Path("./models").resolve()


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

    surrogate = train_phate_surrogate(
        X, X_phate, TransformerConfig(), epochs=args.epochs
    )
    _, state = nnx.split(surrogate)

    # Use *synchronous* checkpointer
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        MODEL_SAVE_DIR
        / f"surrogate_{args.dataset}_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}.nnx",
        state,
    )


if __name__ == "__main__":
    main()
