from contextlib import nullcontext
import jax
import time
import phate
import phate.tree
from uphate.uphate import get_phate_embedding

device = jax.devices("gpu")[0]


def get_data(n_samples, n_features, key):
    X, _ = phate.tree.gen_dla(
        n_dim=n_features,
        n_branch=5,
        branch_length=n_samples // 5,
    )
    return jax.numpy.array(X)


def benchmark_jacobian(n_samples, n_features, n_landmark, use_jacfwd, trace, save):
    if n_landmark is not None:
        n_landmark = None if n_landmark > n_samples else n_landmark
    key = jax.random.PRNGKey(0)
    X = get_data(n_samples, n_features, key)

    print(
        f"Benchmarking Jacobian for N={n_samples}, D={n_features}, L={n_landmark}, dtype={X.dtype}, device={device.platform.upper()}"
    )

    # Warmup
    print("Warmup...")
    _ = get_phate_embedding(X, key, t=2, n_components=2)

    # Define function to differentiate
    def embedding_fn(x):
        return get_phate_embedding(x, key, t=2, n_components=2, n_landmark=n_landmark)

    # Measure time
    ctx = jax.profiler.trace("./profiler_data") if trace else nullcontext()
    with ctx:
        start_time = time.time()
        print("Computing Jacobian...")
        if use_jacfwd:
            J = jax.jit(jax.jacfwd(embedding_fn))(X)
        else:
            J = jax.jit(jax.jacrev(embedding_fn))(X)
        # Block until ready
        J.block_until_ready()
        end_time = time.time()

    print(f"Peak GB: {device.memory_stats()['peak_bytes_in_use'] / 1e9: .2f} GB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Jacobian shape: {J.shape}")

    if save:
        suffix = f"_N{n_samples}_D{n_features}_L{n_landmark}_{'fwd' if use_jacfwd else 'rev'}.npy"
        print(f"Saving to {suffix}...")
        jax.numpy.save(f"J{suffix}", J)
        jax.numpy.save(f"X{suffix}", X)


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Benchmark Jacobian computation time.")
    parser.add_argument("--save", action="store_true", help="Save results to a file.")
    parser.add_argument("--n_landmark", type=int, help="Number of samples.")
    parser.add_argument(
        "--jacfwd", action="store_true", help="Use jacfwd instead of jacrev."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        help="Logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable JAX profiling trace output.",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=1000,
        help="Maximum number of samples to benchmark.",
    )
    args = parser.parse_args()

    # Get the JAX XLA bridge logger
    logger = logging.getLogger("jax._src.xla_bridge")

    # Set the logging level to ERROR or CRITICAL to suppress warnings and info messages
    logger.setLevel(args.log_level)

    # Small scale test
    n_samples = [100, 200, 500, 1000]
    features = [10, 20, 50, 100]
    for n_s, n_f in zip(n_samples, features):
        if n_s > args.max_n:
            break
        benchmark_jacobian(
            n_samples=n_s,
            n_features=n_f,
            n_landmark=args.n_landmark,
            use_jacfwd=args.jacfwd,
            trace=args.trace,
            save=args.save,
        )
