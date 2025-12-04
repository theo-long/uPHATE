import jax
import time
from uphate.uphate import get_phate_embedding

device = jax.devices("gpu")[0]


def benchmark_jacobian(n_samples, n_features, n_landmark):
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (n_samples, n_features))

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
    start_time = time.time()
    print("Computing Jacobian...")
    J = jax.jacobian(embedding_fn)(X)
    # Block until ready
    J.block_until_ready()
    end_time = time.time()

    print(f"Peak GB: {device.memory_stats()['peak_bytes_in_use'] / 1e9: .2f} GB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Jacobian shape: {J.shape}")


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Benchmark Jacobian computation time.")
    parser.add_argument("--n_landmark", type=int, help="Number of samples.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        help="Logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    # Get the JAX XLA bridge logger
    logger = logging.getLogger("jax._src.xla_bridge")

    # Set the logging level to ERROR or CRITICAL to suppress warnings and info messages
    logger.setLevel(args.log_level)

    # Small scale test
    benchmark_jacobian(n_samples=50, n_features=5, n_landmark=args.n_landmark)
    # Medium scale test
    benchmark_jacobian(n_samples=100, n_features=10, n_landmark=args.n_landmark)
    # Large scale test
    benchmark_jacobian(n_samples=200, n_features=20, n_landmark=args.n_landmark)
