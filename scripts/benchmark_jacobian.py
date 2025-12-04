
import jax
import time
from uphate.uphate import get_phate_embedding

def benchmark_jacobian(n_samples=100, n_features=10):
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (n_samples, n_features))
    
    print(f"Benchmarking Jacobian for N={n_samples}, D={n_features}")
    
    # Warmup
    print("Warmup...")
    _ = get_phate_embedding(X, key, t=2, n_components=2)
    
    # Define function to differentiate
    def embedding_fn(x):
        return get_phate_embedding(x, key, t=2, n_components=2)
    
    # Measure time
    start_time = time.time()
    print("Computing Jacobian...")
    J = jax.jacobian(embedding_fn)(X)
    # Block until ready
    J.block_until_ready()
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Jacobian shape: {J.shape}")

if __name__ == "__main__":
    # Small scale test
    benchmark_jacobian(n_samples=50, n_features=5)
    # Medium scale test
    benchmark_jacobian(n_samples=100, n_features=10)
    # Large scale test
    benchmark_jacobian(n_samples=500, n_features=20)