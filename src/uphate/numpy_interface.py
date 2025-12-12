import phate
import numpy as np
import scipy.sparse
from sklearn.exceptions import NotFittedError
import tasklogger

_logger = tasklogger.get_tasklogger("graphtools")


class BootstrappedPHATE(phate.PHATE):
    def __init__(
        self, *, n_boostrap_samples: int, dirichlet_alpha: float = 1.0, **phate_kwargs
    ):
        super().__init__(**phate_kwargs)
        self.n_bootstrap_samples = n_boostrap_samples
        self.dirichlet_alpha = dirichlet_alpha

    def _reset_for_embedding(self):
        if hasattr(self.graph, "_landmark_op"):
            delattr(self.graph, "_landmark_op")
        if hasattr(self.graph, "_diff_op"):
            delattr(self.graph, "_diff_op")
        self._diff_potential = None
        self.embedding = None

    def transform(self, X=None, t_max=100, plot_optimal_t=False, ax=None):
        if self.graph is None:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )
        orig_kernel = self.graph._kernel
        with _logger.log_task("Baseline embedding"):
            orig_embedding = super().transform(X, t_max, plot_optimal_t, ax)

        embeddings = []
        for i in range(self.n_bootstrap_samples):
            with _logger.log_task(f"Bootstrap embedding {i}"):
                weights = np.random.dirichlet(
                    alpha=np.ones(orig_kernel.shape[1]) * self.dirichlet_alpha
                )
                weighted_kernel = scipy.sparse.csr_matrix.multiply(
                    orig_kernel, scipy.sparse.csr_matrix(weights[None, :])
                )
                self.graph._kernel = weighted_kernel
                self._reset_for_embedding()
                embeddings.append(super().transform(X, t_max, plot_optimal_t, ax))

        return orig_embedding, np.stack(embeddings)
