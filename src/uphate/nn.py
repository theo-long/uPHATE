import jax.numpy as jnp
import jax
import optax
from flax import nnx, struct
from typing import Any, Callable
from pathlib import Path
import orbax.checkpoint as ocp


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    dtype: Any = jnp.float32
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 2048
    kernel_init: Callable = nnx.initializers.xavier_uniform()
    bias_init: Callable = nnx.initializers.normal(stddev=1e-6)


class MlpBlock(nnx.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    def __init__(self, rngs: nnx.Rngs, config: TransformerConfig, out_dim: int) -> None:
        self.lin1 = nnx.Linear(
            in_features=config.qkv_dim,
            out_features=config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            rngs=rngs,
        )
        self.lin2 = nnx.Linear(
            in_features=config.mlp_dim,
            out_features=out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            rngs=rngs,
        )

    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        x = self.lin1(inputs)
        x = nnx.elu(x)
        output = self.lin2(x)
        return output


class Encoder1DBlock(nnx.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        config: TransformerConfig,
    ):
        self.ln1 = nnx.LayerNorm(
            num_features=in_features, dtype=config.dtype, rngs=rngs
        )
        self.ln2 = nnx.LayerNorm(
            num_features=in_features, dtype=config.dtype, rngs=rngs
        )
        self.mha = nnx.MultiHeadAttention(
            in_features=in_features,
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            rngs=rngs,
            decode=False,
            deterministic=True,
        )
        self.mlp_block = MlpBlock(rngs, config, out_features)

    def __call__(self, inputs):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.

        Returns:
          output after transformer encoder block.
        """
        x = self.ln1(inputs)
        x = self.mha(x)
        x = self.ln2(x)
        y = self.mlp_block(x)
        return x + y  # residual connection


class Transformer(nnx.Module):
    """Transformer Model for sequence tagging."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        config: TransformerConfig,
    ) -> None:
        self.lin_embedding = nnx.Linear(
            rngs=rngs, in_features=in_features, out_features=config.qkv_dim
        )
        self.transformer_blocks = nnx.Sequential(
            *(
                Encoder1DBlock(config.qkv_dim, config.qkv_dim, rngs, config)
                for _ in range(config.num_layers)
            )
        )
        self.ln = nnx.LayerNorm(config.qkv_dim, dtype=config.dtype, rngs=rngs)
        self.readout = nnx.Linear(
            rngs=rngs, in_features=config.qkv_dim, out_features=out_features
        )

    def __call__(self, inputs):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          train: if it is training.

        Returns:
          output of a transformer encoder.

        """
        x = self.lin_embedding(inputs)
        x = self.transformer_blocks(x)
        x = self.ln(x)
        logits = self.readout(x)
        return logits


def loss_fn(model: Transformer, X: jax.Array, X_phate: jax.Array):
    predictions = model(X)
    loss = optax.squared_error(predictions, X_phate).mean()
    return loss, predictions


@nnx.jit
def train_step(
    model: Transformer,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    X: jax.Array,
    X_phate: jax.Array,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, emb), grads = grad_fn(model, X, X_phate)
    metrics.update(loss=loss, std=jnp.std(emb, axis=0))
    optimizer.update(grads)
    return loss


def train_phate_surrogate(
    X,
    X_phate,
    config: TransformerConfig,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed=42,
):
    """Train differentiable surrogate model for PHATE embedding."""
    X, X_phate = jnp.array(X), jnp.array(X_phate)
    rngs = nnx.Rngs(seed)
    model = Transformer(X.shape[1], X_phate.shape[1], rngs=rngs, config=config)
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate, momentum, weight_decay=weight_decay),
        wrt=nnx.Param,
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        std=nnx.metrics.Average("std"),
    )

    loss = jnp.inf
    for step in range(epochs):
        model.train()
        loss = train_step(model, optimizer, metrics, X, X_phate)
        print(
            f"step {step}",
            *(
                f" | {metric}: {value:.4f}"
                for metric, value in metrics.compute().items()
            ),
        )
        metrics.reset()

    print(f"Final loss: {loss:.4f}")
    model.eval()
    return model


def load_orbax_checkpoint(base_model, checkpoint_path: str | Path):
    abstract_model = nnx.eval_shape(
        lambda: base_model,
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        checkpoint_path,
        item=jax.tree.map(lambda x: jnp.zeros(x.shape), abstract_state),
    )
    return nnx.merge(graphdef, state_restored)
