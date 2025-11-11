import jax.numpy as jnp
import pytest
from phate import PHATE
from sklearn.datasets import make_swiss_roll

from uphate import get_phate_embedding
from uphate.uphate import (
    compute_affinity_matrix,
    compute_diff_op,
    compute_diffusion_potential,
)


@pytest.fixture
def swiss_roll_data():
    X, _ = make_swiss_roll(n_samples=300, random_state=42)
    return X


@pytest.mark.parametrize("gamma", [1])
@pytest.mark.parametrize("t", [5, 25, 50])
@pytest.mark.parametrize("decay", [10, 20, 40])
@pytest.mark.parametrize("knn", [2, 5, 10])
def test_phate_match(
    swiss_roll_data,
    knn,
    decay,
    t,
    gamma,
):
    X = swiss_roll_data

    print(X.shape)

    n_components = 2
    phate_op = PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        t=t,
        n_landmark=None,  # type: ignore
        mds_solver="smacof",
        gamma=gamma,
    )

    _ = phate_op.fit_transform(X)

    affinity_matrix = compute_affinity_matrix(
        jnp.array(X),
        n_landmark=1,
        knn=knn,
        decay=decay,
    )

    diff_op = compute_diff_op(affinity_matrix)
    diff_potential = compute_diffusion_potential(diff_op=diff_op, t=t, gamma=gamma)

    assert jnp.allclose(phate_op.graph.K.toarray(), affinity_matrix, atol=1e-5), (  # type: ignore
        "Affinity matrices do not match"
    )
    assert jnp.allclose(diff_op, phate_op.diff_op, atol=1e-5), "Diff Ops do not match"
    assert jnp.allclose(diff_potential, phate_op.diff_potential, atol=1e-5), (
        "Diff Potentials do not match"
    )
