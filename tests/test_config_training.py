"""Unit tests for the `TrainingConfig` fields added in this change set.

Covers the six new fields (`momentum_loss_weight`, `continuity_loss_weight`,
`positivity_loss_weight`, `conservation_loss_weight`,
`n_conservation_stations`, `n_conservation_quadrature_points`): their
defaults, that a zero weight is a legal way to disable a term, and every
invalid value `__post_init__` is documented to reject.
"""

from __future__ import annotations

import math

import pytest

import magnetofluidics_pinn as mfp


def test_defaults_are_valid_and_documented_values() -> None:
    config = mfp.TrainingConfig()
    assert config.momentum_loss_weight == 1.0
    assert config.continuity_loss_weight == 20.0
    assert config.positivity_loss_weight == 1.0
    assert config.conservation_loss_weight == 20.0
    assert config.n_conservation_stations == 8
    assert config.n_conservation_quadrature_points == 64


@pytest.mark.parametrize(
    "field_name",
    ["momentum_loss_weight", "continuity_loss_weight", "positivity_loss_weight", "conservation_loss_weight"],
)
def test_zero_weight_is_allowed(field_name: str) -> None:
    """A weight of exactly zero disables a term without needing special-casing elsewhere."""
    config = mfp.TrainingConfig(**{field_name: 0.0})
    assert getattr(config, field_name) == 0.0


@pytest.mark.parametrize(
    "field_name",
    ["momentum_loss_weight", "continuity_loss_weight", "positivity_loss_weight", "conservation_loss_weight"],
)
@pytest.mark.parametrize("bad_value", [-1.0, -0.001, math.nan, math.inf, -math.inf])
def test_invalid_weight_raises(field_name: str, bad_value: float) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be finite and non-negative"):
        mfp.TrainingConfig(**{field_name: bad_value})


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_non_positive_n_conservation_stations_raises(bad_value: int) -> None:
    with pytest.raises(ValueError, match="n_conservation_stations must be strictly positive"):
        mfp.TrainingConfig(n_conservation_stations=bad_value)


@pytest.mark.parametrize("bad_value", [0, 1, -5])
def test_too_few_conservation_quadrature_points_raises(bad_value: int) -> None:
    with pytest.raises(ValueError, match="n_conservation_quadrature_points must be at least 2"):
        mfp.TrainingConfig(n_conservation_quadrature_points=bad_value)


def test_custom_weights_round_trip() -> None:
    """Non-default values are stored verbatim (the dataclass is otherwise frozen)."""
    config = mfp.TrainingConfig(
        momentum_loss_weight=2.5,
        continuity_loss_weight=15.0,
        positivity_loss_weight=0.5,
        conservation_loss_weight=30.0,
        n_conservation_stations=12,
        n_conservation_quadrature_points=128,
    )
    assert config.momentum_loss_weight == 2.5
    assert config.continuity_loss_weight == 15.0
    assert config.positivity_loss_weight == 0.5
    assert config.conservation_loss_weight == 30.0
    assert config.n_conservation_stations == 12
    assert config.n_conservation_quadrature_points == 128
