"""Unit tests for `TrainingConfig`.

The first section covers the six fields added in an earlier change set
(`momentum_loss_weight`, `continuity_loss_weight`, `positivity_loss_weight`,
`conservation_loss_weight`, `n_conservation_stations`,
`n_conservation_quadrature_points`): their defaults, that a zero weight is a
legal way to disable a term, and every invalid value `__post_init__` is
documented to reject. The second section covers every remaining
`__post_init__` branch (point counts, learning rate, epochs, logging,
device, and the `axis_clearance_fraction`/`residual_form` cross-field
guard), plus the `torch_device`/`torch_dtype` properties.
"""

from __future__ import annotations

import math

import pytest
import torch

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


# ---------------------------------------------------------------------------
# The remaining `TrainingConfig.__post_init__` validation, not covered above:
# point counts, the learning-rate/epoch/logging/device checks, and the
# axis_clearance_fraction/residual_form cross-field guard - the one thing
# standing between a caller and the 1/r**2 blow-up
# `sampling.collocation`'s own module docstring warns about.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", ["n_interior_points", "n_boundary_points"])
def test_non_positive_point_count_raises(field_name: str) -> None:
    with pytest.raises(ValueError, match="n_interior_points and n_boundary_points must be strictly"):
        mfp.TrainingConfig(**{field_name: 0})


def test_too_few_boundary_points_raises() -> None:
    with pytest.raises(ValueError, match="n_boundary_points must be at least 3"):
        mfp.TrainingConfig(n_boundary_points=2)


def test_non_positive_n_axis_points_raises() -> None:
    with pytest.raises(ValueError, match="n_axis_points must be strictly positive"):
        mfp.TrainingConfig(n_axis_points=0)


@pytest.mark.parametrize("bad_value", [-0.1, 1.0, 1.5, math.nan, math.inf])
def test_axis_clearance_fraction_out_of_range_raises(bad_value: float) -> None:
    with pytest.raises(ValueError, match="axis_clearance_fraction must be finite and lie in"):
        mfp.TrainingConfig(axis_clearance_fraction=bad_value, residual_form="r_weighted")


def test_zero_axis_clearance_with_standard_form_raises() -> None:
    """The exact cross-field guard against sampling flush to the 1/r singularity."""
    with pytest.raises(ValueError, match="only numerically safe with residual_form='r_weighted'"):
        mfp.TrainingConfig(axis_clearance_fraction=0.0, residual_form="standard")


def test_zero_axis_clearance_with_r_weighted_form_is_allowed() -> None:
    """The r-weighted residual removes the singularity analytically, so 0.0 is safe there."""
    config = mfp.TrainingConfig(axis_clearance_fraction=0.0, residual_form="r_weighted")
    assert config.axis_clearance_fraction == 0.0


def test_none_axis_clearance_fraction_is_allowed_with_either_form() -> None:
    """`None` reproduces the sampling module's own default clearance; never rejected."""
    assert mfp.TrainingConfig(axis_clearance_fraction=None, residual_form="standard").axis_clearance_fraction is None


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_non_positive_learning_rate_raises(bad_value: float) -> None:
    with pytest.raises(ValueError, match=f"learning_rate must be finite and strictly positive; got {bad_value!r}."):
        mfp.TrainingConfig(learning_rate=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_non_positive_n_epochs_raises(bad_value: int) -> None:
    with pytest.raises(ValueError, match="n_epochs must be strictly positive"):
        mfp.TrainingConfig(n_epochs=bad_value)


def test_verbose_with_log_every_below_one_raises() -> None:
    with pytest.raises(ValueError, match="log_every must be at least 1 when verbose=True"):
        mfp.TrainingConfig(verbose=True, log_every=0)


def test_log_every_below_one_is_allowed_when_not_verbose() -> None:
    """The guard is only meaningful once verbose logging would actually divide by it."""
    config = mfp.TrainingConfig(verbose=False, log_every=0)
    assert config.log_every == 0


def test_invalid_device_string_raises() -> None:
    with pytest.raises(ValueError, match="device type must be 'cuda' or 'cpu'"):
        mfp.TrainingConfig(device="tpu")


def test_torch_device_property_resolves_cpu() -> None:
    config = mfp.TrainingConfig(device="cpu")
    assert config.torch_device == mfp.resolve_device("cpu")


@pytest.mark.parametrize("dtype_name,expected_dtype", [("float32", torch.float32), ("float64", torch.float64)])
def test_torch_dtype_property(dtype_name: str, expected_dtype: torch.dtype) -> None:
    assert mfp.TrainingConfig(dtype=dtype_name).torch_dtype == expected_dtype
