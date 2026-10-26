"""Unit tests for the configuration dataclasses in `magnetofluidics_pinn.config`.

`TrainingConfig` has its own module (`test_config_training.py`); the radius
check of `ParticleConfig` is covered in `test_particle_two_way_coupling.py`.
This module covers every other `__post_init__` branch, the computed
properties (`DomainConfig.reynolds`, `ParticleConfig.volume` and friends),
`TwoWayCouplingConfig`'s helper properties, and `set_random_seed`.

Each dataclass follows the same pattern: a valid construction, then one test
per documented rejection, so a regression in any single guard names itself.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn import config as config_module

_BAD_POSITIVE_VALUES = [0.0, -1.0, math.nan, math.inf, -math.inf]


class TestFluidConfig:
    def test_defaults_describe_a_stokes_water_like_fluid(self) -> None:
        config = mfp.FluidConfig()
        assert config.regime == "stokes"
        assert config.dynamic_viscosity == pytest.approx(1.0e-3)
        assert config.density == pytest.approx(1.0e3)

    @pytest.mark.parametrize("field_name", ["dynamic_viscosity", "density"])
    @pytest.mark.parametrize("bad_value", _BAD_POSITIVE_VALUES)
    def test_invalid_field_raises(self, field_name: str, bad_value: float) -> None:
        with pytest.raises(ValueError, match=f"{field_name} must be finite and strictly positive"):
            mfp.FluidConfig(**{field_name: bad_value})


class TestDomainConfig:
    def test_default_is_a_valid_channel(self) -> None:
        config = mfp.DomainConfig()
        assert config.kind == "channel"
        assert config.branch_angle is None

    @pytest.mark.parametrize("field_name", ["length", "radius"])
    @pytest.mark.parametrize("bad_value", _BAD_POSITIVE_VALUES)
    def test_invalid_geometry_raises(self, field_name: str, bad_value: float) -> None:
        with pytest.raises(ValueError, match=f"{field_name} must be finite and strictly positive"):
            mfp.DomainConfig(**{field_name: bad_value})

    def test_valid_bifurcation_is_accepted(self) -> None:
        config = mfp.DomainConfig(kind="bifurcation", branch_angle=0.5)
        assert config.branch_angle == pytest.approx(0.5)

    def test_bifurcation_without_branch_angle_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a finite branch_angle"):
            mfp.DomainConfig(kind="bifurcation")

    @pytest.mark.parametrize("bad_angle", [math.nan, math.inf])
    def test_bifurcation_with_non_finite_branch_angle_raises(self, bad_angle: float) -> None:
        with pytest.raises(ValueError, match="requires a finite branch_angle"):
            mfp.DomainConfig(kind="bifurcation", branch_angle=bad_angle)

    def test_channel_with_branch_angle_raises(self) -> None:
        with pytest.raises(ValueError, match="branch_angle is only meaningful when kind='bifurcation'"):
            mfp.DomainConfig(kind="channel", branch_angle=0.5)

    def test_reference_scales_follow_radius_and_u_max(self) -> None:
        config = mfp.DomainConfig(radius=2.0e-3, u_max=5.0e-3)
        assert config.reference_length == pytest.approx(2.0e-3)
        assert config.reference_velocity == pytest.approx(5.0e-3)

    def test_reynolds_matches_hand_computation(self) -> None:
        """$Re = \\rho U_c L_c / \\mu$, with $U_c$ = u_max and $L_c$ = radius.

        This is exactly the property whose location (DomainConfig, not
        FluidConfig) caused the `navier_stokes_residual` regression, so its
        formula gets a direct, hand-checked test.
        """
        fluid = mfp.FluidConfig(dynamic_viscosity=1.0e-3, density=1.0e3)
        config = mfp.DomainConfig(radius=1.0e-3, u_max=2.0e-3, fluid=fluid)
        assert config.reynolds == pytest.approx(1.0e3 * 2.0e-3 * 1.0e-3 / 1.0e-3)

    def test_reynolds_scales_linearly_with_velocity(self) -> None:
        slow = mfp.DomainConfig(u_max=1.0e-3).reynolds
        fast = mfp.DomainConfig(u_max=4.0e-3).reynolds
        assert fast == pytest.approx(4.0 * slow)


class TestMagneticFieldConfig:
    def test_default_is_a_valid_uniform_field(self) -> None:
        config = mfp.MagneticFieldConfig()
        assert config.source == "uniform"
        assert config.orientation == (1.0, 0.0)

    @pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
    def test_invalid_magnitude_raises(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="magnitude must be finite and non-negative"):
            mfp.MagneticFieldConfig(magnitude=bad_value)

    @pytest.mark.parametrize("bad_orientation", [(math.nan, 0.0), (1.0, math.inf)])
    def test_non_finite_orientation_raises(self, bad_orientation: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="orientation components must be finite"):
            mfp.MagneticFieldConfig(orientation=bad_orientation)

    @pytest.mark.parametrize("bad_orientation", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.0)])
    def test_non_unit_orientation_raises(self, bad_orientation: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="orientation must be a unit vector"):
            mfp.MagneticFieldConfig(orientation=bad_orientation)

    def test_diagonal_unit_vector_is_accepted(self) -> None:
        assert mfp.MagneticFieldConfig(orientation=(0.6, 0.8)).orientation == (0.6, 0.8)


class TestParticleConfigValidation:
    """Every `ParticleConfig` rejection other than `radius` (see `test_particle_two_way_coupling.py`)."""

    def test_non_finite_magnetic_moment_raises(self) -> None:
        with pytest.raises(ValueError, match="magnetic_moment components must be finite"):
            mfp.ParticleConfig(magnetic_moment=(math.nan, 0.0))

    def test_zero_magnetic_moment_raises(self) -> None:
        with pytest.raises(ValueError, match="magnetic_moment must be non-zero"):
            mfp.ParticleConfig(magnetic_moment=(0.0, 0.0))

    def test_negative_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length must be finite and non-negative"):
            mfp.ParticleConfig(length=-1.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan])
    def test_invalid_aspect_ratio_raises(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="aspect_ratio must be finite and strictly positive"):
            mfp.ParticleConfig(aspect_ratio=bad_value)

    def test_negative_number_raises(self) -> None:
        with pytest.raises(ValueError, match="number must be positive"):
            mfp.ParticleConfig(number=-1)

    def test_negative_magnetic_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="magnetic_ratio must be finite and non-negative"):
            mfp.ParticleConfig(magnetic_ratio=-0.5)

    def test_swarm_with_zero_particles_raises(self) -> None:
        with pytest.raises(ValueError, match="number of particles must be positive for swarms"):
            mfp.ParticleConfig(kind="swarms", radius=1.0e-5, number=0)

    def test_cylinder_without_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length of cylinder must be positive"):
            mfp.ParticleConfig(kind="cylinder", radius=1.0e-5, length=0.0)

    @pytest.mark.parametrize("bad_aspect_ratio", [1.5, 2.0])
    def test_spheroid_aspect_ratio_above_one_raises(self, bad_aspect_ratio: float) -> None:
        with pytest.raises(ValueError, match="aspect_ratio of spheroid must be between 0 and 1"):
            mfp.ParticleConfig(kind="spheroid", radius=1.0e-5, length=2.0e-5, aspect_ratio=bad_aspect_ratio)


class TestParticleConfigProperties:
    """Closed-form geometry and magnetization, checked against hand-derived values."""

    def test_sphere_volume(self) -> None:
        config = mfp.ParticleConfig(kind="spherical", radius=2.0)
        assert config.volume == pytest.approx(4.0 / 3.0 * math.pi * 2.0**3)

    def test_cylinder_volume(self) -> None:
        config = mfp.ParticleConfig(kind="cylinder", radius=1.0, length=2.0)
        assert config.volume == pytest.approx(2.0 * math.pi)

    def test_spheroid_volume(self) -> None:
        config = mfp.ParticleConfig(kind="spheroid", radius=1.0, length=2.0, aspect_ratio=0.5)
        assert config.volume == pytest.approx(4.0 / 3.0 * math.pi)

    def test_swarm_volume_scales_with_particle_count(self) -> None:
        single = mfp.ParticleConfig(kind="swarms", radius=1.0, number=1).volume
        triple = mfp.ParticleConfig(kind="swarms", radius=1.0, number=3).volume
        assert triple == pytest.approx(3.0 * single)

    def test_max_surface_extension_of_a_sphere_is_its_radius(self) -> None:
        assert mfp.ParticleConfig(radius=3.0).max_surface_extension == pytest.approx(3.0)

    def test_max_surface_extension_of_a_cylinder_is_its_half_diagonal(self) -> None:
        config = mfp.ParticleConfig(kind="cylinder", radius=3.0, length=8.0)
        assert config.max_surface_extension == pytest.approx(5.0)  # hypot(3, 4)

    def test_magnetic_volume_applies_the_magnetic_ratio(self) -> None:
        config = mfp.ParticleConfig(radius=1.0, magnetic_ratio=0.25)
        assert config.magnetic_volume == pytest.approx(0.25 * config.volume)

    def test_magnetization_is_moment_per_magnetic_volume(self) -> None:
        config = mfp.ParticleConfig(radius=1.0, magnetic_moment=(2.0, 4.0), magnetic_ratio=0.5)
        expected = (2.0 / config.magnetic_volume, 4.0 / config.magnetic_volume)
        assert config.magnetization == pytest.approx(expected)

    def test_with_magnetization_round_trips_the_moment(self) -> None:
        config = mfp.ParticleConfig(radius=1.0, magnetic_moment=(2.0, 4.0), magnetic_ratio=0.5)
        rebuilt = config.with_magnetization(config.magnetization)
        assert rebuilt.magnetic_moment == pytest.approx(config.magnetic_moment)

    def test_with_magnetization_returns_a_new_instance_and_leaves_the_original_untouched(self) -> None:
        config = mfp.ParticleConfig(radius=1.0, magnetic_moment=(2.0, 4.0))
        updated = config.with_magnetization((1.0, 1.0))
        assert updated is not config
        assert config.magnetic_moment == (2.0, 4.0)


class TestTwoWayCouplingConfig:
    """Remaining `TwoWayCouplingConfig` behavior (finite-velocity and default checks
    live in `test_training_trainer.py::TestTrainWithCoupling`).
    """

    def test_non_positive_n_surface_points_raises(self, particle_config: mfp.ParticleConfig) -> None:
        with pytest.raises(ValueError, match="n_surface_points must be strictly positive"):
            mfp.TwoWayCouplingConfig(particle_config=particle_config, n_surface_points=0)

    @pytest.mark.parametrize("bad_weight", [-1.0, math.nan, math.inf])
    def test_invalid_particle_loss_weight_raises(self, particle_config: mfp.ParticleConfig, bad_weight: float) -> None:
        with pytest.raises(ValueError, match="particle_loss_weight must be finite"):
            mfp.TwoWayCouplingConfig(particle_config=particle_config, particle_loss_weight=bad_weight)

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "Docstring/code mismatch: TwoWayCouplingConfig documents particle_loss_weight as "
            "'finite and non-negative' (and every TrainingConfig loss weight treats 0.0 as a legal "
            "way to disable a term), but __post_init__ validates it with "
            "_check_if_finite_positive_strictly, which rejects 0.0. Flagged, not changed here: "
            "decide whether the docstring or the guard is the intended contract."
        ),
    )
    def test_zero_particle_loss_weight_is_allowed(self, particle_config: mfp.ParticleConfig) -> None:
        """A zero weight should disable the particle-surface term, as for every other loss weight."""
        assert mfp.TwoWayCouplingConfig(particle_config=particle_config, particle_loss_weight=0.0).particle_loss_weight == 0.0

    def test_velocity_helper_properties(self, particle_config: mfp.ParticleConfig) -> None:
        config = mfp.TwoWayCouplingConfig(particle_config=particle_config, particle_velocity=(3.0, 4.0))
        assert config.particle_velocity_magnitude == pytest.approx(5.0)
        assert config.radial_particle_velocity == pytest.approx(3.0)
        assert config.axial_particle_velocity == pytest.approx(4.0)

    def test_particle_state_reflects_position_velocity_and_time(self, particle_config: mfp.ParticleConfig) -> None:
        config = mfp.TwoWayCouplingConfig(
            particle_config=particle_config, particle_position=(0.0, 1.5),
            particle_velocity=(0.0, -0.25), time=2.0,
        )
        state = config.particle_state
        assert state.position.tolist() == pytest.approx([0.0, 1.5])
        assert state.velocity.tolist() == pytest.approx([0.0, -0.25])
        assert state.time == pytest.approx(2.0)

    def test_default_particle_loss_weight_matches_the_shared_boundary_weight(
            self, particle_config: mfp.ParticleConfig
    ) -> None:
        config = mfp.TwoWayCouplingConfig(particle_config=particle_config)
        assert config.particle_loss_weight == mfp.DEFAULT_BOUNDARY_LOSS_WEIGHT


class TestSetRandomSeed:
    def test_none_leaves_every_generator_untouched(self) -> None:
        torch.manual_seed(7)
        expected = torch.rand(3)
        torch.manual_seed(7)
        mfp.set_random_seed(None)
        assert torch.equal(torch.rand(3), expected)

    def test_same_seed_reproduces_torch_numpy_and_python_draws(self) -> None:
        mfp.set_random_seed(123)
        first = (torch.rand(3), np.random.rand(3), random.random())
        mfp.set_random_seed(123)
        second = (torch.rand(3), np.random.rand(3), random.random())

        assert torch.equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_different_seeds_give_different_draws(self) -> None:
        mfp.set_random_seed(1)
        first = torch.rand(5)
        mfp.set_random_seed(2)
        assert not torch.equal(first, torch.rand(5))

    def test_cuda_branch_seeds_every_device_when_cuda_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercised via monkeypatching, so it runs on machines with no GPU too."""
        recorded: list[int] = []
        monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(config_module.torch.cuda, "manual_seed_all", recorded.append)

        mfp.set_random_seed(99)

        # `torch.manual_seed` itself already seeds every CUDA device, and
        # `set_random_seed` then calls `manual_seed_all` explicitly as well, so the
        # recorder may see the seed more than once; what matters is that it only
        # ever sees the requested one.
        assert recorded, "manual_seed_all was never called on the CUDA branch"
        assert set(recorded) == {99}
