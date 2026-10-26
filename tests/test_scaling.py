"""Tests for `magnetofluidics_pinn.scaling`, and the real configuration pipeline.

Two halves:

1. Unit tests of the conversion machinery. Every function is closed-form
   arithmetic, so each is checked against a hand-computed value and, for the
   `nondimensionalize_*` / `redimensionalize_*` pairs, against its own
   inverse.
2. `TestEndToEndPipeline`: the chain every user actually follows,

       DomainConfig -> build_channel_domain -> compute_scales
                    -> nondimensionalize_domain -> (network, residual,
                       training, trajectory)

   which no other test executes, because the shared `domain` fixture builds
   an already-dimensionless `Domain` by hand and `conftest.py` supplies
   `Scales(1, 1, 1, 1, 1)` where a scale is needed. A units mistake anywhere in
   this chain (a stray factor of `L`, a swapped ratio) would pass every other
   test in the suite unnoticed.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn import scaling

# A microfluidic channel: 750 um radius, 3 mm long, 1 mm/s peak flow, water-like fluid.
RADIUS = 7.5e-4
LENGTH = 3.0e-3
U_MAX = 1.0e-3
VISCOSITY = 1.0e-3
DENSITY = 1.0e3
FIELD_MAGNITUDE = 1.0e-2


@pytest.fixture
def physical_config() -> mfp.DomainConfig:
    return mfp.DomainConfig(
        kind="channel", length=LENGTH, radius=RADIUS, u_max=U_MAX,
        fluid=mfp.FluidConfig(dynamic_viscosity=VISCOSITY, density=DENSITY),
    )


@pytest.fixture
def scales(physical_config: mfp.DomainConfig) -> mfp.Scales:
    return mfp.compute_scales(physical_config, mfp.MagneticFieldConfig(magnitude=FIELD_MAGNITUDE))


class TestComputeScales:
    def test_length_and_velocity_come_from_the_domain_configuration(self, scales: mfp.Scales) -> None:
        assert scales.length == pytest.approx(RADIUS)
        assert scales.velocity == pytest.approx(U_MAX)

    def test_time_scale_is_length_over_velocity(self, scales: mfp.Scales) -> None:
        assert scales.time == pytest.approx(RADIUS / U_MAX)

    def test_pressure_scale_is_the_viscous_stress_scale(self, scales: mfp.Scales) -> None:
        """$P_c = \\mu U_c / L_c$, the natural pressure scale for Stokes flow."""
        assert scales.pressure == pytest.approx(VISCOSITY * U_MAX / RADIUS)

    def test_magnetic_scale_is_the_field_magnitude(self, scales: mfp.Scales) -> None:
        assert scales.magnetic_field == pytest.approx(FIELD_MAGNITUDE)

    def test_scales_are_immutable(self, scales: mfp.Scales) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            scales.length = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize("bad_velocity", [0.0, -1.0e-3])
    def test_non_positive_reference_velocity_raises(self, bad_velocity: float) -> None:
        # `DomainConfig` does not validate u_max, so this guard is reachable directly.
        config = mfp.DomainConfig(u_max=bad_velocity)
        with pytest.raises(ValueError, match="reference_velocity must be strictly positive"):
            mfp.compute_scales(config, mfp.MagneticFieldConfig())

    def test_non_positive_reference_length_raises(self) -> None:
        config = dataclasses.replace(mfp.DomainConfig())
        object.__setattr__(config, "radius", 0.0)  # bypass DomainConfig's own validation
        with pytest.raises(ValueError, match="reference_length must be strictly positive"):
            mfp.compute_scales(config, mfp.MagneticFieldConfig())


class TestDomainConversion:
    def _domain(self) -> mfp.Domain:
        return mfp.Domain(kind="channel", length=LENGTH, radius=RADIUS, u_max=U_MAX, branch_angle=None)

    def test_nondimensionalize_divides_lengths_and_velocity_by_their_scales(self, scales: mfp.Scales) -> None:
        result = mfp.nondimensionalize_domain(self._domain(), scales)
        assert result.length == pytest.approx(LENGTH / RADIUS)
        assert result.radius == pytest.approx(1.0)
        assert result.u_max == pytest.approx(1.0)

    def test_redimensionalize_multiplies_them_back(self, scales: mfp.Scales) -> None:
        dimensionless = mfp.Domain(kind="channel", length=4.0, radius=1.0, u_max=1.0)
        result = mfp.redimensionalize_domain(dimensionless, scales)
        assert (result.length, result.radius, result.u_max) == pytest.approx((4.0 * RADIUS, RADIUS, U_MAX))

    def test_round_trip_recovers_the_physical_domain(self, scales: mfp.Scales) -> None:
        original = self._domain()
        recovered = mfp.redimensionalize_domain(mfp.nondimensionalize_domain(original, scales), scales)
        assert (recovered.length, recovered.radius, recovered.u_max) == pytest.approx(
            (original.length, original.radius, original.u_max)
        )

    def test_kind_and_branch_angle_pass_through_unchanged(self, scales: mfp.Scales) -> None:
        """Angles are already dimensionless, so a bifurcation's must not be rescaled."""
        bifurcation = mfp.Domain(kind="bifurcation", length=LENGTH, radius=RADIUS, branch_angle=0.7)
        result = mfp.nondimensionalize_domain(bifurcation, scales)
        assert result.kind == "bifurcation"
        assert result.branch_angle == pytest.approx(0.7)

    def test_input_domain_is_not_mutated(self, scales: mfp.Scales) -> None:
        original = self._domain()
        mfp.nondimensionalize_domain(original, scales)
        assert original.radius == RADIUS


_SCALAR_CONVERSIONS = [
    ("velocity", scaling.nondimensionalize_velocity, scaling.redimensionalize_velocity, "velocity"),
    ("pressure", scaling.nondimensionalize_pressure, scaling.redimensionalize_pressure, "pressure"),
    ("time", scaling.nondimensionalize_time, scaling.redimensionalize_time, "time"),
    ("magnetic field", scaling.nondimensionalize_magnetic_field, scaling.redimensionalize_magnetic_field,
     "magnetic_field"),
]


class TestScalarConversions:
    """Each pair divides / multiplies by its own characteristic scale."""

    @pytest.mark.parametrize("label,nondim,redim,scale_name", _SCALAR_CONVERSIONS, ids=[c[0] for c in _SCALAR_CONVERSIONS])
    def test_nondimensionalize_divides_by_the_scale(self, label, nondim, redim, scale_name, scales: mfp.Scales) -> None:
        scale = getattr(scales, scale_name)
        assert nondim(3.0 * scale, scales) == pytest.approx(3.0)

    @pytest.mark.parametrize("label,nondim,redim,scale_name", _SCALAR_CONVERSIONS, ids=[c[0] for c in _SCALAR_CONVERSIONS])
    def test_redimensionalize_multiplies_by_the_scale(self, label, nondim, redim, scale_name, scales: mfp.Scales) -> None:
        scale = getattr(scales, scale_name)
        assert redim(3.0, scales) == pytest.approx(3.0 * scale)

    @pytest.mark.parametrize("label,nondim,redim,scale_name", _SCALAR_CONVERSIONS, ids=[c[0] for c in _SCALAR_CONVERSIONS])
    @pytest.mark.parametrize("value", [-2.5, 0.0, 1.0e-6, 42.0])
    def test_round_trip_is_the_identity(self, label, nondim, redim, scale_name, value: float, scales: mfp.Scales) -> None:
        assert redim(nondim(value, scales), scales) == pytest.approx(value)


class TestParticleScaling:
    def test_nondimensionalize_particle_divides_the_radius_by_the_length_scale(self, scales: mfp.Scales) -> None:
        particle = mfp.ParticleConfig(radius=1.0e-5)
        assert mfp.nondimensionalize_particle(particle, scales) == pytest.approx(1.0e-5 / RADIUS)


    def test_magnetic_moment_is_deliberately_not_converted(self, scales: mfp.Scales) -> None:
        """Documented: no dimensionally-consistent moment scaling exists yet."""
        particle = mfp.ParticleConfig(radius=1.0e-5, magnetic_moment=(1.0e-11, 0.0))
        mfp.nondimensionalize_particle(particle, scales)
        assert particle.magnetic_moment == (1.0e-11, 0.0)

    def test_mobility_scale_is_the_stokes_mobility_of_the_reference_length(self, physical_config: mfp.DomainConfig) -> None:
        """$M_\\text{ref} = 1 / (6 \\pi \\mu L_\\text{ref})$."""
        expected = 1.0 / (6.0 * math.pi * VISCOSITY * RADIUS)
        assert scaling.compute_mobility_scale(physical_config) == pytest.approx(expected)

    def test_sphere_mobility_is_isotropic_and_inversely_proportional_to_the_radius(self, scales: mfp.Scales) -> None:
        particle = mfp.ParticleConfig(kind="spherical", radius=1.0e-5)
        mobility = mfp.nondimensionalize_mobility(particle, mfp.DomainConfig(), scales)

        a_star = 1.0e-5 / RADIUS
        assert mobility.shape == (2, 2)
        assert torch.allclose(mobility, torch.diag(torch.tensor([1.0 / a_star, 1.0 / a_star])))
        assert mobility[0, 1].item() == 0.0 and mobility[1, 0].item() == 0.0

    def test_larger_particle_is_less_mobile(self, scales: mfp.Scales) -> None:
        small = mfp.nondimensionalize_mobility(mfp.ParticleConfig(radius=1.0e-5), mfp.DomainConfig(), scales)
        large = mfp.nondimensionalize_mobility(mfp.ParticleConfig(radius=2.0e-5), mfp.DomainConfig(), scales)
        assert small[0, 0].item() == pytest.approx(2.0 * large[0, 0].item())

    @pytest.mark.parametrize(
        "particle",
        [
            mfp.ParticleConfig(kind="cylinder", radius=1.0e-5, length=4.0e-5),
            mfp.ParticleConfig(kind="spheroid", radius=1.0e-5, length=4.0e-5, aspect_ratio=0.5),
            mfp.ParticleConfig(kind="swarms", radius=1.0e-5, number=5),
        ],
        ids=["cylinder", "spheroid", "swarms"],
    )
    def test_non_spherical_shapes_are_not_yet_supported(self, particle: mfp.ParticleConfig, scales: mfp.Scales) -> None:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            mfp.nondimensionalize_mobility(particle, mfp.DomainConfig(), scales)


class TestEndToEndPipeline:
    """The real, documented route from physical configuration to a trained-network verdict."""

    @pytest.fixture
    def pipeline(self, physical_config: mfp.DomainConfig, scales: mfp.Scales):
        physical_domain = mfp.build_channel_domain(physical_config)
        return physical_domain, mfp.nondimensionalize_domain(physical_domain, scales)

    def test_reference_length_equals_the_radius_so_the_dimensionless_radius_is_exactly_one(self, pipeline) -> None:
        _, domain = pipeline
        assert domain.radius == pytest.approx(1.0)

    def test_dimensionless_length_is_the_physical_aspect_ratio(self, pipeline) -> None:
        _, domain = pipeline
        assert domain.length == pytest.approx(LENGTH / RADIUS)
        assert domain.kind == "channel"

    def test_redimensionalizing_recovers_the_physical_domain(self, pipeline, scales: mfp.Scales) -> None:
        physical, domain = pipeline
        recovered = mfp.redimensionalize_domain(domain, scales)
        assert (recovered.length, recovered.radius) == pytest.approx((physical.length, physical.radius))

    def test_scales_are_mutually_consistent(self, physical_config: mfp.DomainConfig, scales: mfp.Scales) -> None:
        """$Re = \\rho U L / \\mu$ must be recoverable from the scales alone."""
        reynolds_from_scales = DENSITY * scales.velocity * scales.length / VISCOSITY
        assert reynolds_from_scales == pytest.approx(physical_config.reynolds)
        assert scales.time == pytest.approx(scales.length / scales.velocity)

    def test_dimensionless_domain_supports_a_stokes_residual_evaluation(
            self, pipeline, physical_config: mfp.DomainConfig
    ) -> None:
        """Sampling, the hard-wall network and the residual all accept the scaled domain."""
        _, domain = pipeline
        training_config = mfp.TrainingConfig(device="cpu", random_seed=0)
        network = mfp.apply_hard_wall_constraint(
            mfp.build_mlp(2, 3, (8, 8), training_config=training_config), domain
        )
        collocation = mfp.sample_collocation_points(domain, n_interior=32, n_boundary=9, random_seed=0, device="cpu")
        interior = collocation.interior.clone().requires_grad_(True)

        residual = mfp.stokes_residual(network, interior, physical_config)

        assert residual.shape == (32, 3)
        assert torch.isfinite(residual).all()
        # Every interior point must actually lie inside the *scaled* channel.
        assert (interior[:, 0] <= domain.radius).all() and (interior[:, 1] <= domain.length).all()

    def test_a_short_training_run_completes_on_the_scaled_domain(
            self, pipeline, physical_config: mfp.DomainConfig
    ) -> None:
        _, domain = pipeline
        training_config = mfp.TrainingConfig(
            n_interior_points=32, n_boundary_points=9, n_epochs=2, device="cpu", use_lbfgs_refinement=False,
        )
        network = mfp.apply_hard_wall_constraint(
            mfp.build_mlp(2, 3, (8, 8), training_config=training_config), domain
        )
        # `train`'s parameter is still *named* `fluid_config`, but it is threaded straight
        # into `stokes_residual`, which now needs a `DomainConfig` (see the notes in
        # test_training_trainer.py); a bare `FluidConfig` fails on the first residual.
        _, history = mfp.train(network, domain, fluid_config=physical_config, training_config=training_config)

        assert len(history.adam.total) == 2
        assert all(math.isfinite(value) for value in history.adam.total)

    def test_scaled_particle_and_mobility_drive_a_trajectory_integration(
            self, pipeline, scales: mfp.Scales, physical_config: mfp.DomainConfig
    ) -> None:
        """The integrator normalizes the particle radius and wall limit by `scales.length`;
        `conftest`'s unit `Scales` can never expose a wrong normalization, real scales can.
        """
        _, domain = pipeline
        particle = mfp.ParticleConfig(radius=1.0e-5, magnetic_moment=(1.0e-11, 0.0))
        training_config = mfp.TrainingConfig(device="cpu", random_seed=0)
        network = mfp.apply_hard_wall_constraint(
            mfp.build_mlp(2, 3, (8, 8), training_config=training_config), domain
        )
        state = mfp.ParticleState(
            position=torch.tensor([0.3 * domain.radius, 0.05 * domain.length]), velocity=torch.zeros(2), time=0.0,
        )
        mobility = mfp.nondimensionalize_mobility(particle, physical_config, scales).unsqueeze(0)

        def field_fn(coordinates: torch.Tensor) -> mfp.MagneticFieldSample:
            return mfp.uniform_field(coordinates, magnitude=FIELD_MAGNITUDE, orientation=(1.0, 0.0))

        trajectories = mfp.integrate_trajectory(
            network, field_fn, domain, scales, particle, [state], mobility, time_span=(0.0, 0.01),
        )

        assert len(trajectories) == 1 and len(trajectories[0]) >= 2
        assert trajectories[0][0].position.tolist() == pytest.approx(state.position.tolist())
        assert all(torch.isfinite(step.position).all() for step in trajectories[0])
        # The scaled wall limit is R* - a/L; a healthy short run must stay inside it.
        wall_limit = domain.radius - particle.max_surface_extension / scales.length
        assert all(step.position[0].item() <= wall_limit + 1e-6 for step in trajectories[0])
