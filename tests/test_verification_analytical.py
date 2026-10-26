"""Unit tests for `magnetofluidics_pinn.verification.analytical`.

This module is the closed-form ground truth every other verification check
(and the shipped notebook) trusts blindly, and it had zero direct test
coverage: everything here is deterministic, pure math, so each function's
formula is checked against a hand-derived value rather than merely
"does not raise".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import magnetofluidics_pinn as mfp


class TestAnalyticalPressureGradient:
    """$\\partial p^* / \\partial z^* = -4 u_\\text{peak}^* / R^{*2}$."""

    def test_matches_closed_form(self) -> None:
        radius, peak_velocity = 2.0, 3.0
        gradient = mfp.analytical_pressure_gradient(radius, peak_velocity)
        assert gradient == pytest.approx(-4.0 * peak_velocity / radius**2)

    @pytest.mark.parametrize("radius,peak_velocity", [(1.0, 1.0), (0.5, 2.0), (5.0, 0.1)])
    def test_always_non_positive(self, radius: float, peak_velocity: float) -> None:
        """Pressure must fall (or stay flat) along the flow direction."""
        assert mfp.analytical_pressure_gradient(radius, peak_velocity) <= 0.0

    @pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_radius_raises(self, radius: float) -> None:
        with pytest.raises(ValueError, match="radius must be finite and strictly positive"):
            mfp.analytical_pressure_gradient(radius, 1.0)

    @pytest.mark.parametrize("peak_velocity", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_peak_velocity_raises(self, peak_velocity: float) -> None:
        with pytest.raises(ValueError, match="peak_velocity must be finite and strictly positive"):
            mfp.analytical_pressure_gradient(1.0, peak_velocity)


class TestHagenPoiseuillePressure:
    """$p^*(z^*) = p_\\text{ref}^* + \\partial p^*/\\partial z^* \\, (z^* - L^*)$."""

    def test_equals_reference_pressure_at_outlet(self) -> None:
        domain_length = 4.0
        pressure_at_outlet = mfp.hagen_poiseuille_pressure(
            axial_position=torch.tensor([domain_length]), radius=1.3, peak_velocity=1.0,
            domain_length=domain_length, reference_pressure=0.25,
        )
        assert pressure_at_outlet.item() == pytest.approx(0.25)

    def test_slope_matches_analytical_pressure_gradient(self) -> None:
        radius, peak_velocity, domain_length = 1.3, 2.0, 4.0
        axial_positions = torch.tensor([0.0, 1.0, 2.0, domain_length])
        pressure = mfp.hagen_poiseuille_pressure(axial_positions, radius, peak_velocity, domain_length)
        fitted_slope = ((pressure[1:] - pressure[:-1]) / (axial_positions[1:] - axial_positions[:-1]))
        expected_slope = mfp.analytical_pressure_gradient(radius, peak_velocity)
        assert torch.allclose(fitted_slope, torch.full_like(fitted_slope, expected_slope))

    def test_shape_matches_axial_position(self) -> None:
        axial_positions = torch.linspace(0.0, 4.0, 11).unsqueeze(1)
        pressure = mfp.hagen_poiseuille_pressure(axial_positions, radius=1.3, peak_velocity=1.0, domain_length=4.0)
        assert pressure.shape == axial_positions.shape

    def test_invalid_domain_length_raises(self) -> None:
        with pytest.raises(ValueError, match="domain_length must be finite and strictly positive"):
            mfp.hagen_poiseuille_pressure(torch.tensor([0.0]), radius=1.0, peak_velocity=1.0, domain_length=0.0)


class TestHagenPoiseuilleVelocityField:
    """$u_r^* \\equiv 0$, $u_z^*(r^*) = u_\\text{peak}^*(1 - (r^*/R^*)^2)$."""

    def test_axial_velocity_at_axis_equals_peak_velocity(self, domain: mfp.Domain) -> None:
        coordinates = torch.tensor([[0.0, 0.0]])
        field = mfp.hagen_poiseuille_velocity_field(coordinates, domain, peak_velocity=2.5)
        assert field[0, 1].item() == pytest.approx(2.5)
        assert field[0, 0].item() == pytest.approx(0.0)

    def test_axial_velocity_at_wall_is_zero(self, domain: mfp.Domain) -> None:
        coordinates = torch.tensor([[domain.radius, 0.0]])
        field = mfp.hagen_poiseuille_velocity_field(coordinates, domain, peak_velocity=2.5)
        assert field[0, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_radial_velocity_is_identically_zero(self, domain: mfp.Domain) -> None:
        radial_positions = torch.linspace(0.0, domain.radius, 20)
        coordinates = torch.stack([radial_positions, torch.zeros_like(radial_positions)], dim=1)
        field = mfp.hagen_poiseuille_velocity_field(coordinates, domain, peak_velocity=1.0)
        assert torch.allclose(field[:, 0], torch.zeros_like(field[:, 0]))

    def test_pressure_column_matches_hagen_poiseuille_pressure(self, domain: mfp.Domain) -> None:
        axial_positions = torch.linspace(0.0, domain.length, 5)
        coordinates = torch.stack([torch.zeros_like(axial_positions), axial_positions], dim=1)
        field = mfp.hagen_poiseuille_velocity_field(coordinates, domain, peak_velocity=1.0, reference_pressure=0.3)
        expected_pressure = mfp.hagen_poiseuille_pressure(
            axial_positions.unsqueeze(1), domain.radius, 1.0, domain.length, reference_pressure=0.3
        ).squeeze(1)
        assert torch.allclose(field[:, 2], expected_pressure)

    def test_output_shape(self, domain: mfp.Domain) -> None:
        coordinates = torch.rand(7, 2) * domain.radius
        field = mfp.hagen_poiseuille_velocity_field(coordinates, domain, peak_velocity=1.0)
        assert field.shape == (7, 3)

    def test_invalid_coordinates_shape_raises(self, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, 2\)"):
            mfp.hagen_poiseuille_velocity_field(torch.rand(4, 3), domain, peak_velocity=1.0)


class TestFaxenOffsetVelocity:
    """$U_z^\\text{Faxén} - u_z^*(r_0^*) = -\\frac{2}{3}\\frac{a^{*2}}{R^{*2}} u_\\text{peak}^*$."""

    def test_zero_particle_radius_gives_zero_offset(self) -> None:
        assert mfp.faxen_offset_velocity(peak_velocity=1.0, radius=1.3, particle_radius=0.0) == pytest.approx(0.0)

    def test_matches_closed_form(self) -> None:
        peak_velocity, radius, particle_radius = 2.0, 1.3, 0.1
        offset = mfp.faxen_offset_velocity(peak_velocity, radius, particle_radius)
        expected = -(2.0 / 3.0) * (particle_radius**2 / radius**2) * peak_velocity
        assert offset == pytest.approx(expected)

    def test_offset_is_non_positive(self) -> None:
        assert mfp.faxen_offset_velocity(peak_velocity=1.0, radius=1.3, particle_radius=0.2) <= 0.0

    def test_offset_grows_with_particle_radius_squared(self) -> None:
        small = mfp.faxen_offset_velocity(1.0, 1.3, 0.1)
        large = mfp.faxen_offset_velocity(1.0, 1.3, 0.2)
        assert large == pytest.approx(4.0 * small)

    @pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
    def test_invalid_radius_raises(self, radius: float) -> None:
        with pytest.raises(ValueError, match="radius must be finite and strictly positive"):
            mfp.faxen_offset_velocity(1.0, radius, 0.1)

    @pytest.mark.parametrize("particle_radius", [-0.1, float("nan"), float("-inf")])
    def test_invalid_particle_radius_raises(self, particle_radius: float) -> None:
        with pytest.raises(ValueError, match="particle_radius must be finite and non-negative"):
            mfp.faxen_offset_velocity(1.0, 1.3, particle_radius)


class TestAnalyticalTracerStreamline:
    """A point tracer under a uniform field drifts at the constant, local Poiseuille velocity."""

    def test_radial_position_stays_constant(self) -> None:
        radial, _ = mfp.analytical_tracer_streamline(
            initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3, time_grid=np.linspace(0.0, 5.0, 11),
        )
        assert np.allclose(radial, 0.3)

    def test_axial_position_advances_at_the_local_poiseuille_velocity(self) -> None:
        radius, peak_velocity, initial_radius = 1.3, 1.0, 0.3
        time_grid = np.linspace(0.0, 5.0, 11)
        _, axial = mfp.analytical_tracer_streamline(
            initial_position=(initial_radius, 0.2), peak_velocity=peak_velocity, radius=radius, time_grid=time_grid,
        )
        expected_velocity = peak_velocity * (1.0 - (initial_radius / radius) ** 2)
        assert np.allclose(axial, 0.2 + expected_velocity * time_grid)

    def test_centerline_tracer_moves_at_peak_velocity(self) -> None:
        time_grid = np.array([0.0, 1.0, 2.0])
        _, axial = mfp.analytical_tracer_streamline(
            initial_position=(0.0, 0.0), peak_velocity=2.0, radius=1.3, time_grid=time_grid,
        )
        assert np.allclose(axial, 2.0 * time_grid)

    def test_faxen_correction_slows_a_finite_particle(self) -> None:
        """A finite-size tracer must lag its own point-particle limit (a negative offset)."""
        time_grid = np.array([0.0, 1.0])
        _, axial_point = mfp.analytical_tracer_streamline(
            initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3, time_grid=time_grid, particle_radius=0.0,
        )
        _, axial_finite = mfp.analytical_tracer_streamline(
            initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3, time_grid=time_grid, particle_radius=0.1,
        )
        assert axial_finite[-1] < axial_point[-1]

    def test_honors_initial_time_offset(self) -> None:
        _, axial = mfp.analytical_tracer_streamline(
            initial_position=(0.0, 1.0), peak_velocity=1.0, radius=1.3,
            time_grid=np.array([2.0, 3.0]), initial_time=2.0,
        )
        assert np.allclose(axial, [1.0, 2.0])

    def test_initial_radius_outside_domain_raises(self) -> None:
        with pytest.raises(ValueError, match=r"radial component must lie in"):
            mfp.analytical_tracer_streamline(
                initial_position=(2.0, 0.0), peak_velocity=1.0, radius=1.3, time_grid=np.array([0.0, 1.0]),
            )


    def test_negative_particle_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="particle_radius must be finite and non-negative"):
            mfp.analytical_tracer_streamline(
                initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3,
                time_grid=np.array([0.0, 1.0]), particle_radius=-0.1,
            )

    def test_empty_time_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            mfp.analytical_tracer_streamline(
                initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3, time_grid=np.array([]),
            )

    def test_non_1d_time_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            mfp.analytical_tracer_streamline(
                initial_position=(0.3, 0.0), peak_velocity=1.0, radius=1.3,
                time_grid=np.zeros((2, 2)),
            )
