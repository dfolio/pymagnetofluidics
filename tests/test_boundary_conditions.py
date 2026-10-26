"""Tests for `magnetofluidics_pinn.boundary_conditions`.

Both modules were only ever exercised indirectly (through the trainer's batch
builder and the trajectory integrator), always with well-formed input, so
every validation branch was untested. `rigid_body_velocity_condition` in
particular supplies the particle-surface target of the two-way-coupled
problem and had no direct test at all.

The flow boundary conditions are pure functions of a coordinate tensor, so
each test states the expected value in closed form.
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp


class TestInletVelocityCondition:
    """$u_r = 0$, $u_z = u_\\text{peak}\\,(1 - (r/R)^2)$ (Hagen-Poiseuille inlet)."""

    def test_profile_matches_closed_form(self, domain: mfp.Domain) -> None:
        radial = torch.linspace(0.0, domain.radius, 6).unsqueeze(1)
        coordinates = torch.cat([radial, torch.zeros_like(radial)], dim=1)
        velocity = mfp.inlet_velocity_condition(domain, coordinates, peak_velocity=2.0)

        expected_axial = 2.0 * (1.0 - (radial / domain.radius) ** 2)
        assert torch.allclose(velocity[:, 1:2], expected_axial)
        assert torch.allclose(velocity[:, 0:1], torch.zeros_like(radial))

    def test_peak_at_axis_and_zero_at_wall(self, domain: mfp.Domain) -> None:
        coordinates = torch.tensor([[0.0, 0.0], [domain.radius, 0.0]])
        velocity = mfp.inlet_velocity_condition(domain, coordinates, peak_velocity=3.0)
        assert velocity[0, 1].item() == pytest.approx(3.0)
        assert velocity[1, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_output_shape_matches_number_of_points(self, domain: mfp.Domain) -> None:
        assert mfp.inlet_velocity_condition(domain, torch.rand(9, 2), 1.0).shape == (9, 2)

    @pytest.mark.parametrize("bad_shape", [(5,), (5, 3), (5, 1)])
    def test_wrong_coordinate_shape_raises(self, domain: mfp.Domain, bad_shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, 2\)"):
            mfp.inlet_velocity_condition(domain, torch.zeros(bad_shape), 1.0)

    def test_non_positive_domain_radius_raises(self) -> None:
        degenerate = mfp.Domain(kind="channel", length=1.0, radius=0.0)
        with pytest.raises(ValueError, match="domain.radius must be strictly positive"):
            mfp.inlet_velocity_condition(degenerate, torch.zeros(3, 2), 1.0)


class TestOutletPressureCondition:
    def test_returns_constant_reference_pressure_column(self, domain: mfp.Domain) -> None:
        pressure = mfp.outlet_pressure_condition(domain, torch.rand(7, 2), reference_pressure=0.75)
        assert pressure.shape == (7, 1)
        assert torch.allclose(pressure, torch.full((7, 1), 0.75))

    def test_inherits_dtype_of_the_coordinates(self, domain: mfp.Domain) -> None:
        coordinates = torch.rand(4, 2, dtype=torch.float64)
        assert mfp.outlet_pressure_condition(domain, coordinates, 0.0).dtype == torch.float64

    def test_non_2d_coordinates_raise(self, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, n_dims\)"):
            mfp.outlet_pressure_condition(domain, torch.zeros(5), 0.0)


class TestNoSlipCondition:
    def test_returns_zero_velocity_of_matching_shape(self, domain: mfp.Domain) -> None:
        coordinates = torch.rand(6, 2)
        target = mfp.no_slip_condition(domain, coordinates)
        assert target.shape == coordinates.shape
        assert torch.count_nonzero(target) == 0

    def test_non_2d_coordinates_raise(self, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, n_dims\)"):
            mfp.no_slip_condition(domain, torch.zeros(4))


class TestRigidBodyVelocityCondition:
    """Every point of a purely translating rigid body carries the body's own velocity."""

    def test_broadcasts_the_translational_velocity_to_every_point(self) -> None:
        coordinates = torch.rand(5, 2)
        target = mfp.rigid_body_velocity_condition(coordinates, (0.0, 0.3))
        assert target.shape == (5, 2)
        assert torch.allclose(target, torch.tensor([0.0, 0.3]).expand(5, 2))

    def test_zero_velocity_reduces_to_no_slip(self, domain: mfp.Domain) -> None:
        """`no_slip_condition` is documented as the zero-velocity special case."""
        coordinates = torch.rand(5, 2)
        assert torch.equal(
            mfp.rigid_body_velocity_condition(coordinates, (0.0, 0.0)), mfp.no_slip_condition(domain, coordinates)
        )

    def test_result_is_independent_of_where_the_points_are(self) -> None:
        near = mfp.rigid_body_velocity_condition(torch.zeros(3, 2), (0.1, 0.2))
        far = mfp.rigid_body_velocity_condition(torch.full((3, 2), 100.0), (0.1, 0.2))
        assert torch.equal(near, far)

    def test_result_does_not_alias_a_shared_buffer(self) -> None:
        """The target is `.clone()`d, so in-place edits to one call's output must not leak."""
        first = mfp.rigid_body_velocity_condition(torch.zeros(3, 2), (0.0, 1.0))
        first.zero_()
        second = mfp.rigid_body_velocity_condition(torch.zeros(3, 2), (0.0, 1.0))
        assert second[0, 1].item() == pytest.approx(1.0)

    def test_inherits_dtype_of_the_coordinates(self) -> None:
        target = mfp.rigid_body_velocity_condition(torch.zeros(2, 2, dtype=torch.float64), (0.0, 1.0))
        assert target.dtype == torch.float64

    def test_non_2d_coordinates_raise(self) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, n_dims\)"):
            mfp.rigid_body_velocity_condition(torch.zeros(4), (0.0, 1.0))

    def test_velocity_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="translational_velocity must have as many components"):
            mfp.rigid_body_velocity_condition(torch.zeros(4, 3), (0.0, 1.0))


class TestUniformField:
    def test_field_is_magnitude_times_orientation_at_every_point(self) -> None:
        coordinates = torch.rand(6, 2)
        sample = mfp.uniform_field(coordinates, magnitude=0.02, orientation=(0.6, 0.8))
        expected = torch.tensor([0.012, 0.016]).expand(6, 2)
        assert torch.allclose(sample.field, expected)

    def test_returns_the_coordinates_it_was_given(self) -> None:
        coordinates = torch.rand(3, 2)
        assert mfp.uniform_field(coordinates, 1.0, (1.0, 0.0)).coordinates is coordinates

    def test_field_does_not_depend_on_position(self) -> None:
        sample = mfp.uniform_field(torch.tensor([[0.0, 0.0], [5.0, -3.0]]), 1.0, (1.0, 0.0))
        assert torch.equal(sample.field[0], sample.field[1])

    def test_dtype_follows_the_coordinates(self) -> None:
        sample = mfp.uniform_field(torch.rand(2, 2, dtype=torch.float64), 1.0, (1.0, 0.0))
        assert sample.field.dtype == torch.float64

    def test_field_rows_do_not_share_memory(self) -> None:
        """`expand(...).clone()` must give a writable, non-aliased tensor."""
        sample = mfp.uniform_field(torch.rand(3, 2), 1.0, (1.0, 0.0))
        sample.field[0, 0] = 42.0
        assert sample.field[1, 0].item() == pytest.approx(1.0)

    def test_non_2d_coordinates_raise(self) -> None:
        with pytest.raises(ValueError, match=r"coordinates must have shape \(n_points, n_dims\)"):
            mfp.uniform_field(torch.zeros(4), 1.0, (1.0, 0.0))

    def test_orientation_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="orientation must have as many components"):
            mfp.uniform_field(torch.zeros(4, 3), 1.0, (1.0, 0.0))

    @pytest.mark.parametrize("bad_orientation", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.0)])
    def test_non_unit_orientation_raises(self, bad_orientation: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="orientation must be a unit vector"):
            mfp.uniform_field(torch.zeros(4, 2), 1.0, bad_orientation)


class TestNotYetImplementedFields:
    """Documented Phase-2 stubs: they must fail loudly, not return a wrong field."""

    def test_gradient_field_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            mfp.gradient_field(torch.zeros(2, 2), magnitude=1.0, gradient=(0.0, 1.0))

    def test_biot_savart_field_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            mfp.biot_savart_field(torch.zeros(2, 2), torch.ones(3), torch.zeros(3, 2))
