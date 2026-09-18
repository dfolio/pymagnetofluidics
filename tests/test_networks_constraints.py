"""Unit tests for `magnetofluidics_pinn.networks.constraints`.

Focuses on the change in this session: `apply_hard_wall_constraint` now
also hard-enforces the tangential no-slip condition `u_z(R, z) = 0`, in
addition to the radial conditions it already enforced. Verifies both the
new guarantee and that the pre-existing ones (`u_r(0,z) = u_r(R,z) = 0`,
and the wrapper's drop-in calling convention) still hold.
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp


class TestHardWallConstraint:
    def test_u_z_vanishes_at_the_wall(self, constrained_network: torch.nn.Module, domain: mfp.Domain) -> None:
        """NEW guarantee: u_z(R, z) = 0 for every z, exactly (not just approximately)."""
        z = torch.linspace(0.0, domain.length, 25)
        wall_coordinates = torch.stack([torch.full_like(z, domain.radius), z], dim=1)
        output = constrained_network(wall_coordinates)
        axial_velocity_at_wall = output[:, 1]
        assert torch.allclose(axial_velocity_at_wall, torch.zeros_like(axial_velocity_at_wall), atol=1.0e-6)

    def test_u_z_is_generally_nonzero_off_the_wall(
        self, constrained_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        """Sanity check that the constraint only zeroes u_z at r=R, not everywhere."""
        z = torch.linspace(0.1, domain.length - 0.1, 5)
        interior_coordinates = torch.stack([torch.full_like(z, domain.radius / 2.0), z], dim=1)
        output = constrained_network(interior_coordinates)
        assert not torch.allclose(output[:, 1], torch.zeros_like(output[:, 1]), atol=1.0e-6)

    def test_u_r_still_vanishes_at_axis_and_wall(
        self, constrained_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        """Pre-existing guarantee, must not have regressed: u_r(0,z) = u_r(R,z) = 0."""
        z = torch.linspace(0.0, domain.length, 10)
        axis_coordinates = torch.stack([torch.zeros_like(z), z], dim=1)
        wall_coordinates = torch.stack([torch.full_like(z, domain.radius), z], dim=1)

        radial_velocity_at_axis = constrained_network(axis_coordinates)[:, 0]
        radial_velocity_at_wall = constrained_network(wall_coordinates)[:, 0]

        assert torch.allclose(radial_velocity_at_axis, torch.zeros_like(radial_velocity_at_axis), atol=1.0e-6)
        assert torch.allclose(radial_velocity_at_wall, torch.zeros_like(radial_velocity_at_wall), atol=1.0e-6)

    def test_pressure_is_not_constrained_at_the_wall(
        self, constrained_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        """Pressure has no Dirichlet condition at the wall; only axis regularity applies to it."""
        z = torch.linspace(0.1, domain.length - 0.1, 5)
        wall_coordinates = torch.stack([torch.full_like(z, domain.radius), z], dim=1)
        pressure_at_wall = constrained_network(wall_coordinates)[:, 2]
        assert not torch.allclose(pressure_at_wall, torch.zeros_like(pressure_at_wall), atol=1.0e-6)

    def test_output_shape_is_drop_in_compatible(
        self, constrained_network: torch.nn.Module
    ) -> None:
        """The wrapper keeps the same (n_points, 2) -> (n_points, 3) calling convention."""
        coordinates = torch.rand(17, 2)
        output = constrained_network(coordinates)
        assert output.shape == (17, 3)

    def test_wrapped_network_shares_raw_network_parameters(
        self, raw_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        """No parameters are added or frozen by wrapping; it's the same trainable set."""
        wrapped = mfp.apply_hard_wall_constraint(raw_network, domain)
        raw_params = list(raw_network.parameters())
        wrapped_params = list(wrapped.parameters())
        assert len(raw_params) == len(wrapped_params)
        assert all(a is b for a, b in zip(raw_params, wrapped_params))

    def test_raises_on_non_positive_domain_radius(self, raw_network: torch.nn.Module) -> None:
        bad_domain = mfp.Domain(kind="channel", length=1.0, radius=0.0, branch_angle=None)
        with pytest.raises(ValueError, match="domain.radius must be strictly positive"):
            mfp.apply_hard_wall_constraint(raw_network, bad_domain)
