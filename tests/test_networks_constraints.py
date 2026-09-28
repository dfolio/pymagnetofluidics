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


class TestBaseProfileFromMaximumVelocity:
    """`Domain.u_max > 0` adds a parabolic base profile on top of the network's correction."""

    @staticmethod
    def _zero_raw_network() -> torch.nn.Module:
        class ZeroRaw(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.zeros(coordinates.shape[0], 3) + 0.0 * self._p

        return ZeroRaw()

    def test_no_base_profile_when_u_max_is_unknown(self) -> None:
        domain = mfp.Domain(kind="channel", length=4.0, radius=1.0, u_max=0.0)
        network = mfp.apply_hard_wall_constraint(self._zero_raw_network(), domain)
        assert network(torch.tensor([[0.0, 1.0]]))[0, 1].item() == pytest.approx(0.0)

    def test_base_profile_is_parabolic_and_vanishes_at_the_wall(self) -> None:
        domain = mfp.Domain(kind="channel", length=4.0, radius=1.0, u_max=1.0)
        network = mfp.apply_hard_wall_constraint(self._zero_raw_network(), domain)
        radial = torch.linspace(0.0, 1.0, 7)
        profile = network(torch.stack([radial, torch.ones_like(radial)], dim=1))[:, 1]

        assert profile[-1].item() == pytest.approx(0.0, abs=1e-6)
        # Parabolic: u_z(r) / u_z(0) = 1 - (r/R)^2, whatever the peak is.
        assert torch.allclose(profile / profile[0], 1.0 - radial**2, atol=1e-5)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The base profile is built as 2 * u_max * (1 - (r/R)^2), so its centerline value is "
            "2 * u_max, but Domain.u_max is documented as the *maximum* inlet velocity and the trainer "
            "imposes a unit inlet peak (training.trainer._DIMENSIONLESS_PEAK_INLET_VELOCITY = 1.0). The "
            "factor 2 is only right if u_max means the mean velocity (peak = 2 * mean for Poiseuille "
            "flow). Latent today, because build_channel_domain never forwards u_max - see "
            "test_geometry.py's matching xfail. Flagged, not fixed."
        ),
    )
    def test_centerline_value_equals_the_documented_maximum_velocity(self) -> None:
        domain = mfp.Domain(kind="channel", length=4.0, radius=1.0, u_max=1.0)
        network = mfp.apply_hard_wall_constraint(self._zero_raw_network(), domain)
        assert network(torch.tensor([[0.0, 1.0]]))[0, 1].item() == pytest.approx(domain.u_max)
