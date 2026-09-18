"""Shared fixtures for the test suite.

Centralizes the small, fast-to-construct objects (a dimensionless domain, a
tiny network, minimal configs) that most tests in this suite need, so each
test module states only what differs from these defaults.
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp


@pytest.fixture
def domain() -> mfp.Domain:
    """A small, already-dimensionless straight-channel domain."""
    return mfp.Domain(kind="channel", length=4.0, radius=1.3, branch_angle=None)


@pytest.fixture
def fluid_config() -> mfp.FluidConfig:
    """A default Stokes-regime fluid configuration."""
    return mfp.FluidConfig()


@pytest.fixture
def field_config() -> mfp.MagneticFieldConfig:
    """A default uniform-field magnetic configuration."""
    return mfp.MagneticFieldConfig()


@pytest.fixture
def raw_network() -> torch.nn.Module:
    """A small, unconstrained (r, z) -> (u_r, u_z, p) network, CPU, fixed seed."""
    torch.manual_seed(0)
    return mfp.build_mlp(n_inputs=2, n_outputs=3, hidden_layers=(8, 8), device="cpu")


@pytest.fixture
def constrained_network(raw_network: torch.nn.Module, domain: mfp.Domain) -> torch.nn.Module:
    """`raw_network` wrapped with the hard axis/wall constraint."""
    return mfp.apply_hard_wall_constraint(raw_network, domain)


def make_poiseuille_network(radius: float, peak_velocity: float):
    """Build a synthetic network implementing the exact analytic Poiseuille field.

    Not a fixture itself (it needs per-test parameters); used by tests that
    need a network whose output is known in closed form, to check
    `physics.conservation` and the loss terms against an exact reference
    rather than only checking that they run.
    """
    
    def network(coordinates: torch.Tensor) -> torch.Tensor:
        radial = coordinates[:, 0:1]
        velocity_r = torch.zeros_like(radial)
        velocity_z = peak_velocity * (1.0 - (radial / radius).square())
        pressure = torch.zeros_like(radial)
        return torch.cat([velocity_r, velocity_z, pressure], dim=1)
    
    return network
