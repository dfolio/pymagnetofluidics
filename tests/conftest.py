"""Shared fixtures for the test suite.

Centralizes the small, fast-to-construct objects (a dimensionless domain, a
tiny network, minimal configs) that most tests in this suite need, so each
test module states only what differs from these defaults.
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide the preferred compute device (CUDA when available, else CPU).

    Returns:
    - `torch.device` configured for the host hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@pytest.fixture
def domain() -> mfp.Domain:
    """A small, already-dimensionless straight-channel domain."""
    return mfp.Domain(kind="channel", length=4.0, radius=1.3, branch_angle=None)


@pytest.fixture
def fluid_config() -> mfp.FluidConfig:
    """A default Stokes-regime fluid configuration."""
    return mfp.FluidConfig()


@pytest.fixture
def domain_config() -> mfp.DomainConfig:
    """A default domain configuration."""
    return mfp.DomainConfig()


@pytest.fixture
def field_config() -> mfp.MagneticFieldConfig:
    """A default uniform-field magnetic configuration."""
    return mfp.MagneticFieldConfig()


@pytest.fixture
def particle_config() -> mfp.ParticleConfig:
    """A small spherical particle, comfortably inside `domain`'s radius."""
    return mfp.ParticleConfig(radius=0.2)


@pytest.fixture
def particle_state() -> mfp.ParticleState:
    """An on-axis particle state, inside `domain`'s axial extent."""
    return mfp.ParticleState(position=torch.tensor([0.0, 2.0]), velocity=torch.zeros(2), time=0.0)


@pytest.fixture
def raw_network() -> torch.nn.Module:
    """A small, unconstrained (r, z) -> (u_r, u_z, p) network, CPU, fixed seed."""
    training_config = mfp.TrainingConfig(device="cpu", random_seed=0)
    return mfp.build_mlp(n_inputs=2, n_outputs=3, hidden_layers=(8, 8), training_config=training_config)


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


@pytest.fixture
def navier_stokes_domain_config() -> mfp.DomainConfig:
    """A default unsteady, convective Navier-Stokes fluid configuration.

    Returns:
    - `mfp.FluidConfig` with `regime="navier_stokes"` and non-zero Reynolds number.
    """
    return mfp.DomainConfig(fluid=mfp.FluidConfig(regime="navier_stokes"))


class _AnalyticalFlowNetwork(torch.nn.Module):
    """A network stand-in returning the exact closed-form Hagen-Poiseuille field.

    Used to check `verification.metrics`'s evaluators against a "network"
    whose true error is known in closed form to be exactly zero (up to
    floating-point noise), rather than only checking that they run.
    Carries one dummy parameter purely so `device_utils.resolve_module_device`/
    `resolve_module_dtype` (which every metrics.py evaluator calls) can infer a
    device and dtype from it, exactly as they would from a real trained network.
    """

    def __init__(self, domain: mfp.Domain, peak_velocity: float, reference_pressure: float = 0.0) -> None:
        super().__init__()
        self.domain = domain
        self.peak_velocity = peak_velocity
        self.reference_pressure = reference_pressure
        self._dummy_parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return mfp.hagen_poiseuille_velocity_field(
            coordinates, self.domain, self.peak_velocity, self.reference_pressure
        )


@pytest.fixture
def analytical_flow_network(domain: mfp.Domain) -> torch.nn.Module:
    """A network whose output is exactly the analytical Hagen-Poiseuille field for `domain`."""
    return _AnalyticalFlowNetwork(domain, peak_velocity=1.0)


@pytest.fixture
def unsteady_network() -> torch.nn.Module:
    """A small, unconstrained (r, z, t) -> (u_r, u_z, p) network on CPU.

    Returns:
    - `torch.nn.Module` mapping 3 coordinate inputs to 3 output channels.
    """
    training_config = mfp.TrainingConfig(device="cpu", random_seed=42)
    return mfp.build_mlp(n_inputs=3, n_outputs=3, hidden_layers=(8, 8), training_config=training_config)
