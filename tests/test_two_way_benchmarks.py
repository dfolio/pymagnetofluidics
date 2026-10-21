"""Tests for the Stokes-sphere benchmark, shell invariance and the linear force balance."""

import math

import pytest
import torch

from magnetofluidics_pinn.config import ParticleConfig, TwoWayCouplingConfig
from magnetofluidics_pinn.physics.hydrodynamic_drag import surface_traction_force
from magnetofluidics_pinn.trajectory.two_way_coupling import affine_force_balance
from magnetofluidics_pinn.types import ParticleState
from magnetofluidics_pinn.verification.benchmarks import (
    haberman_sayre_factor, reference_resistances, shell_force_profile, shell_force_spread, stokes_sphere_field_polar,
)

A, ZC, U = 0.25, 4.0, 1.0


def _state() -> ParticleState:
    return ParticleState(position=torch.tensor([0.0, ZC]), velocity=torch.zeros(2), time=0.0)


def _field(coordinates: torch.Tensor) -> torch.Tensor:
    return stokes_sphere_field_polar(coordinates, A, ZC, U)


@pytest.mark.parametrize("n_nodes", [181, 361])
def test_traction_recovers_stokes_law(n_nodes):
    force = surface_traction_force(_field, ParticleConfig(kind="spherical", radius=A), _state(), n_nodes).item()
    assert force == pytest.approx(-6.0 * math.pi * A * U, rel=1.0e-4)


def test_shell_force_is_radius_independent_for_exact_field():
    profile = shell_force_profile(_field, ParticleConfig(kind="spherical", radius=A), _state(),
                                  shell_factors=(1.0, 1.5, 2.0, 3.0))
    assert shell_force_spread(profile) < 1.0e-3


def test_field_satisfies_rigid_body_condition_on_surface():
    theta = torch.linspace(0.0, math.pi, 25)
    points = torch.stack([A * torch.sin(theta), ZC + A * torch.cos(theta)], dim=1)
    velocity = _field(points)[:, :2]
    assert torch.allclose(velocity[:, 0], torch.zeros(25), atol=1.0e-5)
    assert torch.allclose(velocity[:, 1], torch.full((25,), U), atol=1.0e-5)


def test_shell_spread_detects_a_wrong_field():
    biased = lambda c: _field(c) * (1.0 + 0.5 * torch.exp(-((c[:, 0:1].square() + (c[:, 1:2] - ZC).square()).sqrt() - A) / 0.1))
    profile = shell_force_profile(biased, ParticleConfig(kind="spherical", radius=A), _state(), (1.0, 1.5, 2.0))
    assert shell_force_spread(profile) > 1.0e-2


def test_reference_resistances_are_consistent():
    ref = reference_resistances(0.25, A)
    assert ref["force_free_velocity"] == pytest.approx(-ref["force_fixed"] / ref["force_unit_translation"])
    assert ref["force_free_velocity"] == pytest.approx(1.0 - (2.0 / 3.0) * 0.25 ** 2, abs=2.0e-3)
    assert haberman_sayre_factor(0.0) == pytest.approx(1.0)


def test_force_balance_is_linear_and_ordered():
    f0, f1 = 8.94, -9.33
    passive, opposing, assisting = (affine_force_balance(f0, f1, f) for f in (0.0, -0.3, 0.3))
    assert opposing < passive < assisting
    assert (assisting - opposing) == pytest.approx(0.6 / -f1)


@pytest.mark.parametrize("args", [(1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (float("nan"), -1.0, 0.0)])
def test_force_balance_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        affine_force_balance(*args)


@pytest.mark.parametrize("field", ["ambient_peak_velocity", "core_strain_weight"])
def test_coupling_config_rejects_negative_values(field):
    with pytest.raises(ValueError):
        TwoWayCouplingConfig(particle_config=ParticleConfig(kind="spherical", radius=A), **{field: -1.0})
