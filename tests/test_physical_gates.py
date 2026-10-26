"""Tests for `verification.physical_gates` on exact Poiseuille fields."""
import math

import pytest
import torch

from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.verification import physical_gates as pg

DOMAIN = Domain(kind="channel", length=8.0, radius=1.0, u_max=1.0)


class _Poiseuille(torch.nn.Module):
    """Exact field; a parameter gives the module a device and dtype."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, c):
        r, z = c[:, 0:1], c[:, 1:2]
        return torch.cat([torch.zeros_like(r), 1.0 - r.square(), -4.0 * (z - 8.0)], 1)


def test_divergence_probe_exact_field():
    assert pg.divergence_probe(_Poiseuille(), DOMAIN) < 1e-10


def test_pressure_jump_zero_without_particle():
    assert abs(pg.axis_pressure_jump(_Poiseuille(), DOMAIN, 4.0, 1.0)) < 1e-10


def test_pressure_jump_windows_validated():
    with pytest.raises(ValueError):
        pg.axis_pressure_jump(_Poiseuille(), DOMAIN, 4.0, 1.0, offset=3.0, window=2.0)


def test_free_velocity_bounds():
    lo, hi = pg.free_velocity_bounds_check(0.95, 1.0)
    assert lo.passed and hi.passed
    assert not pg.free_velocity_bounds_check(1.48, 1.0)[1].passed
    with pytest.raises(ValueError):
        pg.free_velocity_bounds_check(0.5, 0.0)


def test_generalization_gate_and_tolerance():
    assert not pg.generalization_gate(0.12, 3.0).passed
    assert pg.generalization_gate(0.5, 1.0).passed
    assert math.isclose(pg.reference_tolerance(0.05, 0.1), 0.075)
    with pytest.raises(ValueError):
        pg.reference_tolerance(-1.0, 0.1)