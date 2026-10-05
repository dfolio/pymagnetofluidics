import math

import pandas as pd
import pytest
import torch

from magnetofluidics_pinn.boundary_conditions.magnetic_field_bc import gradient_field
from magnetofluidics_pinn.config import DomainConfig, FluidConfig, MagneticFieldConfig
from magnetofluidics_pinn.physics.magnetic_forcing import (
    axial_dipole_force_si, force_scale, make_dimensionless_dipole_force_fn,
)
from magnetofluidics_pinn.physics.stokes_sphere import oseen_drag_correction
from magnetofluidics_pinn.scaling import compute_scales
from magnetofluidics_pinn.trajectory.mobility import position_independence


def field(x):
    return gradient_field(x, 1.0e-2, (-0.5, 1.0))


def test_axial_force_equals_moment_times_gradient():
    force = axial_dipole_force_si(field, 1.0e-3, 2.0e-13)
    assert force == pytest.approx(2.0e-13 * 1.0, rel=1e-12)
    # A linear field gives a position-independent force.
    assert axial_dipole_force_si(field, 5.0e-3, 2.0e-13) == pytest.approx(force, rel=1e-12)


def test_gradient_field_is_maxwell_consistent_and_validates():
    pts = torch.tensor([[0.2, 0.3], [0.5, 1.0]], dtype=torch.float64).requires_grad_(True)
    out = gradient_field(pts, 1.0e-2, (-0.5, 1.0)).field
    # div B = 2 g_r + g_z = 0, curl B = 0
    assert -0.5 * 2 + 1.0 == 0.0
    jac_bz = torch.autograd.grad(out[:, 1].sum(), pts, retain_graph=True)[0]
    jac_br = torch.autograd.grad(out[:, 0].sum(), pts)[0]
    assert torch.allclose(jac_br[:, 1], torch.zeros(2, dtype=torch.float64))   # dB_r/dz = 0
    assert torch.allclose(jac_bz[:, 0], torch.zeros(2, dtype=torch.float64))   # dB_z/dr = 0
    with pytest.raises(ValueError):
        gradient_field(pts, 1.0e-2, (1.0, 1.0))                                # violates div B = 0
    with pytest.raises(ValueError):
        gradient_field(pts[:, :1], 1.0e-2, (-0.5, 1.0))                        # wrong shape
    with pytest.raises(ValueError):
        gradient_field(pts, float("nan"), (-0.5, 1.0))


def test_dimensionless_force_matches_scale():
    cfg = DomainConfig()
    scales = compute_scales(cfg, MagneticFieldConfig())
    f_nd = make_dimensionless_dipole_force_fn(field, 2.0e-13, scales)(4.0)
    assert f_nd == pytest.approx(2.0e-13 * 1.0 / force_scale(scales), rel=1e-12)
    assert force_scale(scales) == pytest.approx(
        FluidConfig().dynamic_viscosity * scales.velocity * scales.length, rel=1e-12)


def test_oseen_correction():
    assert oseen_drag_correction(0.0) == 1.0
    assert oseen_drag_correction(0.1875) == pytest.approx(1.0703125)
    with pytest.raises(ValueError):
        oseen_drag_correction(-1.0)


def test_position_independence_metric():
    table = pd.DataFrame({"F_A": [8.9, 9.0, 8.95], "F_B": [-9.3, -9.3, -9.3]})
    assert position_independence(table) == pytest.approx(0.1 / 8.95, rel=1e-3)
    with pytest.raises(ValueError):
        position_independence(pd.DataFrame({"F_A": [1.0]}))
    with pytest.raises(ValueError):
        position_independence(pd.DataFrame({"F_A": [1.0, -1.0], "F_B": [-1.0, -1.0]}))
        