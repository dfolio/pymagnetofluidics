import math

import pytest
import torch

from magnetofluidics_pinn.config import FluidConfig
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.physics.stokes_sphere import (
    haberman_sayre_wall_factor, stokes_drag_reference, stokes_sphere_field, tube_resistance_reference,
)
from magnetofluidics_pinn.verification.force_checks import (
    cross_section_momentum_force, global_momentum_balance, sphere_traction_force, surface_independence_report,
)

A, ZP = 0.25, 4.0


def exact(x):
    return stokes_sphere_field(x, ZP, A, 1.0)


def test_surface_velocity_is_rigid_translation():
    theta = torch.linspace(0.0, math.pi, 50, dtype=torch.float64)
    pts = torch.stack([A * torch.sin(theta), ZP + A * torch.cos(theta)], dim=1)
    out = exact(pts)
    assert torch.allclose(out[:, 0], torch.zeros_like(out[:, 0]), atol=1e-12)
    assert torch.allclose(out[:, 1], torch.ones_like(out[:, 1]), atol=1e-12)


def test_field_satisfies_stokes_equations():
    pts = torch.tensor([[0.4, 4.3], [0.7, 3.2], [0.3, 5.0]], dtype=torch.float64).requires_grad_(True)
    res = stokes_residual(exact, pts, FluidConfig(), residual_form="standard")
    assert res.abs().max().item() < 1e-8


@pytest.mark.parametrize("radius", [A, 1.7 * A])
def test_traction_gives_minus_six_pi_a_u(radius):
    force = sphere_traction_force(exact, ZP, radius, 721)
    assert force == pytest.approx(stokes_drag_reference(A, 1.0), rel=1e-4)


def test_surface_independence_on_exact_field():
    rep = surface_independence_report(exact, ZP, (A, 1.2 * A, 2 * A), 721)
    assert rep.relative_deviation.max() < 1e-4


def test_point_inside_sphere_is_rejected():
    with pytest.raises(ValueError):
        stokes_sphere_field(torch.tensor([[0.0, ZP]], dtype=torch.float64), ZP, A, 1.0)


def test_wall_factor_limits_and_errors():
    assert haberman_sayre_wall_factor(0.0) == 1.0
    assert haberman_sayre_wall_factor(0.25) == pytest.approx(1.979, abs=5e-3)
    with pytest.raises(ValueError):
        haberman_sayre_wall_factor(1.0)


def test_tube_resistance_reference_matches_faxen_limit():
    f_a, f_b = tube_resistance_reference(A, 1.0, 1.0)
    assert f_b < 0.0
    assert -f_a / f_b == pytest.approx(1.0 - 2 * 0.25**2 / 3, rel=1e-12)
    with pytest.raises(ValueError):
        tube_resistance_reference(1.2, 1.0, 1.0)


def test_poiseuille_has_zero_momentum_force():
    from magnetofluidics_pinn.types import Domain
    from magnetofluidics_pinn.verification.analytical import hagen_poiseuille_velocity_field
    dom = Domain(kind="channel", length=10.0, radius=1.0, u_max=1.0)
    field = lambda x: hagen_poiseuille_velocity_field(x, dom, 1.0, 0.0)
    assert abs(cross_section_momentum_force(field, dom.radius, 2.0, 6.0, dtype=torch.float64)) < 1e-6
    # Sphere-free fields have zero traction over any control sphere.

    sphere = sphere_traction_force(field, 4.0, 0.3, 361, dtype=torch.float64)
    print(f'{sphere:.9e} ')
    assert abs(sphere_traction_force(field, 4.0, 0.3, 361, dtype=torch.float64)) < 1e-6


def test_force_check_validation():
    with pytest.raises(ValueError):
        sphere_traction_force(exact, ZP, -1.0)
    with pytest.raises(ValueError):
        surface_independence_report(exact, ZP, ())
    with pytest.raises(ValueError):
        cross_section_momentum_force(exact, 1.0, 3.0, 2.0)
    with pytest.raises(ValueError):
        global_momentum_balance(exact, ZP, A, 1.0, section_offset=0.1)


def test_closed_form_has_exact_algebraic_zeros_at_the_equator():
    """At zeta = 0, u_r and p are forced to exactly 0.0 by the formula itself — not
    approximately, since the arithmetic multiplies by a literal zero. A refactor that
    introduces any dependency on zeta through a different path (e.g. an autodiff route
    that doesn't cancel exactly) will fail this, even if it's close."""
    pts = torch.tensor([[A, ZP]], dtype=torch.float64)
    out = stokes_sphere_field(pts, ZP, A, 1.0)
    assert out[0, 0].item() == 0.0   # u_r
    assert out[0, 2].item() == 0.0   # p
    assert out[0, 1].item() == 1.0   # u_z: exact rigid-body no-slip


def test_rejects_points_strictly_inside_the_sphere():
    center = torch.tensor([[0.0, ZP]], dtype=torch.float64)
    with pytest.raises(ValueError):
        stokes_sphere_field(center, ZP, A, 1.0)