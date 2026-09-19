"""Unit tests for refactored fluid residuals and hydrodynamic drag modules using pytest."""

from __future__ import annotations

from typing import Callable

import pytest
import torch
import torch.nn as nn

import magnetofluidics_pinn as mfp
from .conftest import make_poiseuille_network
from magnetofluidics_pinn.physics.fluid_residuals import (
    axisymmetric_vector_laplacian,
    compute_flow_derivatives,
    navier_stokes_residual,
    stokes_residual,
)
from magnetofluidics_pinn.physics.hydrodynamic_drag import (
    faxen_corrected_velocity,
    surface_traction_force,
)
from magnetofluidics_pinn.types import SphericalObstacle


# ============================================================================
# EXACT ANALYTICAL GROUND-TRUTH TESTS (VIA POISEUILLE NETWORK)
# ============================================================================


def test_poiseuille_axisymmetric_laplacian_exact(device: torch.device) -> None:
    r"""Validate `axisymmetric_vector_laplacian` against the exact Poiseuille solution.

    For an axisymmetric Hagen-Poiseuille flow profile:
    $$u_r(r, z) = 0, \qquad u_z(r, z) = U_0\left(1 - \frac{r^2}{R^2}\right),$$
    the cylindrical vector Laplacian components are mathematically exact constants:
    $$\nabla_r^2 \mathbf{u} = 0, \qquad
    \nabla_z^2 \mathbf{u} = \frac{\partial^2 u_z}{\partial r^2} + \frac{1}{r}\frac{\partial u_z}{\partial r}
      = -2\frac{U_0}{R^2} - 2\frac{U_0}{R^2} = -\frac{4 U_0}{R^2}.$$

    Args:
    - `device`: Target PyTorch compute device (`"cuda"` or `"cpu"`).
    """
    radius = 1.5
    peak_velocity = 2.0
    poiseuille_net = make_poiseuille_network(radius=radius, peak_velocity=peak_velocity)

    coords = torch.tensor(
        [[0.2, 1.0], [0.6, 2.5], [1.1, 0.5]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    derivs = compute_flow_derivatives(poiseuille_net, coords, compute_second_order=True)

    lap_r, lap_z = axisymmetric_vector_laplacian(
        derivs=derivs, radius=coords[:, 0:1], residual_form="standard"
    )

    expected_lap_r = torch.zeros_like(lap_r)
    expected_lap_z = torch.full_like(lap_z, -4.0 * peak_velocity / (radius**2))

    torch.testing.assert_close(lap_r, expected_lap_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lap_z, expected_lap_z, atol=1e-6, rtol=1e-5)


def test_poiseuille_faxen_exact(device: torch.device) -> None:
    r"""Validate Faxén velocity correction against the closed-form Poiseuille solution.

    Under Faxén's first law [@faxen1922widerstand; @kim2005microhydrodynamics]:
    $$\mathbf{U} = \mathbf{u}_\infty + \frac{a^2}{6}\nabla^2\mathbf{u}_\infty.$$
    Substituting $\nabla^2\mathbf{u}_\infty = \left(0, -4\frac{U_0}{R^2}\right)$ gives:
    $$U_r = 0, \qquad U_z = u_z(r_p) - \frac{2 a^2 U_0}{3 R^2}.$$

    Args:
    - `device`: Target PyTorch compute device.
    """
    radius = 1.2
    peak_velocity = 3.0
    particle_radius = 0.08
    poiseuille_net = make_poiseuille_network(radius=radius, peak_velocity=peak_velocity)

    positions = torch.tensor([[0.3, 1.0], [0.7, 2.0]], dtype=torch.float32, device=device)
    corrected_velocity = faxen_corrected_velocity(
        flow_network=poiseuille_net,
        positions=positions,
        particle_radius=particle_radius,
    )

    base_velocity = poiseuille_net(positions)[:, :2]
    expected_delta_z = -(2.0 * (particle_radius**2) * peak_velocity) / (3.0 * (radius**2))
    expected_velocity = base_velocity.clone()
    expected_velocity[:, 1] += expected_delta_z

    torch.testing.assert_close(corrected_velocity, expected_velocity, atol=1e-6, rtol=1e-5)


def test_poiseuille_continuity_and_momentum_residuals(
    fluid_config: mfp.FluidConfig, device: torch.device
) -> None:
    r"""Verify Poiseuille flow satisfies exact continuity $\nabla \cdot \mathbf{u} = 0$.

    Args:
    - `fluid_config`: Stokes configuration fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    radius = 1.3
    peak_velocity = 1.5
    poiseuille_net = make_poiseuille_network(radius=radius, peak_velocity=peak_velocity)

    coords = torch.tensor(
        [[0.4, 0.5], [0.8, 1.2]], dtype=torch.float32, device=device, requires_grad=True
    )
    res = stokes_residual(poiseuille_net, coords, fluid_config, residual_form="standard")

    # Column 2 is continuity: \nabla \cdot u = 0
    continuity_residual = res[:, 2:3]
    torch.testing.assert_close(
        continuity_residual, torch.zeros_like(continuity_residual), atol=1e-6, rtol=1e-5
    )

    # Column 0 is radial momentum: -dp/dr + lap_r = 0
    momentum_r_residual = res[:, 0:1]
    torch.testing.assert_close(
        momentum_r_residual, torch.zeros_like(momentum_r_residual), atol=1e-6, rtol=1e-5
    )

    # Column 1 is axial momentum: -dp/dz + lap_z = -4 * U0 / R^2 (since p = 0 in poiseuille_net)
    expected_momentum_z = torch.full_like(res[:, 1:2], -4.0 * peak_velocity / (radius**2))
    torch.testing.assert_close(res[:, 1:2], expected_momentum_z, atol=1e-6, rtol=1e-5)


# ============================================================================
# MLP INTEGRATION TESTS (VIA CONFTEST RAW & CONSTRAINED NETWORKS)
# ============================================================================


def test_stokes_residual_with_mlp(
    raw_network: torch.nn.Module, fluid_config: mfp.FluidConfig, device: torch.device
) -> None:
    """Validate `stokes_residual` on a genuine MLP module transferred to the active device.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `fluid_config`: FluidConfig fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords = torch.tensor(
        [[0.3, 1.0], [0.7, 2.0]], dtype=torch.float32, device=device, requires_grad=True
    )

    res = stokes_residual(net, coords, fluid_config, residual_form="standard")
    assert res.shape == (2, 3)
    assert not torch.isnan(res).any()
    assert res.device == coords.device


def test_surface_traction_with_constrained_mlp(
    constrained_network: torch.nn.Module, device: torch.device
) -> None:
    """Validate `surface_traction_force` on the wall-constrained MLP architecture.

    Args:
    - `constrained_network`: Hard-wall constrained network from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = constrained_network.to(device)
    obstacle = SphericalObstacle(radius=0.15, axial_position=2.0)
    force_z = surface_traction_force(net, obstacle, n_quadrature_points=45)

    assert force_z.numel() == 1
    assert not torch.isnan(force_z)


# ============================================================================
# HELPER & OPERATOR TESTS (ORDER SELECTION & RESIDUAL FORMS)
# ============================================================================


def test_compute_flow_derivatives_order_selection(
    raw_network: torch.nn.Module, device: torch.device
) -> None:
    """Verify `compute_second_order=False` bypasses second autodiff pass on MLP.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=True)

    # 1st-order only: 2nd-order fields must be None
    derivs_first = compute_flow_derivatives(net, coords, compute_second_order=False)
    assert derivs_first.d2_velocity_r_dr2 is None
    assert derivs_first.d2_velocity_z_dz2 is None
    assert derivs_first.d_velocity_r_dr is not None

    # Full 2nd-order evaluation
    derivs_full = compute_flow_derivatives(net, coords, compute_second_order=True)
    assert derivs_full.d2_velocity_r_dr2 is not None
    assert derivs_full.d2_velocity_z_dz2 is not None


def test_compute_flow_derivatives_invalid_columns(device: torch.device) -> None:
    """Ensure `compute_flow_derivatives` rejects networks returning != 3 columns.

    Args:
    - `device`: Target PyTorch compute device.
    """
    invalid_net: Callable[[torch.Tensor], torch.Tensor] = lambda x: x[:, :2]
    coords = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=True)

    with pytest.raises(ValueError, match="network must map coordinates to \\(u_r, u_z, p\\)"):
        compute_flow_derivatives(invalid_net, coords)


def test_axisymmetric_vector_laplacian_missing_second_order(
    raw_network: torch.nn.Module, device: torch.device
) -> None:
    """Verify `axisymmetric_vector_laplacian` raises ValueError when 2nd-order grads are missing.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=True)
    derivs = compute_flow_derivatives(net, coords, compute_second_order=False)

    with pytest.raises(ValueError, match="requires second-order derivatives"):
        axisymmetric_vector_laplacian(derivs, coords[:, 0:1], residual_form="standard")


# ============================================================================
# RESIDUAL FORM EQUIVALENCE & EDGE CASES
# ============================================================================


@pytest.mark.parametrize("residual_form", ["standard", "r_weighted"])
def test_stokes_residual_forms_scaling(
    raw_network: torch.nn.Module,
    fluid_config: mfp.FluidConfig,
    device: torch.device,
    residual_form: str,
) -> None:
    r"""Verify mathematical scaling relation between standard and r-weighted residuals.

    Under `"r_weighted"`, the radial momentum residual is pre-multiplied by $r^2$,
    and axial momentum / continuity residuals are pre-multiplied by $r$.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `fluid_config`: FluidConfig fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    - `residual_form`: Singularity strategy under test.
    """
    net = raw_network.to(device)
    coords = torch.tensor(
        [[0.4, 1.0], [0.9, 2.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    res = stokes_residual(net, coords, fluid_config, residual_form=residual_form)  # type: ignore[arg-type]
    assert res.shape == (2, 3)

    if residual_form == "r_weighted":
        res_std = stokes_residual(net, coords, fluid_config, residual_form="standard")
        r = coords[:, 0:1]
        torch.testing.assert_close(res[:, 0:1], res_std[:, 0:1] * r.square())
        torch.testing.assert_close(res[:, 1:2], res_std[:, 1:2] * r)
        torch.testing.assert_close(res[:, 2:3], res_std[:, 2:3] * r)


@pytest.mark.parametrize(
    "coords_val,form,error_match",
    [
        ([[0.0, 1.0]], "standard", "on or across the symmetry axis"),
        ([[-0.2, 1.0]], "standard", "on or across the symmetry axis"),
        ([[-0.2, 1.0]], "r_weighted", "negative radial coordinate"),
    ],
)
def test_stokes_residual_radial_boundary_rejection(
    raw_network: torch.nn.Module,
    fluid_config: mfp.FluidConfig,
    device: torch.device,
    coords_val: list[list[float]],
    form: str,
    error_match: str,
) -> None:
    """Validate axis boundary rejection across residual forms.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `fluid_config`: FluidConfig fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    - `coords_val`: Coordinate list.
    - `form`: Residual form tested.
    - `error_match`: Error regex expected.
    """
    net = raw_network.to(device)
    coords = torch.tensor(coords_val, dtype=torch.float32, device=device, requires_grad=True)
    with pytest.raises(ValueError, match=error_match):
        stokes_residual(net, coords, fluid_config, residual_form=form)  # type: ignore[arg-type]


def test_stokes_residual_accepts_axis_in_r_weighted(
    raw_network: torch.nn.Module, fluid_config: mfp.FluidConfig, device: torch.device
) -> None:
    """Verify r=0 is accepted as a regular point under r_weighted form.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `fluid_config`: FluidConfig fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords_axis = torch.tensor([[0.0, 1.5]], dtype=torch.float32, device=device, requires_grad=True)
    res = stokes_residual(net, coords_axis, fluid_config, residual_form="r_weighted")
    assert res.shape == (1, 3)
    assert not torch.isnan(res).any()


def test_stokes_residual_requires_grad_validation(
    raw_network: torch.nn.Module, fluid_config: mfp.FluidConfig, device: torch.device
) -> None:
    """Verify coordinates without requires_grad=True are rejected.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `fluid_config`: FluidConfig fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=False)
    with pytest.raises(ValueError, match="coordinates must require gradients"):
        stokes_residual(net, coords, fluid_config)


def test_stokes_residual_wrong_regime(
    raw_network: torch.nn.Module, navier_stokes_fluid_config: mfp.FluidConfig, device: torch.device
) -> None:
    """Verify `stokes_residual` rejects non-Stokes configurations.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `navier_stokes_fluid_config`: Navier-Stokes configuration fixture.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    coords = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=True)
    with pytest.raises(ValueError, match="Expected a Stokes fluid configuration"):
        stokes_residual(net, coords, navier_stokes_fluid_config)


# ============================================================================
# NAVIER-STOKES UNSTEADY RESIDUAL TESTS
# ============================================================================


def test_navier_stokes_residual_nominal(
    unsteady_network: torch.nn.Module,
    navier_stokes_fluid_config: mfp.FluidConfig,
    device: torch.device,
) -> None:
    """Validate unsteady convective Navier-Stokes residual computation.

    Args:
    - `unsteady_network`: Unsteady (r, z, t) network fixture.
    - `navier_stokes_fluid_config`: Navier-Stokes configuration fixture.
    - `device`: Target PyTorch compute device.
    """
    net = unsteady_network.to(device)
    coords = torch.tensor(
        [[0.5, 1.0, 0.0], [0.8, 2.0, 0.1]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    res = navier_stokes_residual(net, coords, navier_stokes_fluid_config, residual_form="standard")

    assert res.shape == (2, 3)
    assert not torch.isnan(res).any()
    assert res.device == coords.device


def test_navier_stokes_residual_dimension_rejection(
    unsteady_network: torch.nn.Module,
    navier_stokes_fluid_config: mfp.FluidConfig,
    device: torch.device,
) -> None:
    """Verify `navier_stokes_residual` rejects 2D (r, z) coordinates lacking time t.

    Args:
    - `unsteady_network`: Unsteady network fixture.
    - `navier_stokes_fluid_config`: Navier-Stokes configuration fixture.
    - `device`: Target PyTorch compute device.
    """
    net = unsteady_network.to(device)
    coords_2d = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device, requires_grad=True)

    with pytest.raises(ValueError, match="coordinates must have shape \\(n_points, 3\\)"):
        navier_stokes_residual(net, coords_2d, navier_stokes_fluid_config)


# ============================================================================
# HYDRODYNAMIC DRAG & SURFACE TRACTION TESTS
# ============================================================================


def test_faxen_zero_radius_bypass(raw_network: torch.nn.Module, device: torch.device) -> None:
    """Verify particle_radius=0 returns flow velocity directly without 2nd derivative autodiff.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    """
    net = raw_network.to(device)
    pos = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device=device)
    u_point = faxen_corrected_velocity(net, pos, particle_radius=0.0)

    expected_u = net(pos)[:, :2]
    torch.testing.assert_close(u_point, expected_u)


@pytest.mark.parametrize(
    "radius,pos,error_match",
    [
        (-0.05, [[0.5, 1.0]], "particle_radius must be non-negative"),
        (0.05, [[0.0, 1.0]], "positions on or across the symmetry axis"),
        (0.05, [[0.5, 1.0, 0.0]], "positions must have shape \\(n_points, 2\\)"),
    ],
)
def test_faxen_invalid_inputs(
    raw_network: torch.nn.Module,
    device: torch.device,
    radius: float,
    pos: list[list[float]],
    error_match: str,
) -> None:
    """Validate input rejection in `faxen_corrected_velocity`.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    - `device`: Target PyTorch compute device.
    - `radius`: Particle radius.
    - `pos`: Position coordinates.
    - `error_match`: Error regex expected.
    """
    net = raw_network.to(device)
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match=error_match):
        faxen_corrected_velocity(net, positions, particle_radius=radius)


def test_surface_traction_invalid_quadrature_points(raw_network: torch.nn.Module) -> None:
    """Verify `surface_traction_force` rejects n_quadrature_points < 2.

    Args:
    - `raw_network`: MLP network fixture from `conftest.py`.
    """
    obstacle = SphericalObstacle(radius=0.2, axial_position=1.0)
    with pytest.raises(ValueError, match="n_quadrature_points must be at least 2"):
        surface_traction_force(raw_network, obstacle, n_quadrature_points=1)