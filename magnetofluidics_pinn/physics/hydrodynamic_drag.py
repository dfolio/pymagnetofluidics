r"""Hydrodynamic coupling between a finite-size particle and the ambient flow.

`trajectory.integrate_trajectory` treats the magnetic microrobot as a
point — a passive tracer advected by the locally-evaluated flow velocity,
with no explicit particle radius anywhere in the model. That is the
$a \to 0$ (point-particle) limit of the true, finite-size problem, and
until now the package left the limit implicit: no function represented the
particle's presence in the flow, and no parameter recorded its radius.

This module makes that assumption explicit and gives it a well-established,
leading-order finite-size correction: **Faxén's first law**. For a rigid
sphere of radius $a$ translating in a Stokes ambient flow
$u_\infty(\mathbf{x})$ that varies slowly over the particle's scale, the
sphere's velocity is
$$U = u_\infty(\mathbf{x}_p) + \frac{a^2}{6} \nabla^2 u_\infty(\mathbf{x}_p)$$
[@faxen1922widerstand; @kim2005microhydrodynamics, section 7.2, for the
standard modern derivation; @maxey1983equation, for its place in the
general point-particle equation of motion this package's trajectory
integrator otherwise follows]. The correction term is the same viscous
(vector-Laplacian) operator already evaluated inside
[`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
for the momentum equation, here evaluated at the particle's position
instead of at collocation points.

**Regime of validity.** Faxén's law is a leading-order correction, accurate
when the particle is small relative to both the channel radius and the
length scale over which the flow curves (formally, $a / R$ and
$a^2 / R^2$ small). It does *not* model the reverse effect — the particle
disturbing the flow around it — which needs a genuinely two-way-coupled
(excluded-volume) solve and is out of scope for Phase 1's single-tracer
model; see that limitation noted again in
[`integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory].

**Units.** This function is dimensionless throughout: `positions` and the
value `flow_network` returns are already in the package's dimensionless
system, and `particle_radius` must be the corresponding dimensionless
radius $a / L$ (with $L$ the length scale from
[`scaling.compute_scales`][magnetofluidics_pinn.scaling.compute_scales]) —
see
[`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]
for converting a physical (SI, meter) particle radius to that quantity.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype, resolve_device
from magnetofluidics_pinn.physics.fluid_residuals import (
    axisymmetric_vector_laplacian,
    compute_flow_derivatives,
)
from magnetofluidics_pinn.config import ParticleConfig
from magnetofluidics_pinn.types import ParticleState


# CHANGED: Refactored to leverage compute_flow_derivatives and axisymmetric_vector_laplacian.
def faxen_corrected_velocity(
    flow_network: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    particle_radius: float,
) -> torch.Tensor:
    r"""Evaluate Faxén-corrected ambient velocity with regularized on-axis limits.

    Applies Faxén's first law for a rigid sphere in low-Reynolds-number flow [@faxen1922widerstand; @kim2005microhydrodynamics]:
    $$\mathbf{U} = \mathbf{u}_\infty(\mathbf{x}_p) + \frac{a^2}{6} \nabla^2 \mathbf{u}_\infty(\mathbf{x}_p).$$
    CHANGED: On the symmetry axis ($r = 0$), L'Hôpital limits are applied directly to avoid
    singular $1/r$ coordinate division and remove the requirement for off-axis epsilon perturbations.

    Args:
    - `flow_network`: Callable mapping coordinates of shape `(n_points, 2)`
      to `(n_points, 3)`, the trained `(u_r, u_z, p)` flow field.
    - `positions`: Tensor of shape `(n_points, 2)` with `(r, z)` particle
      positions. Must satisfy `r > 0` (off the symmetry axis), since the
      axisymmetric vector Laplacian involves a `1 / r^2` term, exactly as
      in `stokes_residual`.
    - `particle_radius`: The (dimensionless) particle radius `a`. Zero
      recovers the plain point-particle velocity (no correction).

    Returns:
    - Tensor of shape `(n_points, 2)` with the Faxén-corrected `(u_r, u_z)`
      velocity the particle's center would move at.

    Raises:
    - `ValueError`: If `particle_radius` is negative, if `positions` does
      not have shape `(n_points, 2)`, or if any radial coordinate is on or
      across the symmetry axis (`r <= 0`).
    """
    if particle_radius < 0.0:
        raise ValueError(f"particle_radius must be non-negative; got {particle_radius!r}.")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (n_points, 2);"
                         f" got {tuple(positions.shape)}.")
    if torch.any(positions[:, 0:1] < 0.0):
        raise ValueError("positions contain negative radial coordinates.")
    if isinstance(flow_network, nn.Module):
        device = resolve_module_device(flow_network)
        dtype = resolve_module_dtype(flow_network)
    else:
        device = positions.device
        dtype = positions.dtype
    positions_for_grad = (
        positions if positions.requires_grad else positions.clone().requires_grad_(True)
    )
    positions_placed = positions.to(device=device, dtype=dtype)
    positions_for_grad = (
        positions_placed if positions_placed.requires_grad else positions_placed.clone().requires_grad_(True)
    )
    radius_coordinate = positions_for_grad[:, 0:1]
    
    # if torch.any(radius_coordinate <= 0.0):
    #     raise ValueError(
    #         "faxen_corrected_velocity received positions on or across the "
    #         "symmetry axis (r <= 0); the Laplacian correction is singular there."
    #     )
    
    if particle_radius == 0.0:
        # Zero radius: bypass derivative computations entirely for performance
        output = flow_network(positions_for_grad)
        return output[:, :2]
    
    derivs = compute_flow_derivatives(flow_network, positions_for_grad, compute_second_order=True)
    laplacian_r, laplacian_z = axisymmetric_vector_laplacian(
        derivs=derivs,
        radius=radius_coordinate,
        residual_form="standard",
    )
    
    velocity = torch.cat([derivs.velocity_r, derivs.velocity_z], dim=1)
    laplacian_u = torch.cat([laplacian_r, laplacian_z], dim=1)
    correction = (particle_radius ** 2 / 6.0) * laplacian_u
    return velocity + correction


# ==============================================================================
# NEW: Exact Happel-Brenner / Richou Wall-Correction Factor for Poiseuille Flow
def brenner_poiseuille_wall_factor(blockage_ratio: float) -> float:
    r"""Evaluate the exact hydrodynamic drag correction factor for a sphere in Poiseuille flow.

    For a rigid sphere of radius $a$ suspended along the centerline of a circular pipe of radius $R$
    with blockage ratio $\lambda = a/R$, the hydrodynamic drag force relates to the unconfined
    Stokes drag $F_0 = 6 \pi \mu a U_{\mathrm{max}}$ via [@richou2003correction; @happel1983low]:
    $$
    F_z = 6 \pi \mu a U_{\mathrm{max}} \cdot K_P(\lambda),
    $$
    where $K_P(\lambda)$ accounts for both parabolic velocity curvature and wall confinement:
    $$
    K_P(\lambda) = \frac{1 - \frac{2}{3}\lambda^2}{1 - 2.10444 \lambda + 2.08877 \lambda^3 - 0.94813 \lambda^5 - 1.372 \lambda^6 + 3.873 \lambda^8 - 4.192 \lambda^{10}}.
    $$

    Args:
    - `blockage_ratio`: Dimensionless ratio $\lambda = a/R \in [0, 1)$.

    Returns:
    - Dimensionless drag correction factor $K_P(\lambda)$.

    Raises:
    - `ValueError`: If `blockage_ratio` is not in $[0, 1)$.
    """
    if not (0.0 <= blockage_ratio < 1.0):
        raise ValueError(
            f"blockage_ratio must lie in [0, 1); got {blockage_ratio!r}."
        )
    lam = blockage_ratio
    numerator = 1.0 - (2.0 / 3.0) * (lam**2)
    denominator = (
        1.0
        - 2.10444 * lam
        + 2.08877 * (lam**3)
        - 0.94813 * (lam**5)
        - 1.372 * (lam**6)
        + 3.873 * (lam**8)
        - 4.192 * (lam**10)
    )
    if denominator <= 0.0:
        raise ValueError(f"Asymptotic expansion singular at blockage ratio {blockage_ratio!r}.")
    return numerator / denominator


def surface_traction_force(
    flow_network: Callable[[torch.Tensor], torch.Tensor],
    particle_config: ParticleConfig,
    particle_state: ParticleState,
    n_quadrature_points: int = 181,
) -> torch.Tensor:
    r"""Axial hydrodynamic force the flow exerts on an embedded particle.

    Evaluates the surface integral of the Cauchy traction vector over the sphere meridian:
    $$F_z = \oint_S (\sigma_{zz} n_z + \sigma_{zr} n_r) \, dA
        = \int_0^\pi \left( \sigma_{zz}(\theta) \cos\theta + \sigma_{zr}(\theta) \sin\theta \right)
          \cdot 2\pi a^2 \sin\theta \, d\theta.$$

    Args:
    - `flow_network`: A trained (or in-training) network mapping `(r, z)`
      coordinates to `(u_r, u_z, p)` — the *actual*, two-way-coupled flow
      field around `particle_state`, not the undisturbed ambient field. If the
      field is unsteady, evaluate at a fixed time slice first (e.g.
      `lambda rz: network(torch.cat([rz, t_slice], dim=1))`) and pass that
      wrapper here.
    - `particle_state`: The state of the particle for which to compute drag.
    - `n_quadrature_points`: Number of polar-angle quadrature nodes; the
      two poles ($\theta = 0, \pi$) contribute zero regardless of
      resolution, since $r = 0$ there.

    Returns:
    - Scalar tensor: the axial drag force $F_z$ (dimensionless), on the
      same device as `particle_state`'s coordinates are constructed on (CPU by
      default; move the result yourself if `flow_network` lives on
      `"cuda"` and a CPU scalar is inconvenient).

    Raises:
    - `ValueError`: If `n_quadrature_points` is smaller than 2, or if
      `flow_network`'s output does not have exactly 3 columns.
    """
    if n_quadrature_points < 2:
        raise ValueError(
            f"n_quadrature_points must be at least 2 for trapezoidal quadrature; got {n_quadrature_points!r}."
        )
        
    # CHANGED: Dynamically resolve device and dtype from particle state tensor
    if isinstance(flow_network, nn.Module):
        device = resolve_module_device(flow_network)
        dtype = resolve_module_dtype(flow_network)
    else:
        device = resolve_device(None)
        dtype = torch.float32
    
    theta = torch.linspace(0.0, torch.pi, n_quadrature_points, device=device, dtype=dtype)
    a = particle_config.radius
    z_p = particle_state.axial_position
    
    radial_coordinate = a * torch.sin(theta)
    axial_coordinate = z_p + a * torch.cos(theta)
    
    with torch.enable_grad():
        coordinates = torch.stack([radial_coordinate, axial_coordinate], dim=1).requires_grad_(True)
        derivs = compute_flow_derivatives(flow_network, coordinates, compute_second_order=False)
        
        # Cauchy stress components in dimensionless Stokes regime (mu = 1):
        # sigma_zz = -p + 2 * du_z / dz
        # sigma_zr = du_z / dr + du_r / dz
        sigma_zz = -derivs.pressure + 2.0 * derivs.d_velocity_z_dz
        sigma_zr = derivs.d_velocity_z_dr + derivs.d_velocity_r_dz
        
        normal_r = torch.sin(theta).unsqueeze(1)
        normal_z = torch.cos(theta).unsqueeze(1)
        traction_z = (sigma_zz * normal_z + sigma_zr * normal_r).squeeze(1)
        
        # dA = 2 * pi * r * a * dtheta
        integrand = traction_z * (2.0 * torch.pi * a) * radial_coordinate
        return torch.trapezoid(integrand, theta)
