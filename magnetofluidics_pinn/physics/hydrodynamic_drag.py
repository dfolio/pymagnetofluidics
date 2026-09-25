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

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
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
    """Evaluate the Faxén-corrected ambient velocity at given particle positions.

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
        raise ValueError(
            f"positions must have shape (n_points, 2); got {tuple(positions.shape)}."
        )
    
    positions_for_grad = (
        positions if positions.requires_grad else positions.clone().requires_grad_(True)
    )
    radius_coordinate = positions_for_grad[:, 0:1]
    if torch.any(radius_coordinate <= 0.0):
        raise ValueError(
            "faxen_corrected_velocity received positions on or across the "
            "symmetry axis (r <= 0); the Laplacian correction is singular there."
        )
    
    if particle_radius == 0.0:
        # Zero radius: bypass derivative computations entirely for performance
        output = flow_network(positions_for_grad)
        return output[:, :2]
    
    # CHANGED: Eliminated ~20 lines of duplicate autodiff and Laplacian formulas.
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


# CHANGED: Refactored with compute_flow_derivatives(compute_second_order=False) and GPU-safe quadrature.
def surface_traction_force(
    flow_network: Callable[[torch.Tensor], torch.Tensor],
    particle_config: ParticleConfig,
    particle_state: ParticleState,
    n_quadrature_points: int = 181,
) -> torch.Tensor:
    r"""Axial hydrodynamic force the flow exerts on an embedded spherical obstacle.

    NEW. Unlike
    [`faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity],
    which *approximates* the ambient flow's effect on an undisturbed
    point/finite tracer, this function reads the force directly off a flow
    field that was actually solved *around* the obstacle (see
    [`training.trainer.train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle]):
    it integrates the Cauchy stress tensor's traction vector over the
    sphere's own surface,

    $$
    F_z = \oint_{S} t_z \, dA, \qquad
    t_z = \sigma_{zz} n_z + \sigma_{zr} n_r, \qquad
    \sigma_{zz} = -p + 2 \frac{\partial u_z}{\partial z}, \qquad
    \sigma_{zr} = \frac{\partial u_z}{\partial r} + \frac{\partial u_r}{\partial z},
    $$

    with dimensionless viscosity $1$ throughout (the same viscous pressure
    scaling `physics.fluid_residuals` already assumes, valid whether the
    flow field was obtained from `stokes_residual` or
    `navier_stokes_residual` — the stress-strain constitutive relation for
    a Newtonian fluid does not itself depend on which momentum balance
    produced the velocity field). Parametrizing the sphere's meridian by
    the polar angle $\theta \in [0, \pi]$ (so $r = a\sin\theta$,
    $z = z_p + a\cos\theta$, outward normal $(n_r, n_z) = (\sin\theta,
    \cos\theta)$, surface element $dA = 2\pi r \, a\, d\theta$) turns the
    integral into a plain 1-D quadrature over $\theta$, evaluated here with
    `torch.trapezoid` rather than a fixed-weight sum, so accuracy scales
    with `n_quadrature_points` rather than being capped by a first-order
    rule regardless of resolution.

    **Only $F_z$ is returned.** By the on-axis symmetry
    [`SphericalObstacle`][magnetofluidics_pinn.types.SphericalObstacle]
    requires (see that type's docstring), the radial force integrates to
    exactly zero for any genuinely axisymmetric flow: every contribution
    at azimuthal angle $\phi$ is exactly cancelled by its counterpart at
    $\phi + \pi$. Reporting a numerically near-zero $F_r$ anyway would
    only add quadrature noise, not information.

    This closes the gap
    [`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]'s
    and
    [`trajectory.integrator.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]'s
    own docstrings flag: a directly-computed drag force, from the actual
    perturbed flow, rather than an assumed unit mobility. See
    [`trajectory.two_way_coupling.solve_force_balanced_velocity`][magnetofluidics_pinn.trajectory.two_way_coupling.solve_force_balanced_velocity]
    for where it closes a force balance into a particle velocity
    [@richou2003correction].

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
        
        # CUDA-safe: infer device from network parameters if available, else default to CPU
    device = torch.device("cpu")
    if hasattr(flow_network, "parameters"):
        first_param = next(flow_network.parameters(), None)
        if first_param is not None:
            device = first_param.device
    
    theta = torch.linspace(0.0, torch.pi, n_quadrature_points, device=device)
    radial_coordinate = particle_config.radius * torch.sin(theta)
    axial_coordinate = particle_state.axial_position + particle_config.radius * torch.cos(theta)
    coordinates = torch.stack([radial_coordinate, axial_coordinate], dim=1).requires_grad_(True)
    
    # CHANGED: Reused compute_flow_derivatives with compute_second_order=False.
    derivs = compute_flow_derivatives(flow_network, coordinates, compute_second_order=False)
    
    # Cauchy stress tensor components in axisymmetric coordinates:
    # sigma_zz = -p + 2 * du_z/dz
    # sigma_zr = du_z/dr + du_r/dz
    sigma_zz = -derivs.pressure + 2.0 * derivs.d_velocity_z_dz
    sigma_zr = derivs.d_velocity_z_dr + derivs.d_velocity_r_dz
    
    normal_r = torch.sin(theta).unsqueeze(1)
    normal_z = torch.cos(theta).unsqueeze(1)
    traction_z = (sigma_zz * normal_z + sigma_zr * normal_r).squeeze(1)
    
    # Integrand over theta: dA = 2 * pi * r * a * dtheta
    integrand = traction_z.detach() * 2.0 * torch.pi * radial_coordinate.detach() * particle_config.radius
    return torch.trapezoid(integrand, theta)
