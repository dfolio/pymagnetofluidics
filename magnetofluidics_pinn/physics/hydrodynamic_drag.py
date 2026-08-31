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

$$ U = u_\infty(\mathbf{x}_p) + \frac{a^2}{6} \nabla^2 u_\infty(\mathbf{x}_p) $$

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


def _gradient(output: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Differentiate a scalar field with respect to its input coordinates.

    Args:
    - `output`: Tensor of shape `(n_points, 1)`, a scalar field evaluated at
      `coordinates`.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients.

    Returns:
    - Tensor of shape `(n_points, n_dims)` with `d(output) / d(coordinates)`.
    """
    (gradient,) = torch.autograd.grad(
        outputs=output,
        inputs=coordinates,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )
    return gradient


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

    output = flow_network(positions_for_grad)
    velocity_r, velocity_z = output[:, 0:1], output[:, 1:2]

    if particle_radius == 0.0:
        # No correction requested: skip the (otherwise harmless, but
        # wasted) second-derivative pass entirely.
        return torch.cat([velocity_r, velocity_z], dim=1)

    grad_velocity_r = _gradient(velocity_r, positions_for_grad)
    grad_velocity_z = _gradient(velocity_z, positions_for_grad)
    d_vr_dr, d_vr_dz = grad_velocity_r[:, 0:1], grad_velocity_r[:, 1:2]
    d_vz_dr, d_vz_dz = grad_velocity_z[:, 0:1], grad_velocity_z[:, 1:2]

    d2_vr_dr2 = _gradient(d_vr_dr, positions_for_grad)[:, 0:1]
    d2_vr_dz2 = _gradient(d_vr_dz, positions_for_grad)[:, 1:2]
    d2_vz_dr2 = _gradient(d_vz_dr, positions_for_grad)[:, 0:1]
    d2_vz_dz2 = _gradient(d_vz_dz, positions_for_grad)[:, 1:2]

    # Same axisymmetric vector-Laplacian operator as the viscous term in
    # `stokes_residual`'s momentum equations, evaluated here at the
    # particle's position rather than at collocation points.
    laplacian_vr = d2_vr_dr2 + d_vr_dr / radius_coordinate - velocity_r / radius_coordinate**2 + d2_vr_dz2
    laplacian_vz = d2_vz_dr2 + d_vz_dr / radius_coordinate + d2_vz_dz2

    correction = (particle_radius**2 / 6.0) * torch.cat([laplacian_vr, laplacian_vz], dim=1)
    return torch.cat([velocity_r, velocity_z], dim=1) + correction
