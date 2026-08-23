"""Physics residuals of the viscous flow equations.

Each residual function takes the network (mapping coordinates to velocity
and pressure) and a set of coordinates, and returns the pointwise PDE
residual through automatic differentiation. A residual of zero everywhere
means the network's output satisfies the governing equations exactly.

Phase 1 targets the axisymmetric, dimensionless Stokes (creeping-flow)
equations in cylindrical coordinates `(r, z)`, with velocity components
`(u_r, u_z)` and pressure `p`; see e.g. Happel & Brenner (1983), "Low
Reynolds Number Hydrodynamics", or Leal (2007), "Advanced Transport
Phenomena", for the classical derivation. With the viscous pressure scale
used by `scaling.compute_scales`, the dimensionless viscosity is exactly 1,
so no viscosity factor appears below.

"""

from __future__ import annotations

from typing import Callable

import torch

from magnetofluidics_pinn.config import FluidConfig


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


def stokes_residual(
        network: Callable[[torch.Tensor], torch.Tensor],
        coordinates: torch.Tensor,
        fluid_config: FluidConfig,
) -> torch.Tensor:
    """Evaluate the incompressible Stokes-flow residual.

    Enforces continuity, `div(u) = 0`, and the steady momentum balance,
    `-grad(p) + laplacian(u) = 0`, in dimensionless form, at each coordinate.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, n_dims)`
      to a tensor of shape `(n_points, n_dims + 1)`, the last column being
      pressure and the preceding columns being velocity components.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients, sampled inside the domain.
    - `fluid_config`: Fluid configuration; must have `regime == "stokes"`.

    Returns:
    - Tensor of shape `(n_points, n_dims + 1)` with the continuity residual
      in the last column and the momentum residuals in the preceding
      columns.

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"stokes"`, if
      `coordinates` does not have shape `(n_points, 2)` (the axisymmetric
      `(r, z)` convention this function implements), if `coordinates` does
      not require gradients, if any radial coordinate is on or across the
      symmetry axis (`r <= 0`, where the residual is singular), or if
      `network`'s output does not have exactly 3 columns (`u_r`, `u_z`, `p`).
    """
    if fluid_config.regime != "stokes":
        raise ValueError(
            f"Expected a Stokes fluid configuration, got regime={fluid_config.regime!r}."
        )
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (n_points, 2) for the axisymmetric "
            f"(r, z) formulation; got {tuple(coordinates.shape)}."
        )
    if not coordinates.requires_grad:
        raise ValueError(
            "coordinates must require gradients (call `.requires_grad_(True)`) "
            "so that the residual can be evaluated through automatic differentiation."
        )
    radius = coordinates[:, 0:1]
    if torch.any(radius <= 0.0):
        raise ValueError(
            "stokes_residual received coordinates on or across the symmetry "
            "axis (r <= 0); exclude the axis when sampling interior points."
        )
    
    output = network(coordinates)
    if output.shape[1] != 3:
        raise ValueError(
            f"network must map coordinates to (u_r, u_z, p); got {output.shape[1]} output columns."
        )
    velocity_r = output[:, 0:1]
    velocity_z = output[:, 1:2]
    pressure = output[:, 2:3]
    
    grad_velocity_r = _gradient(velocity_r, coordinates)
    grad_velocity_z = _gradient(velocity_z, coordinates)
    grad_pressure = _gradient(pressure, coordinates)
    
    d_velocity_r_dr = grad_velocity_r[:, 0:1]
    d_velocity_r_dz = grad_velocity_r[:, 1:2]
    d_velocity_z_dr = grad_velocity_z[:, 0:1]
    d_velocity_z_dz = grad_velocity_z[:, 1:2]
    dp_dr = grad_pressure[:, 0:1]
    dp_dz = grad_pressure[:, 1:2]
    
    d2_velocity_r_dr2 = _gradient(d_velocity_r_dr, coordinates)[:, 0:1]
    d2_velocity_r_dz2 = _gradient(d_velocity_r_dz, coordinates)[:, 1:2]
    d2_velocity_z_dr2 = _gradient(d_velocity_z_dr, coordinates)[:, 0:1]
    d2_velocity_z_dz2 = _gradient(d_velocity_z_dz, coordinates)[:, 1:2]
    
    # Axisymmetric continuity: (1/r) d(r u_r)/dr + d(u_z)/dz
    #                         = d(u_r)/dr + u_r/r + d(u_z)/dz.
    continuity_residual = d_velocity_r_dr + velocity_r / radius + d_velocity_z_dz
    
    # r-momentum: the Laplacian of an axisymmetric vector field's radial
    # component carries an extra -u_r/r**2 term relative to a scalar
    # Laplacian.
    momentum_r_residual = (
            -dp_dr
            + d2_velocity_r_dr2
            + d_velocity_r_dr / radius
            - velocity_r / radius ** 2
            + d2_velocity_r_dz2
    )
    
    # z-momentum: the axial component behaves like a scalar Laplacian.
    momentum_z_residual = -dp_dz + d2_velocity_z_dr2 + d_velocity_z_dr / radius + d2_velocity_z_dz2
    
    return torch.cat([momentum_r_residual, momentum_z_residual, continuity_residual], dim=1)


def navier_stokes_residual(
        network: Callable[[torch.Tensor], torch.Tensor],
        coordinates: torch.Tensor,
        fluid_config: FluidConfig,
) -> torch.Tensor:
    """Evaluate the incompressible, unsteady Navier-Stokes residual.

    Enforces continuity and the momentum balance including the convective and
    unsteady terms, in dimensionless form, at each coordinate.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, n_dims)`
      (including time) to a tensor of shape `(n_points, n_dims + 1)`, the
      last column being pressure and the preceding columns being velocity
      components.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients, including a time coordinate.
    - `fluid_config`: Fluid configuration; must have
      `regime == "navier_stokes"`.

    Returns:
    - Tensor of shape `(n_points, n_dims + 1)` with the continuity residual
      in the last column and the momentum residuals in the preceding
      columns.

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"navier_stokes"`.
    """
    if fluid_config.regime != "navier_stokes":
        raise ValueError(
            "Expected a Navier-Stokes fluid configuration, "
            f"got regime={fluid_config.regime!r}."
        )
    raise NotImplementedError("Implementation scheduled for a later roadmap phase.")
