"""Boundary conditions for the viscous flow field.

Each function returns the target value(s) that a boundary-loss term should
compare the network's prediction against, at a given set of boundary
coordinates. None of these functions depend on the network itself, keeping
them pure and reusable across regimes (Stokes or Navier-Stokes).
"""

from __future__ import annotations

import torch

from magnetofluidics_pinn.types import Domain


def inlet_velocity_condition(
        domain: Domain, coordinates: torch.Tensor, peak_velocity: float
) -> torch.Tensor:
    """Compute the prescribed inlet velocity profile at given coordinates.

    Args:
    - `domain`: Vessel geometry the inlet belongs to.
    - `coordinates`: Tensor of shape `(n_points, n_dims)` of inlet points.
    - `peak_velocity`: Dimensionless peak velocity at the channel centerline.

    Returns:
    - Tensor of shape `(n_points, n_dims)` with the target velocity vector at
      each coordinate (e.g., a parabolic Poiseuille-like profile).

    Raises:
    - `ValueError`: If `coordinates` do not have shape `(n_points, 2)`, the
      axisymmetric `(r, z)` convention this package uses, or if
      `domain.radius` is not strictly positive.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (n_points, 2) for the axisymmetric "
            f"(r, z) convention; got {tuple(coordinates.shape)}."
        )
    if domain.radius <= 0.0:
        raise ValueError(f"domain.radius must be strictly positive; got {domain.radius!r}.")
    
    # Classical (Hagen-)Poiseuille profile for axisymmetric pipe flow: purely
    # axial, parabolic in r, maximal on the axis, and zero at the wall.
    radial_position = coordinates[:, 0:1]
    axial_velocity = peak_velocity * (1.0 - (radial_position / domain.radius) ** 2)
    radial_velocity = torch.zeros_like(axial_velocity)
    return torch.cat([radial_velocity, axial_velocity], dim=1)


def outlet_pressure_condition(
        domain: Domain, coordinates: torch.Tensor, reference_pressure: float
) -> torch.Tensor:
    """Compute the prescribed outlet pressure at given coordinates.

    Args:
    - `domain`: Vessel geometry the outlet belongs to.
    - `coordinates`: Tensor of shape `(n_points, n_dims)` of outlet points.
    - `reference_pressure`: Dimensionless reference pressure at the outlet.

    Returns:
    - Tensor of shape `(n_points, 1)` with the target pressure at each
      coordinate.

    Raises:
    - `ValueError`: If `coordinates` is not a 2-D tensor.
    """
    if coordinates.ndim != 2:
        raise ValueError(
            f"coordinates must have shape (n_points, n_dims); got {tuple(coordinates.shape)}."
        )
    del domain  # An outlet's reference pressure does not depend on geometry;
    # `domain` is kept in the signature for symmetry with the other
    # boundary-condition functions and for forward compatibility with a
    # geometry-dependent pressure BC in a later phase.
    return torch.full(
        (coordinates.shape[0], 1),
        float(reference_pressure),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )


def no_slip_condition(domain: Domain, coordinates: torch.Tensor) -> torch.Tensor:
    """Compute the no-slip target velocity at wall coordinates.

    Args:
    - `domain`: Vessel geometry the wall belongs to.
    - `coordinates`: Tensor of shape `(n_points, n_dims)` of wall points.

    Returns:
    - Tensor of shape `(n_points, n_dims)` of zeros, matching the no-slip
      condition at solid boundaries.

    Raises:
    - `ValueError`: If `coordinates` is not a 2-D tensor.
    """
    if coordinates.ndim != 2:
        raise ValueError(
            f"coordinates must have shape (n_points, n_dims); got {tuple(coordinates.shape)}."
        )
    del domain  # The no-slip target is always zero velocity, regardless of
    # the specific wall geometry; kept in the signature for API symmetry.
    return torch.zeros_like(coordinates)
