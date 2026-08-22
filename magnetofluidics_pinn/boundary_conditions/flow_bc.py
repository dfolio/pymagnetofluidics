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
    """
    raise NotImplementedError("Implementation scheduled for Step 2.")


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
    """
    raise NotImplementedError("Implementation scheduled for Step 2.")


def no_slip_condition(domain: Domain, coordinates: torch.Tensor) -> torch.Tensor:
    """Compute the no-slip target velocity at wall coordinates.

    Args:
    - `domain`: Vessel geometry the wall belongs to.
    - `coordinates`: Tensor of shape `(n_points, n_dims)` of wall points.

    Returns:
    - Tensor of shape `(n_points, n_dims)` of zeros, matching the no-slip
      condition at solid boundaries.
    """
    raise NotImplementedError("Implementation scheduled for Step 2.")
