"""Magnetic force acting on a magnetized particle.

The force is computed from a *prescribed* magnetic field (see
`magnetofluidics_pinn.boundary_conditions.magnetic_field_bc`), under the
point-dipole approximation. This module has no dependency on the flow
network: it only consumes field values and particle positions.
"""

from __future__ import annotations

from typing import Callable

import torch

from magnetofluidics_pinn.types import FieldSample


def dipole_force(
    field_fn: Callable[[torch.Tensor], FieldSample],
    positions: torch.Tensor,
    magnetic_moment: torch.Tensor,
) -> torch.Tensor:
    """Compute the dipole force on particles at given positions.

    Uses the point-dipole approximation, `F = grad(m . B)`, where the field
    gradient is obtained through automatic differentiation of `field_fn`.

    Args:
    - `field_fn`: Callable returning a
      [`FieldSample`][magnetofluidics_pinn.types.FieldSample] for a batch of
      coordinates, e.g., `uniform_field` or `biot_savart_field`.
    - `positions`: Tensor of shape `(n_particles, n_dims)`, requiring
      gradients, with each particle's position.
    - `magnetic_moment`: Tensor of shape `(n_particles, n_dims)` or `(n_dims,)`
      with each particle's magnetic moment.

    Returns:
    - Tensor of shape `(n_particles, n_dims)` with the force vector acting on
      each particle.
    """
    raise NotImplementedError("Implementation scheduled for Step 2.")
