"""Shared automatic-differentiation helpers for `magnetofluidics_pinn`.

Both `physics.fluid_residuals` (the Stokes momentum/continuity residual)
and `physics.hydrodynamic_drag` (the Faxén-law finite-size correction) need
the same operation: differentiate a scalar field, evaluated by a network,
with respect to its input coordinates, keeping the computation graph alive
for a further round of differentiation (a second derivative, for the
viscous Laplacian each of them needs). This module centralizes that
operation in one place instead of two independently drifting copies.
"""

from __future__ import annotations

import torch


def scalar_field_gradient(output: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
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