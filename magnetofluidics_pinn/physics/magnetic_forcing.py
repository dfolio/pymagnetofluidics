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
    Under a spatially uniform field, this evaluates to exactly zero, since
    `B` then carries no dependency on position for autograd to differentiate
    through (Abbott, Diller, & Petruska, 2020).

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

    Raises:
    - `ValueError`: If `positions` is not a 2-D tensor, or if
      `magnetic_moment` cannot be broadcast to `positions`'s shape.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must have shape (n_particles, n_dims); got {tuple(positions.shape)}."
        )

    moment = torch.as_tensor(magnetic_moment, dtype=positions.dtype, device=positions.device)
    if moment.ndim == 1:
        moment = moment.unsqueeze(0).expand(positions.shape[0], -1)
    if moment.shape != positions.shape:
        raise ValueError(
            "magnetic_moment must broadcast to positions' shape "
            f"{tuple(positions.shape)}; got {tuple(moment.shape)}."
        )

    # `positions` must be attached to a differentiable graph for the force
    # to be recoverable through autograd. If the caller did not already
    # request gradients, work on a detached clone instead of mutating the
    # tensor handed in, preserving this function's purity.
    positions_for_grad = (
        positions if positions.requires_grad else positions.clone().requires_grad_(True)
    )

    field_sample = field_fn(positions_for_grad)
    potential = torch.sum(moment * field_sample.field, dim=1, keepdim=True)

    if not potential.requires_grad:
        # `field_sample.field` was built entirely from constants (e.g. a
        # uniform field), so `potential` never entered the autograd graph in
        # the first place. `torch.autograd.grad` requires its output to
        # require gradients, so this case must be handled before calling
        # it. Physically, the gradient - and therefore the net dipole
        # force - is exactly zero, consistent with F = grad(m . B) = 0 for
        # a spatially constant B.
        return torch.zeros_like(positions)

    (gradient,) = torch.autograd.grad(
        outputs=potential,
        inputs=positions_for_grad,
        grad_outputs=torch.ones_like(potential),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    if gradient is None:
        # `potential` required gradients overall (e.g., it also depends on
        # another differentiable input) but carried no dependency on
        # `positions_for_grad` specifically: the same zero-force conclusion
        # applies.
        return torch.zeros_like(positions)
    return gradient
