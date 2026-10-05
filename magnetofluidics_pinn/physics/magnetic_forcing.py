"""Magnetic force acting on a magnetized particle.

The force is computed from a *prescribed* magnetic field (see
`magnetofluidics_pinn.boundary_conditions.magnetic_field_bc`), under the
point-dipole approximation. This module has no dependency on the flow
network: it only consumes field values and particle positions.
"""

from __future__ import annotations

import math  # NEW
from typing import Callable

import torch

from magnetofluidics_pinn.types import MagneticFieldSample
from magnetofluidics_pinn.scaling import Scales  # NEW


def dipole_force(
    field_fn: Callable[[torch.Tensor], MagneticFieldSample],
    positions: torch.Tensor,
    magnetic_moment: torch.Tensor,
) -> torch.Tensor:
    """Compute the dipole force on particles at given positions.

    Uses the point-dipole approximation, $F = (m\\nabla) B$, where the field
    gradient is obtained through automatic differentiation of `field_fn`.
    Under a spatially uniform field, this evaluates to exactly zero, since
    `B` then carries no dependency on position for autograd to differentiate
    through [@abbott2020magnetic].

    Args:
    - `field_fn`: Callable returning a
      [`MagneticFieldSample`][magnetofluidics_pinn.types.MagneticFieldSample] for a batch of
      coordinates, e.g., `uniform_field` or `biot_savart_field`.
    - `positions`: Tensor of shape `(n_particles, n_dims)`, requiring
      gradients, with each particle's position.
    - `magnetic_moment`: Tensor of shape `(n_particles, n_dims)` or `(n_dims,)`
      with each particle's magnetic moment.

    Returns:
    - Tensor of shape `(n_particles, n_dims)` with the force vector acting on
      each particle.

    Raises:
    - `ValueError`: If `positions` is not a 2-D tensor, if `magnetic_moment`
      cannot be broadcast to `positions`'s shape, or if `field_fn` returns a
      `MagneticFieldSample` whose `field` does not match `positions`'s shape.
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
    if field_sample.field.shape != positions.shape:
        raise ValueError(
            "field_fn must return a `MagneticFieldSample` whose field has the same shape as "
            f"positions; got field shape {tuple(field_sample.field.shape)} vs positions "
            f"shape {tuple(positions.shape)}."
        )
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


def axial_dipole_force_si(
    field_fn: Callable[[torch.Tensor], MagneticFieldSample],
    axial_position: float,
    axial_moment: float,
) -> float:
    r"""Evaluate the axial dipole force $F_z = \partial_z (m_z B_z)$ on the axis, in newton.

    Args:
    - `field_fn`: Field callable taking **SI** coordinates (meter) and returning tesla.
    - `axial_position`: Axial position, in meter.
    - `axial_moment`: Axial magnetic moment $m_z$, in $\si{\ampere\meter\squared}$.

    Returns:
    - The axial force in newton.

    Raises:
    - `ValueError`: If an argument is not finite.
    """
    if not (math.isfinite(axial_position) and math.isfinite(axial_moment)):
        raise ValueError("axial_position and axial_moment must be finite.")
    position = torch.tensor([[0.0, axial_position]], dtype=torch.float64).requires_grad_(True)
    moment = torch.tensor([0.0, axial_moment], dtype=torch.float64)
    with torch.enable_grad():
        force = dipole_force(field_fn, position, moment)
    return force[0, 1].item()


def force_scale(scales: Scales) -> float:
    r"""Return the viscous force scale $F_c = \mu U_c L_c = P_c L_c^2$, in newton.

    Args:
    - `scales`: Characteristic scales from `compute_scales`.

    Returns:
    - The force scale.
    """
    return scales.pressure * scales.length**2


def make_dimensionless_dipole_force_fn(
    field_fn: Callable[[torch.Tensor], MagneticFieldSample],
    axial_moment: float,
    scales: Scales,
) -> Callable[[float], float]:
    r"""Build a dimensionless `applied_force_fn` for the resistance-pair mobility model.

    The returned callable maps a dimensionless axial position $z^*$ to
    $f^* = F_z(z^* L_c)/F_c$, so it composes directly with
    [`force_balanced_velocity`][magnetofluidics_pinn.trajectory.mobility.force_balanced_velocity].
    This replaces the unit-mobility placeholder, which mixed newton with dimensionless velocity.

    Args:
    - `field_fn`: Field callable in SI units (see `axial_dipole_force_si`).
    - `axial_moment`: Axial magnetic moment, in $\si{\ampere\meter\squared}$.
    - `scales`: Characteristic scales.

    Returns:
    - A callable `z_nondim -> f_nondim`.
    """
    scale = force_scale(scales)
    return lambda z_nondim: axial_dipole_force_si(field_fn, z_nondim * scales.length, axial_moment) / scale
