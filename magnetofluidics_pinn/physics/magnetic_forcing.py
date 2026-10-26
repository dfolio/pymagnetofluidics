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

from magnetofluidics_pinn.config import MagneticMomentConfig  # NEW
from magnetofluidics_pinn.data_types import MagneticFieldSample
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
    return scales.pressure * scales.length ** 2


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


# ==============================================================================
# NEW: Tensor-Contracted Magnetic Dipole Force from Spatial Field Gradient
# ==============================================================================

def dipole_force_from_gradient(
        magnetic_gradient: torch.Tensor,
        magnetic_moment: torch.Tensor | MagneticMomentConfig,
) -> torch.Tensor:
    r"""Compute the magnetic dipole force from a prescribed magnetic field gradient tensor.

    Evaluates the Kelvin dipole force governing the actuation of a magnetic particle
    in an external magnetic field [@abbott2020magnetic]:
    $$\mathbf{F}_{\mathrm{mag}} = (\mathbf{m} \cdot \nabla)\mathbf{B} = \mathbf{J}_B \, \mathbf{m},$$
    where $\mathbf{J}_B \in \mathbb{R}^{d \times d}$ is the spatial Jacobian matrix of the
    magnetic flux density, defined component-wise as:
    $$J_{ij} = \frac{\partial B_i}{\partial x_j}, \qquad
    F_i = \sum_{j=1}^d J_{ij} \, m_j.$$
    In current-free fluid media ($\nabla \times \mathbf{B} = \mathbf{0}$), the Jacobian
    tensor is symmetric ($J_{ij} = J_{ji}$), ensuring equivalence with the energy-gradient
    formulation $\mathbf{F}_{\mathrm{mag}} = \nabla (\mathbf{m} \cdot \mathbf{B})$.

    This function is pure, fully vectorized on CUDA devices, and supports arbitrary batch
    broadcasting across spatial evaluation grids and multi-particle configurations:
    
    - Unbatched: $(d, d) \times (d,) \to (d,)$
    - Point-batched: $(N, d, d) \times (d,) \to (N, d)$
    - Particle-batched: $(N, d, d) \times (N, d) \to (N, d)$
    - Uniform-gradient swarm: $(d, d) \times (N, d) \to (N, d)$

    Args:
    - `magnetic_gradient`: Spatial Jacobian tensor $\nabla \mathbf{B}$ of the magnetic
      field, of shape `(d, d)` or `(..., d, d)`, in tesla-per-meter ($\mathrm{T / m}$)
      or dimensionless gradient units.
    - `magnetic_moment`: Dipole moment specification, either as a `torch.Tensor` of
      shape `(d,)` or `(..., d)`, or as an instance of `MagneticMomentConfig`. Expressed
      in ampere-square-meter ($\mathrm{A \cdot m^2}$) or dimensionless moment units.

    Returns:
    - Tensor of physical force components matching the batch dimensions of the inputs,
      residing on the same CUDA/CPU device and dtype as `magnetic_gradient`.

    Raises:
    - `TypeError`: If `magnetic_gradient` is not a `torch.Tensor`, or if
      `magnetic_moment` is neither a tensor nor a `MagneticMomentConfig`.
    - `ValueError`: If `magnetic_gradient` does not have at least 2 dimensions, if its
      last two dimensions are not square ($d \times d$), if coordinate dimension
      $d \notin \{2, 3\}$, or if `magnetic_moment` coordinate dimensionality mismatches $d$.
    """
    if not isinstance(magnetic_gradient, torch.Tensor):
        raise TypeError(
            f"magnetic_gradient must be a torch.Tensor; got {type(magnetic_gradient).__name__}."
        )
    
    if magnetic_gradient.ndim < 2:
        raise ValueError(
            "magnetic_gradient must have at least 2 dimensions with shape (..., d, d); "
            f"got shape {tuple(magnetic_gradient.shape)}."
        )
    
    dim_rows, dim_cols = magnetic_gradient.shape[-2], magnetic_gradient.shape[-1]
    if dim_rows != dim_cols:
        raise ValueError(
            "The spatial Jacobian tensor must be square in its last two dimensions; "
            f"got shape (..., {dim_rows}, {dim_cols})."
        )
    
    if dim_rows not in (2, 3):
        raise ValueError(
            f"Coordinate dimension d must be 2 or 3; got d={dim_rows}."
        )
    
    target_device = magnetic_gradient.device
    target_dtype = magnetic_gradient.dtype
    
    # Extract or cast magnetic moment tensor aligned to gradient device and dtype
    if isinstance(magnetic_moment, MagneticMomentConfig):
        moment_tensor = torch.as_tensor(
            magnetic_moment.moment_vector,
            device=target_device,
            dtype=target_dtype,
        )
    elif isinstance(magnetic_moment, torch.Tensor):
        moment_tensor = magnetic_moment.to(device=target_device, dtype=target_dtype)
    else:
        raise TypeError(
            "magnetic_moment must be a torch.Tensor or MagneticMomentConfig; "
            f"got {type(magnetic_moment).__name__}."
        )
    
    if moment_tensor.shape[-1] != dim_rows:
        raise ValueError(
            f"Coordinate dimension mismatch: magnetic_gradient has d={dim_rows}, "
            f"but magnetic_moment has trailing dimension {moment_tensor.shape[-1]}."
        )
    
    # Perform cuBLAS-accelerated tensor contraction J_ij * m_j
    # Handles unbatched, broadcasted, and multi-particle batch shapes
    if magnetic_gradient.ndim == 2 and moment_tensor.ndim == 1:
        # (d, d) @ (d,) -> (d,)
        return torch.mv(magnetic_gradient, moment_tensor)
    
    if magnetic_gradient.ndim == 2 and moment_tensor.ndim > 1:
        # (d, d) broadcasted across (N, d) particles: (N, d) @ (d, d)^T -> (N, d)
        return torch.matmul(moment_tensor, magnetic_gradient.t())
    
    if moment_tensor.ndim == 1:
        # (N, d, d) contracted against a single shared moment (d,):
        # (N, d, d) @ (N, d, 1) -> (N, d, 1) -> (N, d)
        moment_expanded = moment_tensor.unsqueeze(-1)
        return torch.matmul(magnetic_gradient, moment_expanded).squeeze(-1)
    
    # General batched case: (..., d, d) @ (..., d, 1) -> (..., d)
    moment_expanded = moment_tensor.unsqueeze(-1)
    return torch.matmul(magnetic_gradient, moment_expanded).squeeze(-1)
