"""Shared immutable data structures for `magnetofluidics_pinn`.

This module centralizes the plain data containers exchanged between the
package's modules (geometry, sampling, physics, training, trajectory). Every
structure is an immutable `dataclass` (`frozen=True`), so that functions
consuming or producing them remain pure: no module mutates a structure
in place, it always returns a new one.

Keeping these types in a single, dependency-free module avoids circular
imports between the higher-level modules that need to share data shapes
without depending on each other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Domain:
    """Geometric description of the fluidic vessel.

    A `Domain` built by `geometry.build_channel_domain` or
    `geometry.build_bifurcation_domain` holds physical (SI, meter) values.
    Solvers never consume that physical `Domain` directly: call
    [`scaling.nondimensionalize_domain`]
    [magnetofluidics_pinn.scaling.nondimensionalize_domain] first to obtain
    the dimensionless counterpart that `sampling` and `physics` expect, and
    [`scaling.redimensionalize_domain`]
    [magnetofluidics_pinn.scaling.redimensionalize_domain] to convert back
    for plotting.

    Args:
    - `kind`: Either `"channel"` for a straight vessel or `"bifurcation"` for
      a branched vessel.
    - `length`: Vessel length along the axial coordinate, in `meter` when
      physical, dimensionless when normalized (see class docstring).
    - `radius`: Vessel radius (or half-width) along the radial coordinate, in
      `meter` when physical, dimensionless when normalized.
    - `branch_angle`: Branch half-angle in radians, used only when
      `kind == "bifurcation"`; `None` otherwise.
    """

    kind: Literal["channel", "bifurcation"]
    length: float
    radius: float
    branch_angle: float | None = None


@dataclass(frozen=True)
class CollocationPoints:
    """Sampled points used to evaluate physics residuals during training.

    All tensors are dimensionless: `sampling.sample_collocation_points`
    always draws from an already-nondimensionalized `Domain`, so the
    network, autograd, and PDE residuals never see raw SI magnitudes.

    Args:
    - `interior`: Tensor of shape `(n_interior, n_dims)` sampled inside the
      domain, used to enforce the PDE residual.
    - `boundary`: Tensor of shape `(n_boundary, n_dims)` sampled on the domain
      boundary, used to enforce boundary conditions.
    - `initial`: Optional tensor of shape `(n_initial, n_dims)` sampled at the
      initial time, used only for unsteady (Navier-Stokes) problems.
    """

    interior: torch.Tensor
    boundary: torch.Tensor
    initial: torch.Tensor | None = None


@dataclass(frozen=True)
class FieldSample:
    """Prescribed magnetic field evaluated at a set of coordinates.

    Both `coordinates` and `field` are dimensionless here: field-generating
    functions in `boundary_conditions.magnetic_field_bc` accept physical (SI)
    parameters but return values already normalized by
    `scaling.compute_scales`.

    Args:
    - `coordinates`: Tensor of shape `(n_points, n_dims)` where the field is
      evaluated.
    - `field`: Tensor of shape `(n_points, n_dims)` holding the magnetic field
      vector at each coordinate.
    """

    coordinates: torch.Tensor
    field: torch.Tensor
    
    def __post_init__(self) -> None:
        if self.field.shape != self.coordinates.shape:
            raise ValueError(
                "FieldSample.field must have the same shape as FieldSample.coordinates "
                f"(the field is evaluated at, and must match the dimensionality of, "
                f"those coordinates); got field shape {tuple(self.field.shape)} vs "
                f"coordinates shape {tuple(self.coordinates.shape)}."
            )


@dataclass(frozen=True)
class ParticleState:
    """Instantaneous kinematic state of a single magnetic particle.

    Args:
    - `position`: Tensor of shape `(n_dims,)` giving the particle position.
    - `velocity`: Tensor of shape `(n_dims,)` giving the particle velocity.
    - `time`: Scalar simulation time associated with this state.
    
    Raises:
    - `ValueError`: If `velocity` does not have the same shape as `position`.
    """
    
    position: torch.Tensor
    velocity: torch.Tensor
    time: float
    
    def __post_init__(self) -> None:
        if self.velocity.shape != self.position.shape:
            raise ValueError(
                "ParticleState.velocity must have the same shape as ParticleState.position; "
                f"got velocity shape {tuple(self.velocity.shape)} vs position shape "
                f"{tuple(self.position.shape)}."
            )
