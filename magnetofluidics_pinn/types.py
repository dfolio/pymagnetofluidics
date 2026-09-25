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

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Domain:
    """Dimensionless vessel geometry used by solvers and sampling.

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
    - `u_max`: Maximum axial velocity at the inlet, in `meter/second` when
       physical, dimensionless when normalized; `0.0` if unknown.
    - `branch_angle`: Branch half-angle in radians, used only when
      `kind == "bifurcation"`; `None` otherwise.
    """

    kind: Literal["channel", "bifurcation"]
    length: float
    radius: float
    u_max: float = 0.0
    branch_angle: float | None = None


@dataclass(frozen=True)
class CollocationPoints:
    """Sampled dimensionless points used to evaluate physics residuals during training.

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
    - `particle_surface`: NEW. Optional tensor of shape `(n_obstacle, n_dims)`
      sampled on the surface of an embedded
      [`SphericalObstacle`][magnetofluidics_pinn.types.SphericalObstacle], as
      returned by
      [`sampling.sample_collocation_points_with_obstacle`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points_with_obstacle].
      `None` for every domain without an embedded obstacle — every existing
      caller that never set this field keeps working unchanged.
    """

    interior: torch.Tensor
    boundary: torch.Tensor
    initial: torch.Tensor | None = None
    particle_surface: torch.Tensor | None = None  # NEW

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32,
           non_blocking: bool = True) -> CollocationPoints:
        """Transfer all contained tensors to target device and dtype without mutation."""
        return CollocationPoints(
            interior=self.interior.to(device=device, dtype=dtype, non_blocking=non_blocking),
            boundary=self.boundary.to(device=device, dtype=dtype, non_blocking=non_blocking),
            initial=self.initial.to(device=device, dtype=dtype,
                                    non_blocking=non_blocking) if self.initial is not None else None,
            particle_surface=self.particle_surface.to(device=device, dtype=dtype,
                                                      non_blocking=non_blocking) if self.particle_surface is not None else None,
        )


@dataclass(frozen=True)
class MagneticFieldSample:
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
                "MagneticFieldSample.field must have the same shape as MagneticFieldSample.coordinates "
                f"(the field is evaluated at, and must match the dimensionality of, "
                f"those coordinates); got field shape {tuple(self.field.shape)} vs "
                f"coordinates shape {tuple(self.coordinates.shape)}."
            )
            
    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32, non_blocking: bool = True) -> MagneticFieldSample:
        return MagneticFieldSample(
            coordinates=self.coordinates.to(device=device, dtype=dtype, non_blocking=non_blocking),
            field=self.field.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ParticleState:
    """Instantaneous kinematic state of a single magnetic particle.

    Args:
    - `position`: Tensor of shape `(n_dims,)` giving the particle position.
       In 2D `position=(r, z)`; in 3D `position=(x, y, z)`.
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
        # if self.position.ndim !=2 :
        #     raise ValueError(
        #         "ParticleState.position must be a 2D tensor of shape (n_dims, 1); "
        #         f"got position shape {tuple(self.position.shape)}."
        #     )
    
    @property
    def axial_position(self) -> float:
        """Convenience property for the axial coordinate of the particle."""
        return self.position[-1].item()
    
    @property
    def radial_position(self) -> float:
        """Convenience property for the radial coordinate of the particle."""
        return self.position[0].item()

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32,
           non_blocking: bool = True) -> ParticleState:
        """Asynchronous CUDA transfer helper."""
        return ParticleState(
            position=self.position.to(device=device, dtype=dtype, non_blocking=non_blocking),
            velocity=self.velocity.to(device=device, dtype=dtype, non_blocking=non_blocking),
            time=self.time,
        )

# @dataclass(frozen=True)
# class SphericalObject:
#     r"""A rigid spherical *object* (obstacle) embedded on the channel's symmetry axis.
#
#     NEW. Describes the excluded-volume particle around which
#     [`sampling.sample_collocation_points_with_obstacle`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points_with_obstacle]
#     and
#     [`training.trainer.train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle]
#     solve a genuinely two-way-coupled flow field, as opposed to the
#     one-way, Faxén-corrected point-tracer model in
#     [`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity].
#
#     **Why the obstacle has no radial position.** This package's flow solver
#     is axisymmetric: every field is represented as a function of `(r, z)`
#     alone, which is only a faithful reduction of the true 3-D field when
#     every geometric feature of the domain — including any embedded
#     obstacle — is itself a solid of revolution about that same axis. A
#     sphere centered anywhere off the axis breaks that symmetry and would
#     need a genuinely 3-D (or non-axisymmetric 2-D azimuthal) solve, well
#     outside this package's architecture. A sphere centered *on* the axis
#     is the one obstacle placement compatible with `(r, z)` at all, so
#     `SphericalObstacle` only stores its axial coordinate: `r_p \equiv 0` is
#     not a simplifying assumption, it is what makes the excluded-volume
#     problem expressible in this coordinate system in the first place.
#
#     Args:
#     - `axial_position`: Dimensionless $z$-coordinate of the sphere's
#       center, in the same units as `Domain.length`.
#     - `radius`: Dimensionless sphere radius $a$ (i.e. $a / L$, matching
#       [`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]'s
#       output), in the same units as `Domain.radius`. Strictly positive.
#
#     Raises:
#     - `ValueError`: If `axial_position` is not finite, or `radius` is not
#       finite and strictly positive.
#     """
#
#     axial_position: float
#     radius: float
#
#     def __post_init__(self) -> None:
#         if not math.isfinite(self.axial_position):
#             raise ValueError(f"axial_position must be finite; got {self.axial_position!r}.")
#         if not math.isfinite(self.radius) or self.radius <= 0.0:
#             raise ValueError(f"radius must be finite and strictly positive; got {self.radius!r}.")