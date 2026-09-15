"""Nondimensionalization utilities for `magnetofluidics_pinn`.

The vessel and flow are described in physical, MKSA/SI quantities (meters,
seconds, pascals, tesla...) everywhere the person configuring a run is
concerned. Direct-solving in SI units at the microscale, however, hurts
gradient-based optimization: a channel radius of `2.0e-4` and a pressure
scale near `1.0e1` sit at very different orders of magnitude, which biases
the network's loss landscape.

This module is the single place responsible for converting between the two
representations:

- Physical (SI) quantities, used at the package boundary (configuration,
  plotting, reporting), and
- Dimensionless quantities, used internally by `sampling`, `physics`, and
  `networks`.

Every function here is a pure, closed-form arithmetic transform: no
randomness, no I/O, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from magnetofluidics_pinn.config import FieldConfig, FluidConfig, ParticleConfig
from magnetofluidics_pinn.types import Domain


@dataclass(frozen=True)
class Scales:
    r"""Characteristic physical scales used to nondimensionalize the problem.

    Args:
    - `length`: Characteristic length, in `meter` (from
      `FluidConfig.reference_length`).
    - `velocity`: Characteristic velocity, in `meter-per-second` (from
      `FluidConfig.reference_velocity`).
    - `time`: Characteristic time, in `second`, derived as
      $\text{length} / \text{velocity}$.
    - `pressure`: Characteristic pressure, in `pascal`, derived from the
      viscous stress scale `dynamic_viscosity * velocity / length`, the
      natural pressure scale for Stokes flow.
    - `magnetic_field`: Characteristic magnetic field, in `tesla` (from
      `FieldConfig.magnitude`).
    """

    length: float
    velocity: float
    time: float
    pressure: float
    magnetic_field: float


def compute_scales(fluid_config: FluidConfig, field_config: FieldConfig) -> Scales:
    """Derive the characteristic scales from the fluid and field configuration.

    Args:
    - `fluid_config`: Fluid configuration providing `reference_length`,
      `reference_velocity`, and `dynamic_viscosity`.
    - `field_config`: Magnetic field configuration providing `magnitude`.

    Returns:
    - A [`Scales`][magnetofluidics_pinn.scaling.Scales] instance.

    Raises:
    - `ValueError`: If `reference_length`, `reference_velocity`, or
      `magnitude` is not strictly positive, which would make the
      normalization ill-defined (division by zero or sign flip).
    """
    if fluid_config.reference_length <= 0.0:
        raise ValueError("fluid_config.reference_length must be strictly positive.")
    if fluid_config.reference_velocity <= 0.0:
        raise ValueError("fluid_config.reference_velocity must be strictly positive.")
    if field_config.magnitude <= 0.0:
        raise ValueError("field_config.magnitude must be strictly positive.")

    length = fluid_config.reference_length
    velocity = fluid_config.reference_velocity
    time = length / velocity
    # Viscous pressure scale, appropriate for the low-Reynolds-number (Stokes)
    # regime this package targets first; both regimes share the same scales
    # so that switching regimes does not require re-deriving a domain.
    pressure = fluid_config.dynamic_viscosity * velocity / length
    magnetic_field = field_config.magnitude

    return Scales(
        length=length,
        velocity=velocity,
        time=time,
        pressure=pressure,
        magnetic_field=magnetic_field,
    )


def nondimensionalize_domain(domain: Domain, scales: Scales) -> Domain:
    """Convert a physical (SI) domain to its dimensionless counterpart.

    Args:
    - `domain`: Physical domain, with `length` and `radius` in `meter`.
    - `scales`: Characteristic scales, as returned by
      [`compute_scales`][magnetofluidics_pinn.scaling.compute_scales].

    Returns:
    - A new [`Domain`][magnetofluidics_pinn.types.Domain] instance with
      `length` and `radius` divided by `scales.length`, and `branch_angle`
      left unchanged (angles are already dimensionless).
    """
    return Domain(
        kind=domain.kind,
        length=domain.length / scales.length,
        radius=domain.radius / scales.length,
        branch_angle=domain.branch_angle,
    )


def redimensionalize_domain(domain: Domain, scales: Scales) -> Domain:
    """Convert a dimensionless domain back to physical (SI) quantities.

    Args:
    - `domain`: Dimensionless domain, as produced by
      [`nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain].
    - `scales`: Characteristic scales used for the original normalization.

    Returns:
    - A new [`Domain`][magnetofluidics_pinn.types.Domain] instance with
      `length` and `radius` multiplied back by `scales.length`, in `meter`.
    """
    return Domain(
        kind=domain.kind,
        length=domain.length * scales.length,
        radius=domain.radius * scales.length,
        branch_angle=domain.branch_angle,
    )


def nondimensionalize_velocity(velocity_si: float, scales: Scales) -> float:
    """Convert a velocity from `meter-per-second` to dimensionless units.

    Args:
    - `velocity_si`: Velocity magnitude, in `meter-per-second`.
    - `scales`: Characteristic scales.

    Returns:
    - The dimensionless velocity, `velocity_si / scales.velocity`.
    """
    return velocity_si / scales.velocity


def redimensionalize_velocity(velocity_dimensionless: float, scales: Scales) -> float:
    """Convert a dimensionless velocity back to `meter-per-second`.

    Args:
    - `velocity_dimensionless`: Dimensionless velocity magnitude.
    - `scales`: Characteristic scales.

    Returns:
    - The physical velocity, in `meter-per-second`.
    """
    return velocity_dimensionless * scales.velocity


def nondimensionalize_pressure(pressure_si: float, scales: Scales) -> float:
    """Convert a pressure from `pascal` to dimensionless units.

    Args:
    - `pressure_si`: Pressure magnitude, in `pascal`.
    - `scales`: Characteristic scales.

    Returns:
    - The dimensionless pressure, `pressure_si / scales.pressure`.
    """
    return pressure_si / scales.pressure


def redimensionalize_pressure(pressure_dimensionless: float, scales: Scales) -> float:
    """Convert a dimensionless pressure back to `pascal`.

    Args:
    - `pressure_dimensionless`: Dimensionless pressure magnitude.
    - `scales`: Characteristic scales.

    Returns:
    - The physical pressure, in `pascal`.
    """
    return pressure_dimensionless * scales.pressure


def nondimensionalize_time(time_si: float, scales: Scales) -> float:
    """Convert a time duration from `second` to dimensionless units.

    Args:
    - `time_si`: Time duration, in `second`.
    - `scales`: Characteristic scales.

    Returns:
    - The dimensionless time, `time_si / scales.time`.
    """
    return time_si / scales.time


def redimensionalize_time(time_dimensionless: float, scales: Scales) -> float:
    """Convert a dimensionless time duration back to `second`.

    Args:
    - `time_dimensionless`: Dimensionless time duration.
    - `scales`: Characteristic scales.

    Returns:
    - The physical time duration, in `second`.
    """
    return time_dimensionless * scales.time


def nondimensionalize_magnetic_field(field_si: float, scales: Scales) -> float:
    """Convert a magnetic field magnitude from `tesla` to dimensionless units.

    Args:
    - `field_si`: Magnetic field magnitude, in `tesla`.
    - `scales`: Characteristic scales.

    Returns:
    - The dimensionless field magnitude, `field_si / scales.magnetic_field`.
    """
    return field_si / scales.magnetic_field


def redimensionalize_magnetic_field(field_dimensionless: float, scales: Scales) -> float:
    """Convert a dimensionless magnetic field magnitude back to `tesla`.

    Args:
    - `field_dimensionless`: Dimensionless magnetic field magnitude.
    - `scales`: Characteristic scales.

    Returns:
    - The physical magnetic field magnitude, in `tesla`.
    """
    return field_dimensionless * scales.magnetic_field


def nondimensionalize_particle(particle_config: ParticleConfig, scales: Scales) -> float:
    r"""Convert a physical particle radius to the dimensionless value used elsewhere.

    Args:
    - `particle_config`: Physical particle configuration; only `radius` is
      used here (see the note on `magnetic_moment` below).
    - `scales`: Characteristic scales, as returned by
      [`compute_scales`][magnetofluidics_pinn.scaling.compute_scales].

    Returns:
    - The dimensionless particle radius $a / L$ (with $L$ =
      `scales.length`), suitable for
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]'s
      `particle_radius` argument and
      [`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity]'s
      `particle_radius` argument.

    Note:
    - `particle_config.magnetic_moment` is *not* converted here, on
      purpose. `dipole_force` computes $\mathbf{F} = \nabla(\mathbf{m}
      \cdot \mathbf{B})$ and the current trajectory integrator adds that
      force directly to velocity under an explicit unit-mobility
      assumption (see
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]).
      A dimensionally consistent conversion for `magnetic_moment` requires
      a translational mobility (from the particle radius and the fluid
      viscosity) that is not yet part of this package's scaling model;
      inventing one here would silently imply a level of physical
      correctness the non-uniform-field path does not actually have.
      `magnetic_moment` must currently be supplied directly, already in
      whatever dimensionless units are consistent with that unit-mobility
      assumption - exact only when the field is uniform, per
      `trajectory.integrate_trajectory`'s docstring.
    """
    return particle_config.radius / scales.length


def compute_mobility_scale(fluid_config: FluidConfig) -> float:
    """Calculates the characteristic hydrodynamic mobility scale (SI).

    $$M_{ref} = 1 / (6 * \\pi * \\mu * L_{ref})$$
    """
    return 1.0 / (6.0 * np.pi * fluid_config.dynamic_viscosity * fluid_config.reference_length)


def nondimensionalize_mobility(particle_config: ParticleConfig, fluid_config: FluidConfig,
                               scales: Scales) -> torch.Tensor:
    """Generates the dimensionless mobility tensor based on the particle’s shape.

    Takes the surface configuration/shape into account to construct M*.
    For a sphere:  $M^* = diag(1/a^*, 1/a^*)$
    """
    a_nd = particle_config.radius / scales.length
    kind = getattr(particle_config, "kind", "spherical")
    
    if kind in ["sphere", "spherical"]:
        # Adimensional isotropic Stroke mobility
        mobility_scalar = 1.0 / a_nd
        return torch.diag(torch.tensor([mobility_scalar, mobility_scalar]))
    
    # elif kind == "spheroid":
        # aspect_ratio = particle_config.aspect_ratio  # ex: longueur / diamètre
        # Calculation of parallel and perpendicular drag coefficients
        # S_parallel, S_perpendicular = compute_oberbeck_factors(aspect_ratio)
        # M_parallel = 1.0 / (a_nd * S_parallel)
        # M_perp = 1.0 / (a_nd * S_perpendicular)
        # return torch.diag(torch.tensor([M_perp, M_parallel])) # Aligné sur le repère local
        pass
    
    raise NotImplementedError(f"The '{kind}' shape type is not yet supported.")
