r"""Closed-form Hagen-Poiseuille reference solution for Phase 1 verification.

`magnetofluidics_pinn.verification.analytical` provides the exact,
closed-form axisymmetric Stokes solution for a straight channel driven by
the parabolic inlet profile
[`boundary_conditions.flow_bc.inlet_velocity_condition`][magnetofluidics_pinn.boundary_conditions.flow_bc.inlet_velocity_condition]
imposes, together with its constant pressure gradient and the resulting
tracer streamline. Every function here is exposed with the same calling
convention a trained flow network already uses — `(n_points, 2) ->
(n_points, 3)` for the velocity/pressure field — so a verification notebook
can feed either the analytical solution or a trained network through the
same downstream evaluators in
[`verification.metrics`][magnetofluidics_pinn.verification.metrics] without
any special-casing.

Nothing here depends on a trained network or on training itself: this
module only encodes the closed-form mathematics of Hagen-Poiseuille flow
[@happel1983low; @leal2007advanced] and its Faxén-law finite-size
correction [@faxen1922widerstand; @kim2005microhydrodynamics], reusing
[`inlet_velocity_condition`][magnetofluidics_pinn.boundary_conditions.flow_bc.inlet_velocity_condition]
for the velocity profile itself rather than re-deriving it, so the
analytical reference and the boundary condition the network is actually
trained against are guaranteed to stay in lockstep.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from magnetofluidics_pinn.boundary_conditions.flow_bc import inlet_velocity_condition
from magnetofluidics_pinn.types import Domain


def analytical_pressure_gradient(radius: float, peak_velocity: float) -> float:
    r"""Compute the constant axial pressure gradient of Hagen-Poiseuille flow.

    For the parabolic profile $u_z^*(r^*) = u_\text{peak}^*\left(1 -
    (r^*/R^*)^2\right)$, the $z$-momentum equation $-\pdv{p^*}{z^*} +
    \laplacian u_z^* = 0$ integrates to a spatially constant gradient,

    $$ \pdv{p^*}{z^*} = -\frac{4 u_\text{peak}^*}{R^{*2}}. $$

    Args:
    - `radius`: Dimensionless channel radius $R^*$.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.

    Returns:
    - The constant pressure gradient $\pdv{p^*}{z^*}$, always non-positive
      for a strictly positive `peak_velocity` (pressure decreases along the
      flow direction).

    Raises:
    - `ValueError`: If `radius` or `peak_velocity` is not finite and
      strictly positive.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and strictly positive; got {radius!r}.")
    if not math.isfinite(peak_velocity) or peak_velocity <= 0.0:
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    return -4.0 * peak_velocity / radius**2


def hagen_poiseuille_pressure(
    axial_position: torch.Tensor,
    radius: float,
    peak_velocity: float,
    domain_length: float,
    reference_pressure: float = 0.0,
) -> torch.Tensor:
    r"""Evaluate the closed-form Hagen-Poiseuille pressure field.

    Integrates the constant gradient from
    [`analytical_pressure_gradient`][magnetofluidics_pinn.verification.analytical.analytical_pressure_gradient]
    from the outlet ($z^*=L^*$), where `reference_pressure` is imposed —
    matching the gauge choice
    [`training.trainer`][magnetofluidics_pinn.training.trainer]'s
    `_DIMENSIONLESS_OUTLET_REFERENCE_PRESSURE` uses:

    $$ p^*(z^*) = p_\text{ref}^* + \pdv{p^*}{z^*}\left(z^* - L^*\right). $$

    Args:
    - `axial_position`: Tensor of shape `(n_points, 1)` (or broadcastable)
      with the axial coordinate(s) at which pressure is evaluated.
    - `radius`: Dimensionless channel radius $R^*$.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.
    - `domain_length`: Dimensionless channel length $L^*$, where the
      reference pressure gauge is anchored.
    - `reference_pressure`: Dimensionless pressure imposed at the outlet;
      `0.0` by default, the standard gauge choice for an incompressible
      flow (pressure is only defined up to an additive constant).

    Returns:
    - A tensor with the same shape as `axial_position`, holding the
      pressure at each axial coordinate.

    Raises:
    - `ValueError`: If `radius` or `peak_velocity` is not finite and
      strictly positive, or if `domain_length` is not finite and strictly
      positive.
    """
    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError(f"domain_length must be finite and strictly positive; got {domain_length!r}.")
    pressure_gradient = analytical_pressure_gradient(radius, peak_velocity)
    return reference_pressure + pressure_gradient * (axial_position - domain_length)


def hagen_poiseuille_velocity_field(
    coordinates: torch.Tensor,
    domain: Domain,
    peak_velocity: float,
    reference_pressure: float = 0.0,
) -> torch.Tensor:
    r"""Evaluate the closed-form Hagen-Poiseuille $(u_r, u_z, p)$ field.

    Reuses
    [`inlet_velocity_condition`][magnetofluidics_pinn.boundary_conditions.flow_bc.inlet_velocity_condition]
    for the velocity components — the profile is $z$-independent, so
    evaluating it away from the inlet is exactly as valid — and appends the
    pressure field from
    [`hagen_poiseuille_pressure`][magnetofluidics_pinn.verification.analytical.hagen_poiseuille_pressure].
    Exposed with the `(n_points, 2) -> (n_points, 3)` calling convention a
    trained flow network already uses, so this function is a drop-in
    substitute for `network` in
    [`physics.fluid_residuals.stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
    or any evaluator in
    [`verification.metrics`][magnetofluidics_pinn.verification.metrics].

    Args:
    - `coordinates`: Tensor of shape `(n_points, 2)`, the axisymmetric
      `(r, z)` coordinates at which the field is evaluated.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.
    - `reference_pressure`: Dimensionless outlet reference pressure; see
      `hagen_poiseuille_pressure`.

    Returns:
    - Tensor of shape `(n_points, 3)` with columns $(u_r^*, u_z^*, p^*)$.

    Raises:
    - `ValueError`: If `coordinates` does not have shape `(n_points, 2)`.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (n_points, 2); got {tuple(coordinates.shape)}.")
    velocity = inlet_velocity_condition(domain, coordinates, peak_velocity)
    axial_position = coordinates[:, 1:2]
    pressure = hagen_poiseuille_pressure(
        axial_position, domain.radius, peak_velocity, domain.length, reference_pressure
    )
    return torch.cat([velocity, pressure], dim=1)


def faxen_offset_velocity(peak_velocity: float, radius: float, particle_radius: float) -> float:
    r"""Compute the closed-form Faxén-law axial-velocity offset for Hagen-Poiseuille flow.

    Because $\laplacian u_z^*$ is spatially constant for the parabolic
    profile (see
    [`analytical_pressure_gradient`][magnetofluidics_pinn.verification.analytical.analytical_pressure_gradient]'s
    derivation), Faxén's first law [@faxen1922widerstand;
    @kim2005microhydrodynamics] reduces to a single, position-independent
    offset rather than a function of $r^*$:

    $$ U_z^\text{Faxén} = u_z^*(r^*) - \frac{2}{3}\frac{a^{*2}}{R^{*2}}
    u_\text{peak}^*, \qquad U_r^\text{Faxén} = 0, $$

    since $u_r^* \equiv 0$ and its Laplacian vanishes identically. This
    function returns only the (negative) offset term; add it to
    $u_z^*(r^*)$ to recover $U_z^\text{Faxén}$.

    Args:
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.
    - `radius`: Dimensionless channel radius $R^*$.
    - `particle_radius`: Dimensionless particle radius $a^*$ (see
      [`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]);
      `0.0` recovers the point-particle limit (zero offset).

    Returns:
    - The scalar offset $-\frac{2}{3}\frac{a^{*2}}{R^{*2}}u_\text{peak}^*$.

    Raises:
    - `ValueError`: If `radius` or `peak_velocity` is not finite and
      strictly positive, or if `particle_radius` is not finite or
      negative.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and strictly positive; got {radius!r}.")
    if not math.isfinite(peak_velocity) or peak_velocity <= 0.0:
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    if not math.isfinite(particle_radius) or particle_radius < 0.0:
        raise ValueError(f"particle_radius must be finite and non-negative; got {particle_radius!r}.")
    laplacian_axial_velocity = -4.0 * peak_velocity / radius**2
    return (particle_radius**2 / 6.0) * laplacian_axial_velocity


def analytical_tracer_streamline(
    initial_position: tuple[float, float],
    peak_velocity: float,
    radius: float,
    time_grid: np.ndarray,
    initial_time: float = 0.0,
    particle_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the closed-form tracer streamline under a uniform magnetic field.

    Under a spatially uniform $\vb{B}$, the dipole force vanishes
    identically [@abbott2020magnetic] (see
    [`physics.magnetic_forcing.dipole_force`][magnetofluidics_pinn.physics.magnetic_forcing.dipole_force]),
    so a tracer is advected by the ambient flow velocity alone. For
    Hagen-Poiseuille flow that velocity is constant along a streamline
    (steady, $z$-independent profile), giving the closed form

    $$ r^*(t^*) = r_0^*, \qquad z^*(t^*) = z_0^* + U_z^\text{Faxén}(r_0^*)
    \left(t^* - t_0^*\right), $$

    where $U_z^\text{Faxén}$ reduces to $u_z^*(r_0^*)$ itself when
    `particle_radius` is `0.0`, or its Faxén-law-corrected value (see
    [`faxen_offset_velocity`][magnetofluidics_pinn.verification.analytical.faxen_offset_velocity])
    otherwise.

    Args:
    - `initial_position`: Tuple $(r_0^*, z_0^*)$, the tracer's initial
      dimensionless position.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.
    - `radius`: Dimensionless channel radius $R^*$.
    - `time_grid`: 1-D array of dimensionless times at which the streamline
      is evaluated.
    - `initial_time`: Dimensionless time corresponding to `initial_position`;
      `0.0` by default.
    - `particle_radius`: Dimensionless particle radius $a^*$; `0.0` (the
      default) recovers the exact point-tracer limit.

    Returns:
    - A tuple `(radial_positions, axial_positions)`, each a 1-D `numpy`
      array the same length as `time_grid`.

    Raises:
    - `ValueError`: If `radius` or `peak_velocity` is not finite and
      strictly positive, if `initial_position`'s radial component lies
      outside `[0.0, radius]`, if `particle_radius` is not finite or
      negative, if `particle_radius > 0.0` while the initial radial
      component is `0.0` or negative — the Faxén correction is only
      defined off the symmetry axis, mirroring
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]'s
      own guard — or if `time_grid` is not a non-empty 1-D array.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and strictly positive; got {radius!r}.")
    if not math.isfinite(peak_velocity) or peak_velocity <= 0.0:
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    if not math.isfinite(particle_radius) or particle_radius < 0.0:
        raise ValueError(f"particle_radius must be finite and non-negative; got {particle_radius!r}.")

    initial_radius, initial_axial_position = initial_position
    if not (0.0 <= initial_radius <= radius):
        raise ValueError(
            f"initial_position's radial component must lie in [0.0, {radius!r}]; got {initial_radius!r}."
        )
    if particle_radius > 0.0 and initial_radius <= 0.0:
        raise ValueError(
            "analytical_tracer_streamline received a non-zero particle_radius together with an "
            "on-axis initial position (r <= 0), where the Faxén correction is not defined."
        )

    time_grid = np.asarray(time_grid, dtype=np.float64)
    if time_grid.ndim != 1 or time_grid.size == 0:
        raise ValueError(f"time_grid must be a non-empty 1-D array; got shape {time_grid.shape!r}.")

    axial_velocity = peak_velocity * (1.0 - (initial_radius / radius) ** 2)
    if particle_radius > 0.0:
        axial_velocity += faxen_offset_velocity(peak_velocity, radius, particle_radius)

    radial_positions = np.full_like(time_grid, initial_radius)
    axial_positions = initial_axial_position + axial_velocity * (time_grid - initial_time)
    return radial_positions, axial_positions
