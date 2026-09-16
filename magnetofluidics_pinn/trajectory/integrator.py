"""Particle trajectory integration.

Integrates the equation of motion of one or several magnetic particles,
combining the trained flow velocity field with the magnetic dipole force
from a prescribed field. Implemented as a pure function returning the full
sequence of states rather than mutating a persistent particle object.

**The particle-flow coupling condition.** A magnetic microrobot is not
truly a mathematical point: it has a finite radius `a`, and the flow it is
embedded in varies over that scale. This module now makes that relationship
explicit through an optional `particle_radius`, applying Faxén's first law
(see
[`physics.hydrodynamic_drag`][magnetofluidics_pinn.physics.hydrodynamic_drag])
as the closure condition connecting the continuum flow field to the
particle's (discrete) equation of motion — the correction that was
previously left implicit behind a "unit-mobility" comment. `particle_radius
= 0.0` (the default) recovers the exact point-particle limit used
throughout the rest of Phase 1's verification.

**What this does not do.** Faxén's law is a one-way correction: the flow
field is unaffected by the particle's presence. A microrobot whose radius
is not small relative to the channel — where the reverse effect, the
particle perturbing the flow around it, matters — needs a genuinely
two-way-coupled solve with an excluded spherical region and a no-slip
condition on the particle's own moving surface. That is a materially larger
problem (the domain geometry itself becomes a function of the particle's
position, effectively adding it as a network input), consistent with this
package's phased roadmap deferring particle-scale flow perturbation past
Phase 1's single-tracer scope, rather than an oversight in this module.

**Units and dtype.** `magnetic_moment` and every position fed to
`flow_network` are expected to already be dimensionless, matching whatever
the network was trained on (see
[`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]
for the particle radius specifically). `initial_states` and
`magnetic_moment` also do not need to already match `flow_network`'s
device *or* dtype — both are resolved from the network once, up front, and
every new tensor this module creates is matched to them, so a network
moved to, e.g., `"cuda"` and/or `torch.float64` can still be driven
directly from plain CPU, default-dtype tensors.

**Field uniformity is enforced, not just assumed.** Since the drift model
above is only dimensionally correct for a uniform field (see the caveat
two paragraphs up), `integrate_trajectory` probes `field_fn` at two
distinct points before integrating anything and raises
`NotImplementedError` if they disagree, rather than silently producing a
dimensionally-inconsistent trajectory for a non-uniform field. This check
can be switched off via `enforce_uniform_field=False`, for a deliberately
synthetic, non-physical `field_fn` used only to exercise this function's
mechanics (e.g. the wall-collision event below) — not for a
physically-meaningful non-uniform field, which this package does not yet
validate.

**Nondimensionalization.** `domain` and `scales` are passed in separately
(rather than the raw, physical `DomainConfig`) so that every internal
normalization — the wall-collision threshold and the particle radius alike
— is anchored to the same `scales.length` the flow network was trained
against (see
[`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]
and
[`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]).
An earlier version of this module re-derived these normalizations from
`DomainConfig.radius` directly; that is only equivalent to using
`scales.length` when `fluid_config.reference_length == domain_config.radius`
happens to hold, which no other part of this package's API guarantees.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from scipy.integrate import solve_ivp

from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.physics.hydrodynamic_drag import faxen_corrected_velocity
from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force
from magnetofluidics_pinn.types import FieldSample, ParticleState, Domain
from magnetofluidics_pinn.config import ParticleConfig
from magnetofluidics_pinn.scaling import Scales, nondimensionalize_particle

# Offset (in every coordinate) used to probe whether `field_fn` is actually
# spatially uniform; see `_assert_uniform_field`. Small enough to stay a
# local probe, large enough not to be lost to floating-point noise.
_UNIFORMITY_PROBE_OFFSET = 0.1234


def _assert_uniform_field(
    field_fn: Callable[[torch.Tensor], FieldSample], reference_position: torch.Tensor
) -> None:
    """Verify `field_fn` gives the same field at two distinct nearby points.

    The current drift model, `v = u(x) + F(x)` with unit mobility, is only
    dimensionally consistent when the magnetic force `F` is identically
    zero, i.e., when the field is spatially uniform (see this module's
    docstring). `gradient_field` and `biot_savart_field` both already raise
    `NotImplementedError`, so this check exists for the case those don't
    cover: a user-supplied, genuinely non-uniform `field_fn`, which would
    otherwise silently produce a dimensionally incorrect trajectory instead
    of failing loudly.

    Args:
    - `field_fn`: Callable returning a
      [`FieldSample`][magnetofluidics_pinn.types.FieldSample] for a batch of
      coordinates.
    - `reference_position`: Tensor of shape `(1, n_dims)` near which to
      probe; typically the first particle's initial position.

    Raises:
    - `NotImplementedError`: If the field differs between the two probed
      points by more than a small numerical tolerance.
    """
    probe_a = reference_position
    probe_b = reference_position + _UNIFORMITY_PROBE_OFFSET
    with torch.no_grad():
        field_a = field_fn(probe_a).field
        field_b = field_fn(probe_b).field
    if not torch.allclose(field_a, field_b, atol=1.0e-8, rtol=1.0e-5):
        raise NotImplementedError(
            "integrate_trajectory detected a spatially-varying (non-uniform) "
            "magnetic field. The current unit-mobility drift model "
            "(v = u(x) + F(x)) is only dimensionally consistent when "
            "F = grad(m . B) is identically zero, i.e. for a uniform field: "
            "a non-uniform field requires a fully nondimensionalized "
            "mobility model this package does not implement yet (see this "
            "module's docstring). Non-uniform-field trajectory integration "
            "is not yet supported."
        )


# def _select_moment(
#     magnetic_moment: torch.Tensor, particle_index: int, n_dims: int
# ) -> torch.Tensor:
#     """Select the magnetic moment of shape `(1, n_dims)` for one particle.
#
#     Args:
#     - `magnetic_moment`: Tensor of shape `(n_dims,)`, shared by every
#       particle, or `(n_particles, n_dims)`, one row per particle.
#     - `particle_index`: Index of the particle whose moment is requested.
#     - `n_dims`: Expected number of spatial dimensions.
#
#     Returns:
#     - Tensor of shape `(1, n_dims)`.
#
#     Raises:
#     - `ValueError`: If `magnetic_moment` has neither 1 nor 2 dimensions, or
#       if its last dimension does not equal `n_dims`.
#     """
#     if magnetic_moment.ndim not in (1, 2) or magnetic_moment.shape[-1] != n_dims:
#         raise ValueError(
#             "magnetic_moment must have shape (n_dims,) or (n_particles, n_dims) "
#             f"with n_dims={n_dims}; got shape {tuple(magnetic_moment.shape)}."
#         )
#     if magnetic_moment.ndim == 1:
#         return magnetic_moment.unsqueeze(0)
#     return magnetic_moment[particle_index : particle_index + 1]
#
#
# def _drift_velocity(
#     flow_network: nn.Module,
#     field_fn: Callable[[torch.Tensor], FieldSample],
#     position: torch.Tensor,
#     moment: torch.Tensor,
#     particle_radius: float,
#     mobility_tensor: torch.Tensor,
# ) -> torch.Tensor:
#     """Evaluate the instantaneous drift velocity of a tracer particle.
#
#     The particle is transported by the (optionally Faxén-corrected) local
#     flow velocity together with a magnetic drift term obtained under a
#     unit-mobility, overdamped (Stokes) approximation: `v = u(x) + F(x)`,
#     with `F = grad(m . B)` the point-dipole force [@abbott2020magnetic]. A
#     proper translational mobility coefficient - set by
#     the particle radius and the fluid viscosity - is expected to enter the
#     magnetic term once particle-scale properties join the configuration
#     model more broadly; until then, this unit-mobility choice is exact
#     whenever the field is uniform (Phase 1), since the dipole force then
#     vanishes identically and only the flow term survives.
#
#     Args:
#     - `flow_network`: Trained network mapping coordinates to velocity and
#       pressure.
#     - `field_fn`: Callable returning the magnetic field at given
#       coordinates.
#     - `position`: Tensor of shape `(1, n_dims)`, the particle's current
#       position, already on `flow_network`'s device and dtype (see
#       `integrate_trajectory`, which resolves both once for the whole
#       trajectory rather than on every step).
#     - `moment`: Tensor of shape `(1, n_dims)`, the particle's magnetic
#       moment, already on the same device and dtype as `position`.
#     - `particle_radius`: Radius used for the Faxén correction to the flow
#       term; `0.0` skips the correction (plain point-particle velocity).
#
#     Returns:
#     - Tensor of shape `(1, n_dims)` with the drift velocity, on the same
#       device as `position`.
#     """
#     if particle_radius > 0.0:
#         position_for_flow = position.clone().requires_grad_(True)
#         flow_velocity = faxen_corrected_velocity(
#             flow_network, position_for_flow, particle_radius
#         ).detach()
#     else:
#         n_dims = position.shape[1]
#         with torch.no_grad():
#             flow_output = flow_network(position)
#         flow_velocity = flow_output[:, :n_dims]
#
#     position_for_force = position.clone().requires_grad_(True)
#     magnetic_force  = dipole_force(field_fn, position_for_force, moment).detach()
#     magnetic_drift = torch.matmul(magnetic_force, mobility_tensor.T)
#
#     return flow_velocity + magnetic_drift
#
#
# def _enforce_surface_boundary_constraint(
#         position: torch.Tensor,
#         particle_radius: float,
#         channel_radius: float
# ) -> tuple[torch.Tensor, bool]:
#     """Structurally confines the centre of the particle within the domain.
#
#     Prevents the surface of the particle from penetrating the outer wall (`r = R`).
#     """
#     r_coord = position[0]
#     z_coord = position[1]
#
#     # Limite physique pour le centre d'une forme sphérique : R_eff = R - a
#     effective_radius = channel_radius - particle_radius
#
#     if r_coord >= effective_radius:
#         # Collision de surface détectée : projection sur la limite de contact
#         constrained_position = torch.tensor([effective_radius, z_coord], device=position.device, dtype=position.dtype)
#         return constrained_position, True
#
#     return position, False


def _format_trajectory_line(
    step: int, n_steps: int, current_time: float, positions: torch.Tensor
) -> str:
    """Formatte une ligne de progression textuelle pour le suivi de l'intégration.
    
    Args:
        step: Indice du pas actuel.
        n_steps: Nombre total de pas.
        current_time: Temps physique adimensionnel actuel.
        positions: Tensor contenant les positions de toutes les particules au pas actuel.
    """
    return f"{step}/{n_steps} : t={current_time} at pos {positions}"


def integrate_trajectory(
    flow_network: nn.Module,
    field_fn: Callable[[torch.Tensor], FieldSample],
    domain: Domain,
    scales: Scales,
    particle_config: ParticleConfig,
    initial_states: list[ParticleState],
    mobility_tensor: torch.Tensor,
    time_span: tuple[float, float],
    enforce_uniform_field: bool = True,
    r_tol: float = 1.0e-6,
    a_tol: float = 1.0e-8,
    verbose: bool = False,
    log_every: int = 10,
) -> list[list[ParticleState]]:
    """Integrate the trajectories of one or several magnetic particles using SciPy's adaptive IVP solvers.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and pressure.
    - `field_fn`: Callable returning a FieldSample for a batch of coordinates.
    - `domain`: Already-nondimensionalized vessel geometry (see
      `scaling.nondimensionalize_domain`), consistent with the geometry
      `flow_network` was trained on. Its `radius` sets the wall-collision
      threshold together with `particle_config`'s (also nondimensionalized)
      surface extension.
    - `scales`: Characteristic scales the flow network's coordinate system
      is built on (see `scaling.compute_scales`), used to nondimensionalize
      `particle_config.radius` via `scaling.nondimensionalize_particle`
      rather than re-deriving that normalization from `domain.radius`.
    - `particle_config`: Particle configuration containing shape and surface boundary limits.
    - `initial_states`: List of initial ParticleState structures, one per particle.
    - `mobility_tensor`: Tensor of shape `(n_particles, n_dims, n_dims)` for transport coupling.
    - `time_span`: Tuple `(t_start, t_end)` bounding the integration interval.
    - `enforce_uniform_field`: If `True` (default), reject a spatially-varying
      `field_fn` via `_assert_uniform_field` before integrating. Set to
      `False` only for a deliberately synthetic, non-physical `field_fn` used
      to exercise this function's *mechanics* (e.g. the wall-collision event) —
      this package does not otherwise validate a non-uniform field's
      dimensional correctness (see the module docstring).
    - `r_tol`: Relative error tolerance passed to the adaptive numerical solver.
    - `a_tol`: Absolute error tolerance passed to the adaptive numerical solver.
    - `verbose`: Enable verbose message
    - `log_every`: Log progress every `log_every` steps.

    Returns:
    - A list of per-particle trajectories, each containing a sequence of ParticleState instances
      ordered by increasing time and residing on the flow network's operational device.

    Raises:
    - `ValueError`: If `initial_states` is empty, if `time_span` bounds are invalid, if any
      `ParticleState.position` shape is inconsistent across `initial_states`, or if
      `mobility_tensor` does not have shape `(n_particles, n_dims, n_dims)`.
    - `NotImplementedError`: If `field_fn` is not spatially uniform (see `_assert_uniform_field`).
    - `RuntimeError`: If the underlying adaptive solver fails to integrate a particle's
      trajectory (`solve_ivp` reports `success=False`). Reaching the terminal wall-collision
      event below is a *successful*, expected termination, and does not raise.
    """
    if not initial_states:
        raise ValueError("initial_states must contain at least one particle.")
    time_start, time_end = time_span
    if time_end <= time_start:
        raise ValueError(f"time_span must satisfy t_end > t_start; got {time_span!r}.")

    n_dims = initial_states[0].position.shape[-1]
    if any(state.position.shape != (n_dims,) for state in initial_states):
        raise ValueError(
            f"Every ParticleState.position must have shape ({n_dims},), matching the "
            "first particle's position; got a mismatched shape among initial_states."
        )
    if mobility_tensor.shape != (len(initial_states), n_dims, n_dims):
        raise ValueError(
            f"mobility_tensor must have shape ({len(initial_states)}, {n_dims}, {n_dims}) "
            f"(n_particles, n_dims, n_dims); got {tuple(mobility_tensor.shape)}."
        )

    # Resolve execution context properties once up-front to eliminate device shifting overhead
    network_device = resolve_module_device(flow_network)
    network_dtype = resolve_module_dtype(flow_network)

    r_max_particle = particle_config.max_surface_extension / scales.length
    effective_wall_limit = domain.radius - r_max_particle
    a_nd = nondimensionalize_particle(particle_config, scales)

    moment_tensor = torch.as_tensor(
        particle_config.magnetic_moment, device=network_device, dtype=network_dtype
    ).unsqueeze(0)
    
    if enforce_uniform_field:
        # Enforce field uniformity checks prior to launching structural solvers
        _assert_uniform_field(
            field_fn, initial_states[0].position.to(device=network_device, dtype=network_dtype).unsqueeze(0)
        )

    def make_ode_system(p_index: int) -> Callable[[float, np.ndarray], np.ndarray]:
        """Closure factory tracking the structural parameters of an individual particle index."""
        # Isolate the specific particle's mobility tensor slice and move it onto the active device
        current_mobility = mobility_tensor[p_index].to(device=network_device, dtype=network_dtype)

        def ode_system(t: float, y: np.ndarray) -> np.ndarray:
            pos_tensor = torch.tensor(y, device=network_device, dtype=network_dtype).unsqueeze(0)

            # 1. Hydrodynamic flow advection contribution (with optional Faxen Laplacian correction)
            if a_nd > 0.0:
                pos_flow = pos_tensor.clone().requires_grad_(True)
                u_flow = faxen_corrected_velocity(flow_network, pos_flow, a_nd).detach()
            else:
                with torch.no_grad():
                    u_flow = flow_network(pos_tensor)[:, :2]

            # 2. Magnetic forcing contribution mapped through the adimensionless mobility tensor
            pos_force = pos_tensor.clone().requires_grad_(True)
            f_mag = dipole_force(field_fn, pos_force, moment_tensor).detach()
            u_mag = torch.matmul(f_mag, current_mobility.T)

            # Reduce contributions and unpack vector into standard NumPy format
            v_drift = (u_flow + u_mag).squeeze(0).cpu().numpy()
            return v_drift

        return ode_system

    def wall_collision_event(t: float, y: np.ndarray) -> float:
        """Zero when the particle's surface first reaches the channel wall."""
        # Returns 0 once the particle's outer surface contacts the wall.
        return effective_wall_limit - y[0]

    wall_collision_event.terminal = True  # Stop the integration immediately upon impact
    wall_collision_event.direction = -1  # Detection only when approaching the wall

    trajectories: list[list[ParticleState]] = []
    for particle_index, particle_state in enumerate(initial_states):
        y0 = particle_state.position.detach().cpu().numpy().astype(np.float64)
        active_ode = make_ode_system(particle_index)
        
        if verbose:
            print(f"Start to track particle {particle_index}.")
            
        # Execute SciPy's adaptive Runge-Kutta method (RK45): the embedded
        # Dormand-Prince pair [@dormand1980family], not Cash-Karp.
        sol = solve_ivp(
            fun=active_ode,
            t_span=time_span,
            y0=y0,
            method="RK45",
            events=wall_collision_event,
            rtol=r_tol,
            atol=a_tol,
        )

        if not sol.success:
            raise RuntimeError(
                f"solve_ivp failed to integrate particle {particle_index}: {sol.message}"
            )

        history: list[ParticleState] = []
        n_steps = len(sol.t)
        for step in range(n_steps):
            pos_np = sol.y[:, step]
            t_val = sol.t[step]
            v_np = active_ode(t_val, pos_np)

            state = ParticleState(
                position=torch.tensor(pos_np, device=network_device, dtype=network_dtype),
                velocity=torch.tensor(v_np, device=network_device, dtype=network_dtype),
                time=t_val,
            )
            history.append(state)
            if verbose and (step % log_every == 0):
                print(_format_trajectory_line(step,n_steps, t_val, pos_np))
            
        trajectories.append(history)

    return trajectories

