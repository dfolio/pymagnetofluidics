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


---

**NEW — batched, GPU-resident integration (all particles advanced together).**
Every particle used to be integrated by its own, independent call to
`scipy.integrate.solve_ivp`, in a plain Python loop. `solve_ivp`'s adaptive
stepper lives on the CPU and calls back into this module's right-hand side
several times per accepted step; with one particle per call, every one of
those callbacks evaluated `flow_network` and `dipole_force` on a batch of
size one, which starves the GPU (the work per dispatch is tiny, and every
dispatch pays a full host-device synchronization) and, critically, does not
get any faster as `initial_states` grows - exactly the opposite of what a
particle swarm (the Phase 2 roadmap item) needs.

This module now flattens every particle's position into a single state
vector
$$\mathbf{y}(t) = \big(r_1(t), z_1(t), \ldots, r_n(t), z_n(t)\big) \in \mathbb{R}^{2n},
\qquad
\frac{d\mathbf{y}}{dt} = F(t, \mathbf{y}),$$
and drives **one** adaptive integration of $F$ for the whole batch: $F$
evaluates `flow_network`, the optional Faxén correction, and `dipole_force`
once per callback, for every particle at once, as a single batched forward
(and, when `particle_radius > 0`, second-order backward) pass. Wall-collision
events are still resolved exactly, by the same root-finding `solve_ivp`
already performs, through a *restart loop*: a terminal event stops the
whole batched system, so every time one fires, the colliding particle(s)
are identified (from `sol.t_events`), permanently masked out of the
right-hand side (their velocity is forced to exactly zero, freezing their
recorded position at the collision state), and integration resumes for the
remaining active particles from where it stopped. The state vector's size
never changes across restarts - only the mask does - which keeps the batch
shape stable across the whole call.

This is a drop-in replacement: `integrate_trajectory`'s signature, return
type, and per-particle semantics (one `ParticleState` history per particle,
ending at its own collision time if it collides, at `time_span[1]`
otherwise) are unchanged. Two consequences of batching are worth knowing
about, rather than discovering by surprise:

- The adaptive stepper's error control is now driven by the *worst-case*
  particle in the active batch at each step, so the exact set of recorded
  time points for a given particle can differ slightly from what a
  single-particle call would have produced. Final states still agree to
  within `r_tol` / `a_tol`, since both satisfy the same error criterion
  over the same physics.
- Under `particle_radius > 0.0`, a single invalid row (e.g. one particle's
  radial coordinate crossing the symmetry axis mid-trajectory, which
  [`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity]
  has never tolerated, single-particle or batched) now aborts the whole
  batch's integration rather than only that one particle's, since the
  Faxén correction's second-order autodiff is evaluated across the full
  batch in one call. This was already a hard failure in the single-particle
  code; batching only changes its blast radius.
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
from magnetofluidics_pinn.types import MagneticFieldSample, ParticleState, Domain
from magnetofluidics_pinn.config import ParticleConfig
from magnetofluidics_pinn.scaling import Scales, nondimensionalize_particle

# Offset (in every coordinate) used to probe whether `field_fn` is actually
# spatially uniform; see `_assert_uniform_field`. Small enough to stay a
# local probe, large enough not to be lost to floating-point noise.
_UNIFORMITY_PROBE_OFFSET = 0.1234

# Number of scalar state components (r, z) each particle occupies in the
# flattened state vector `y` that `scipy.integrate.solve_ivp` advances.
# Hard-coded to the axisymmetric (r, z) convention this whole package uses;
# a future 3-D extension would thread this through as a parameter instead.
_N_DIMS_PER_PARTICLE = 2


def _assert_uniform_field(
    field_fn: Callable[[torch.Tensor], MagneticFieldSample], reference_position: torch.Tensor
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


def _flatten_particle_positions(positions: torch.Tensor) -> np.ndarray:
    """Flatten a `(n_particles, n_dims)` position tensor into `solve_ivp`'s 1-D state layout.

    Uses row-major (C) order, so particle `i`'s coordinates occupy
    `y[i * n_dims : (i + 1) * n_dims]` - the same layout
    [`_unflatten_particle_positions`][magnetofluidics_pinn.trajectory.integrator._unflatten_particle_positions]
    and every wall-collision event function assume.

    Args:
    - `positions`: Tensor of shape `(n_particles, n_dims)`.

    Returns:
    - A 1-D `float64` NumPy array of shape `(n_particles * n_dims,)`, the
      dtype `scipy.integrate.solve_ivp` requires for its internal error
      control regardless of the flow network's own (typically `float32`)
      dtype.
    """
    return positions.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _unflatten_particle_positions(
        y: np.ndarray, n_particles: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of [`_flatten_particle_positions`][magnetofluidics_pinn.trajectory.integrator._flatten_particle_positions].

    Args:
    - `y`: 1-D state vector of shape `(n_particles * n_dims,)`, as produced
      by `solve_ivp` (always `float64`, regardless of the network's dtype).
    - `n_particles`: Number of particles encoded in `y`.
    - `device`: Device the returned tensor is placed on.
    - `dtype`: Dtype the returned tensor is cast to (typically the flow
      network's own dtype, so it can be fed straight back into it).

    Returns:
    - Tensor of shape `(n_particles, n_dims)` on `device`/`dtype`.
    """
    return torch.as_tensor(y, dtype=torch.float64).reshape(n_particles, _N_DIMS_PER_PARTICLE).to(
        device=device, dtype=dtype, non_blocking=True
    )


def _build_batched_ode_system(
        flow_network: nn.Module,
        field_fn: Callable[[torch.Tensor], MagneticFieldSample],
        magnetic_moment: torch.Tensor,
        mobility_tensor: torch.Tensor,
        particle_radius_nd: float,
        active_mask: torch.Tensor,
        n_particles: int,
        device: torch.device,
        dtype: torch.dtype,
) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""Build one right-hand side evaluating every active particle in a single batched pass.

    Replaces the previous per-particle closure: instead of being called
    once per particle with a batch of size one, the returned callable is
    called once per `solve_ivp` step with every particle's coordinates at
    once, so `flow_network` and `dipole_force` each see a real batch
    dimension - the whole point of this rewrite (see the module docstring).

    Args:
    - `flow_network`: Trained network mapping coordinates of shape
      `(n_particles, 2)` to `(n_particles, 3)`, the `(u_r, u_z, p)` flow
      field.
    - `field_fn`: Callable returning a
      [`MagneticFieldSample`][magnetofluidics_pinn.types.MagneticFieldSample]
      for a batch of coordinates.
    - `magnetic_moment`: Tensor of shape `(n_dims,)`, the (shared) magnetic
      moment every particle carries. Left genuinely 1-D, rather than
      pre-expanded to `(1, n_dims)`, so that
      [`dipole_force`][magnetofluidics_pinn.physics.magnetic_forcing.dipole_force]'s
      own broadcasting (`moment.unsqueeze(0).expand(n_particles, -1)`)
      expands it to match the batch - pre-expanding to `(1, n_dims)`
      would only ever broadcast correctly against a batch of exactly one
      particle.
    - `mobility_tensor`: Tensor of shape `(n_particles, n_dims, n_dims)`,
      already resolved to `device`/`dtype`.
    - `particle_radius_nd`: Dimensionless particle radius; `0.0` skips the
      Faxén correction entirely (plain point-particle velocity).
    - `active_mask`: Tensor of shape `(n_particles, 1)`, `1.0` for a
      particle still being integrated and `0.0` for one that has already
      collided with the wall. Multiplying the right-hand side by this mask
      is what freezes a collided particle's position for the remainder of
      the current restart segment, without changing the state vector's
      shape.
    - `n_particles`: Number of particles encoded in each call's state
      vector.
    - `device`: Device every tensor this closure creates is placed on.
    - `dtype`: Dtype every tensor this closure creates is cast to.

    Returns:
    - A callable `(t, y) -> dy/dt` with the exact signature
      `scipy.integrate.solve_ivp` expects for its `fun` argument, operating
      on the whole batch at once.
    """
    
    def ode_system(t: float, y: np.ndarray) -> np.ndarray:
        del t  # The flow field and the dipole force are both steady (time-independent).
        positions = _unflatten_particle_positions(y, n_particles, device, dtype)
        
        if particle_radius_nd > 0.0:
            positions_for_flow = positions.clone().requires_grad_(True)
            flow_velocity = faxen_corrected_velocity(
                flow_network, positions_for_flow, particle_radius_nd
            ).detach()
        else:
            with torch.inference_mode():
                flow_velocity = flow_network(positions)[:, :_N_DIMS_PER_PARTICLE]
        
        positions_for_force = positions.clone().requires_grad_(True)
        magnetic_force = dipole_force(field_fn, positions_for_force, magnetic_moment).detach()
        # Batched equivalent of the single-particle `f_mag @ mobility.T`:
        # result[n, j] = sum_k magnetic_force[n, k] * mobility_tensor[n, j, k].
        magnetic_drift = torch.bmm(
            magnetic_force.unsqueeze(1), mobility_tensor.transpose(1, 2)
        ).squeeze(1)
        
        velocity = (flow_velocity + magnetic_drift) * active_mask
        return velocity.reshape(-1).to(dtype=torch.float64).cpu().numpy()
    
    return ode_system


def _make_wall_collision_event(
        particle_index: int, effective_wall_limit: float,
) -> Callable[[float, np.ndarray], float]:
    """Build one terminal, root-findable wall-collision event for a single particle.

    Args:
    - `particle_index`: The particle's index; its radial coordinate lives
      at `y[particle_index * n_dims]` in the shared, flattened state vector
      (see
      [`_flatten_particle_positions`][magnetofluidics_pinn.trajectory.integrator._flatten_particle_positions]).
    - `effective_wall_limit`: Radial threshold at which the particle's
      surface (not just its center) first reaches the channel wall.

    Returns:
    - A callable suitable for `scipy.integrate.solve_ivp`'s `events`
      argument, with `.terminal = True` and `.direction = -1` already set:
      zero when the particle's surface first reaches the wall, detected
      only while approaching it (not while receding, which would be a
      stale root left over from the particle's own starting side).
    """
    radial_index = particle_index * _N_DIMS_PER_PARTICLE
    
    def event(t: float, y: np.ndarray) -> float:
        del t
        return effective_wall_limit - y[radial_index]
    
    event.terminal = True
    event.direction = -1
    return event


def _format_batch_progress_line(
        segment_index: int, t_current: float, n_active: int, n_particles: int,
) -> str:
    """Format one verbose line announcing the start of a new restart segment.

    Args:
    - `segment_index`: 0-based index of the restart segment about to run.
    - `t_current`: Time this segment starts integrating from.
    - `n_active`: Number of particles still being integrated in this segment.
    - `n_particles`: Total number of particles in the batch.

    Returns:
    - A single formatted line, e.g. `"Segment 2: t=0.0125, 2/3 particle(s) active."`.
    """
    return f"Segment {segment_index}: t={t_current:.6g}, {n_active}/{n_particles} particle(s) active."


def _recompute_velocities(
        ode_system: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray,
) -> np.ndarray:
    """Evaluate the right-hand side once more at an already-known state, to recover velocity.

    `solve_ivp` returns positions at every accepted step but not the
    corresponding derivative, so recovering velocity for the recorded
    [`ParticleState`][magnetofluidics_pinn.types.ParticleState] history
    means calling the same right-hand side again - exactly what the
    previous, per-particle implementation already did. Doing it once per
    recorded step for the *whole active batch*, rather than once per
    particle, is what keeps this a batched operation end to end.

    Args:
    - `ode_system`: The batched right-hand side built by
      [`_build_batched_ode_system`][magnetofluidics_pinn.trajectory.integrator._build_batched_ode_system].
    - `t`: Time at which `y` was recorded.
    - `y`: The flattened state vector at `t`.

    Returns:
    - The flattened velocity vector at `(t, y)`, same layout as `y`.
    """
    return ode_system(t, y)


def integrate_trajectory(
    flow_network: nn.Module,
    field_fn: Callable[[torch.Tensor], MagneticFieldSample],
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
    """Integrate particle trajectories with optimized host-GPU tensor management.

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
    - A list of per-particle trajectories, each containing a sequence of
      `ParticleState` instances ordered by increasing time and residing on
      the flow network's operational device. A particle that collides with
      the wall has its history end exactly at the (root-found) collision
      state; a particle that does not reach `time_span[1]` unresolved.

    Raises:
    - `ValueError`: If `initial_states` is empty, if `time_span` bounds are invalid, if any
      `ParticleState.position` shape is inconsistent across `initial_states`, or if
      `mobility_tensor` does not have shape `(n_particles, n_dims, n_dims)`.
    - `NotImplementedError`: If `field_fn` is not spatially uniform (see `_assert_uniform_field`).
    - `RuntimeError`: If the underlying adaptive solver fails to integrate the batch
      (`solve_ivp` reports `success=False`). Reaching the terminal wall-collision
      event below is a *successful*, expected termination, and does not raise.
    """
    if not initial_states:
        raise ValueError("initial_states must contain at least one particle.")
    time_start, time_end = time_span
    if time_end <= time_start:
        raise ValueError(f"time_span must satisfy t_end > t_start; got {time_span!r}.")
    
    n_particles = len(initial_states)
    n_dims = initial_states[0].position.shape[-1]
    if n_dims != _N_DIMS_PER_PARTICLE:
        raise ValueError(
            f"integrate_trajectory supports {n_dims} coordinates; expected {_N_DIMS_PER_PARTICLE}."
        )
    if mobility_tensor.shape != (n_particles, n_dims, n_dims):
        raise ValueError(f"mobility_tensor shape mismatch: got {tuple(mobility_tensor.shape)}.")
    
    # CHANGED: Removed the invalid exception banning on-axis particles (r <= 0)
    
    network_device = resolve_module_device(flow_network)
    network_dtype = resolve_module_dtype(flow_network)
    flow_network.eval()
    
    r_max_particle = particle_config.max_surface_extension / scales.length
    effective_wall_limit = domain.radius - r_max_particle
    particle_radius_nd = nondimensionalize_particle(particle_config, scales)
    
    magnetic_moment = torch.as_tensor(
        particle_config.magnetic_moment, device=network_device, dtype=network_dtype
    )
    mobility_tensor = mobility_tensor.to(device=network_device, dtype=network_dtype, non_blocking=True)
    
    if enforce_uniform_field:
        _assert_uniform_field(
            field_fn, initial_states[0].position.to(device=network_device, dtype=network_dtype).unsqueeze(0)
        )
    
    initial_positions = torch.stack([state.position for state in initial_states], dim=0)
    y_current = _flatten_particle_positions(
        initial_positions.to(device=network_device, dtype=network_dtype)
    )
    t_current = time_start
    
    trajectory_records: list[list[ParticleState]] = [[] for _ in range(n_particles)]
    active_mask_bool = np.ones(n_particles, dtype=bool)
    global_step_counter = 0
    segment_index = 0
    
    while t_current < time_end and active_mask_bool.any():
        active_indices = np.flatnonzero(active_mask_bool)
        active_mask_tensor = torch.as_tensor(
            active_mask_bool, device=network_device, dtype=network_dtype
        ).unsqueeze(1)
        
        ode_system = _build_batched_ode_system(
            flow_network=flow_network,
            field_fn=field_fn,
            magnetic_moment=magnetic_moment,
            mobility_tensor=mobility_tensor,
            particle_radius_nd=particle_radius_nd,
            active_mask=active_mask_tensor,
            n_particles=n_particles,
            device=network_device,
            dtype=network_dtype,
        )
        
        segment_events = [
            _make_wall_collision_event(int(index), effective_wall_limit) for index in active_indices
        ]
        
        if verbose:
            print(_format_batch_progress_line(segment_index, t_current, active_indices.size, n_particles))
        
        sol = solve_ivp(
            fun=ode_system,
            t_span=(t_current, time_end),
            y0=y_current,
            method="RK45",
            events=segment_events,
            rtol=r_tol,
            atol=a_tol,
        )
        
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed (segment {segment_index}): {sol.message}")
        
        n_steps = len(sol.t)
        for step in range(n_steps):
            t_val = sol.t[step]
            y_val = sol.y[:, step]
            v_val = _recompute_velocities(ode_system, t_val, y_val)
            for particle_index in active_indices:
                radial_slice = slice(
                    particle_index * _N_DIMS_PER_PARTICLE, (particle_index + 1) * _N_DIMS_PER_PARTICLE
                )
                trajectory_records[particle_index].append(
                    ParticleState(
                        position=torch.tensor(y_val[radial_slice], device=network_device, dtype=network_dtype),
                        velocity=torch.tensor(v_val[radial_slice], device=network_device, dtype=network_dtype),
                        time=float(t_val),
                    )
                )
            global_step_counter += 1
            if verbose and global_step_counter % log_every == 0:
                print(f"  step {global_step_counter}: t={t_val:.6g}")
        
        y_current = sol.y[:, -1]
        t_current = float(sol.t[-1])
        
        if sol.status == 1:
            newly_collided = [
                int(active_indices[event_index])
                for event_index, event_times in enumerate(sol.t_events)
                if event_times.size > 0
            ]
            active_mask_bool[newly_collided] = False
        
        segment_index += 1
    
    return trajectory_records

