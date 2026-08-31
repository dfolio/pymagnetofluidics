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
dimensionally-inconsistent trajectory for a non-uniform field.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.physics.hydrodynamic_drag import faxen_corrected_velocity
from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force
from magnetofluidics_pinn.types import FieldSample, ParticleState

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


def _select_moment(
    magnetic_moment: torch.Tensor, particle_index: int, n_dims: int
) -> torch.Tensor:
    """Select the magnetic moment of shape `(1, n_dims)` for one particle.

    Args:
    - `magnetic_moment`: Tensor of shape `(n_dims,)`, shared by every
      particle, or `(n_particles, n_dims)`, one row per particle.
    - `particle_index`: Index of the particle whose moment is requested.
    - `n_dims`: Expected number of spatial dimensions.

    Returns:
    - Tensor of shape `(1, n_dims)`.

    Raises:
    - `ValueError`: If `magnetic_moment` has neither 1 nor 2 dimensions, or
      if its last dimension does not equal `n_dims`.
    """
    if magnetic_moment.ndim not in (1, 2) or magnetic_moment.shape[-1] != n_dims:
        raise ValueError(
            "magnetic_moment must have shape (n_dims,) or (n_particles, n_dims) "
            f"with n_dims={n_dims}; got shape {tuple(magnetic_moment.shape)}."
        )
    if magnetic_moment.ndim == 1:
        return magnetic_moment.unsqueeze(0)
    return magnetic_moment[particle_index : particle_index + 1]


def _drift_velocity(
    flow_network: nn.Module,
    field_fn: Callable[[torch.Tensor], FieldSample],
    position: torch.Tensor,
    moment: torch.Tensor,
    particle_radius: float,
) -> torch.Tensor:
    """Evaluate the instantaneous drift velocity of a tracer particle.

    The particle is transported by the (optionally Faxén-corrected) local
    flow velocity together with a magnetic drift term obtained under a
    unit-mobility, overdamped (Stokes) approximation: `v = u(x) + F(x)`,
    with `F = grad(m . B)` the point-dipole force [@abbott2020magnetic]. A
    proper translational mobility coefficient - set by
    the particle radius and the fluid viscosity - is expected to enter the
    magnetic term once particle-scale properties join the configuration
    model more broadly; until then, this unit-mobility choice is exact
    whenever the field is uniform (Phase 1), since the dipole force then
    vanishes identically and only the flow term survives.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure.
    - `field_fn`: Callable returning the magnetic field at given
      coordinates.
    - `position`: Tensor of shape `(1, n_dims)`, the particle's current
      position, already on `flow_network`'s device and dtype (see
      `integrate_trajectory`, which resolves both once for the whole
      trajectory rather than on every step).
    - `moment`: Tensor of shape `(1, n_dims)`, the particle's magnetic
      moment, already on the same device and dtype as `position`.
    - `particle_radius`: Radius used for the Faxén correction to the flow
      term; `0.0` skips the correction (plain point-particle velocity).

    Returns:
    - Tensor of shape `(1, n_dims)` with the drift velocity, on the same
      device as `position`.
    """
    if particle_radius > 0.0:
        position_for_flow = position.clone().requires_grad_(True)
        flow_velocity = faxen_corrected_velocity(
            flow_network, position_for_flow, particle_radius
        ).detach()
    else:
        n_dims = position.shape[1]
        with torch.no_grad():
            flow_output = flow_network(position)
        flow_velocity = flow_output[:, :n_dims]

    position_for_force = position.clone().requires_grad_(True)
    magnetic_drift = dipole_force(field_fn, position_for_force, moment).detach()

    return flow_velocity + magnetic_drift


def integrate_trajectory(
    flow_network: nn.Module,
    field_fn: Callable[[torch.Tensor], FieldSample],
    initial_states: list[ParticleState],
    magnetic_moment: torch.Tensor,
    time_span: tuple[float, float],
    n_steps: int,
    particle_radius: float = 0.0,
) -> list[list[ParticleState]]:
    """Integrate the trajectories of one or several magnetic particles.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure. Every computation happens on this network's device (see
      [`device_utils.resolve_module_device`][magnetofluidics_pinn.device_utils.resolve_module_device]);
      `initial_states` and `magnetic_moment` do not need to already be on
      that device — they are moved there automatically — so a network
      trained on `"cuda"` can be integrated directly from, e.g., plain CPU
      tensors constructed with `torch.tensor(...)`.
    - `field_fn`: Callable returning a
      [`FieldSample`][magnetofluidics_pinn.types.FieldSample] for a batch of
      coordinates.
    - `initial_states`: One [`ParticleState`][magnetofluidics_pinn.types.ParticleState]
      per particle, at the initial time.
    - `magnetic_moment`: Tensor of shape `(n_particles, n_dims)` or
      `(n_dims,)` with each particle's magnetic moment.
    - `time_span`: Tuple `(t_start, t_end)` bounding the integration.
    - `n_steps`: Number of integration steps.
    - `particle_radius`: Dimensionless particle radius `a`, used for the
      Faxén-law correction (see
      [`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity])
      to the flow velocity the particle is advected by. `0.0` (the default)
      recovers the exact point-particle limit. Only accurate for `a` small
      relative to the channel radius and to the flow's local radius of
      curvature — see this module's docstring for what a non-small particle
      radius would actually require.

    Returns:
    - A list of per-particle trajectories, each a list of
      [`ParticleState`][magnetofluidics_pinn.types.ParticleState] instances
      ordered by increasing time and living on `flow_network`'s device.

    Raises:
    - `ValueError`: If `initial_states` is empty, if `n_steps` is not
      strictly positive, if `time_span` is not increasing, if
      `particle_radius` is negative, or if `magnetic_moment` cannot be
      matched to each particle's coordinate dimensionality.
    - `NotImplementedError`: If `field_fn` is not spatially uniform (see
      `_assert_uniform_field`).
    """
    if not initial_states:
        raise ValueError("initial_states must contain at least one particle.")
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive.")
    time_start, time_end = time_span
    if time_end <= time_start:
        raise ValueError(f"time_span must satisfy t_end > t_start; got {time_span!r}.")
    if particle_radius < 0.0:
        raise ValueError(f"particle_radius must be non-negative; got {particle_radius!r}.")

    # Resolve the network's device and dtype once, up front, and move every
    # particle onto both: `flow_network(position)` requires `position` to
    # already match where the network's parameters live *and* what dtype
    # they use, and re-deriving either independently per step would both
    # waste time and risk a mismatch if `flow_network` were ever changed
    # mid-integration.
    network_device = resolve_module_device(flow_network)
    network_dtype = resolve_module_dtype(flow_network)
    step_size = (time_end - time_start) / n_steps
    moment_tensor = torch.as_tensor(magnetic_moment, device=network_device, dtype=network_dtype)

    _assert_uniform_field(
        field_fn, initial_states[0].position.to(device=network_device, dtype=network_dtype).unsqueeze(0)
    )

    trajectories: list[list[ParticleState]] = []
    for particle_index, initial_state in enumerate(initial_states):
        n_dims = initial_state.position.shape[-1]
        moment = _select_moment(moment_tensor, particle_index, n_dims).to(
            dtype=network_dtype, device=network_device
        )

        state = ParticleState(
            position=initial_state.position.to(device=network_device, dtype=network_dtype),
            velocity=initial_state.velocity.to(device=network_device, dtype=network_dtype),
            time=initial_state.time,
        )
        history = [state]
        for _ in range(n_steps):
            position = state.position.unsqueeze(0)

            # Fourth-order Runge-Kutta integration of dx/dt = v_drift(x): a
            # standard, fixed-step choice when derivative evaluations
            # (a network forward pass plus a small autograd call) are
            # comparatively cheap.
            k1 = _drift_velocity(flow_network, field_fn, position, moment, particle_radius)
            k2 = _drift_velocity(
                flow_network, field_fn, position + 0.5 * step_size * k1, moment, particle_radius
            )
            k3 = _drift_velocity(
                flow_network, field_fn, position + 0.5 * step_size * k2, moment, particle_radius
            )
            k4 = _drift_velocity(
                flow_network, field_fn, position + step_size * k3, moment, particle_radius
            )

            next_position = (
                position + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            ).squeeze(0)
            next_time = state.time + step_size
            next_velocity = _drift_velocity(
                flow_network, field_fn, next_position.unsqueeze(0), moment, particle_radius
            ).squeeze(0)

            state = ParticleState(position=next_position, velocity=next_velocity, time=next_time)
            history.append(state)

        trajectories.append(history)

    return trajectories
