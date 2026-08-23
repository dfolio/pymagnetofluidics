"""Particle trajectory integration.

Integrates the equation of motion of one or several magnetic particles,
combining the trained flow velocity field with the magnetic dipole force
from a prescribed field. Implemented as a pure function returning the full
sequence of states rather than mutating a persistent particle object.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force
from magnetofluidics_pinn.types import FieldSample, ParticleState


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
) -> torch.Tensor:
    """Evaluate the instantaneous drift velocity of a tracer particle.

    The particle is transported by the local flow velocity together with a
    magnetic drift term obtained under a unit-mobility, overdamped (Stokes)
    approximation: `v = u_f(x) + F(x)`, with `F = grad(m . B)` the point-
    dipole force (Abbott, Diller, & Petruska, 2020). A proper mobility
    coefficient - set by the sphere radius and the fluid viscosity - is
    expected to enter once particle-scale properties join the configuration
    model; until then, this unit-mobility choice is exact whenever the
    field is uniform (Phase 1), since the dipole force then vanishes
    identically and only the flow term survives.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure.
    - `field_fn`: Callable returning the magnetic field at given
      coordinates.
    - `position`: Tensor of shape `(1, n_dims)`, the particle's current
      position.
    - `moment`: Tensor of shape `(1, n_dims)`, the particle's magnetic
      moment.

    Returns:
    - Tensor of shape `(1, n_dims)` with the drift velocity.
    """
    n_dims = position.shape[1]
    with torch.no_grad():
        flow_output = flow_network(position)
    flow_velocity = flow_output[:, :n_dims]

    position_for_force = position.clone().requires_grad_(True)
    magnetic_drift = dipole_force(field_fn, position_for_force, moment).detach()

    return flow_velocity + magnetic_drift


def _get_module_device_and_dtype(
    module: nn.Module,
) -> tuple[torch.device, torch.dtype]:
    """Retrieve the primary device and dtype of a neural network module.

    Args:
    - `module`: The PyTorch module to inspect.

    Returns:
    - A tuple `(device, dtype)` extracted from the module's parameters,
      defaulting to CUDA (if available) and `torch.float32`.
    """
    try:
        first_param = next(module.parameters())
        return first_param.device, first_param.dtype
    except StopIteration:
        default_dev = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        return default_dev, torch.float32


def integrate_trajectory(
    flow_network: nn.Module,
    field_fn: Callable[[torch.Tensor], FieldSample],
    initial_states: list[ParticleState],
    magnetic_moment: torch.Tensor,
    time_span: tuple[float, float],
    n_steps: int,
) -> list[list[ParticleState]]:
    """Integrate the trajectories of one or several magnetic particles.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure.
    - `field_fn`: Callable returning a
      [`FieldSample`][magnetofluidics_pinn.types.FieldSample] for a batch of
      coordinates.
    - `initial_states`: One [`ParticleState`][magnetofluidics_pinn.types.ParticleState]
      per particle, at the initial time.
    - `magnetic_moment`: Tensor of shape `(n_particles, n_dims)` or
      `(n_dims,)` with each particle's magnetic moment.
    - `time_span`: Tuple `(t_start, t_end)` bounding the integration.
    - `n_steps`: Number of integration steps.

    Returns:
    - A list of per-particle trajectories, each a list of
      [`ParticleState`][magnetofluidics_pinn.types.ParticleState] instances
      ordered by increasing time.

    Raises:
    - `ValueError`: If `initial_states` is empty, if `n_steps` is not
      strictly positive, if `time_span` is not increasing, or if
      `magnetic_moment` cannot be matched to each particle's coordinate
      dimensionality.
    """
    if not initial_states:
        raise ValueError("initial_states must contain at least one particle.")
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive.")
    time_start, time_end = time_span
    if time_end <= time_start:
        raise ValueError(f"time_span must satisfy t_end > t_start; got {time_span!r}.")

    step_size = (time_end - time_start) / n_steps
    target_device, target_dtype = _get_module_device_and_dtype(flow_network)
    moment_tensor = torch.as_tensor(
        magnetic_moment, dtype=target_dtype, device=target_device
        )

    trajectories: list[list[ParticleState]] = []
    for particle_index, initial_state in enumerate(initial_states):
        init_pos = initial_state.position.to(device=target_device,
                                             dtype=target_dtype)
        init_vel = initial_state.velocity.to(device=target_device,
                                             dtype=target_dtype)
        n_dims = initial_state.position.shape[-1]
        moment = _select_moment(moment_tensor, particle_index, n_dims).to(
            dtype=target_dtype, device=target_device
        )

        state = ParticleState(
            position=init_pos, velocity=init_vel, time=initial_state.time
        )
        history = [state]
        for _ in range(n_steps):
            position = state.position.unsqueeze(0)

            # Fourth-order Runge-Kutta integration of dx/dt = v_drift(x): a
            # standard, fixed-step choice when derivative evaluations
            # (a network forward pass plus a small autograd call) are
            # comparatively cheap.
            k1 = _drift_velocity(flow_network, field_fn, position, moment)
            k2 = _drift_velocity(flow_network, field_fn,
                                 position + 0.5 * step_size * k1, moment)
            k3 = _drift_velocity(flow_network, field_fn,
                                 position + 0.5 * step_size * k2, moment)
            k4 = _drift_velocity(flow_network, field_fn,
                                 position + step_size * k3, moment)

            next_position = (
                position + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            ).squeeze(0)
            next_time = state.time + step_size
            next_velocity = _drift_velocity(
                flow_network, field_fn, next_position.unsqueeze(0), moment
            ).squeeze(0)

            state = ParticleState(position=next_position,
                                  velocity=next_velocity,
                                  time=next_time)
            history.append(state)

        trajectories.append(history)

    return trajectories
