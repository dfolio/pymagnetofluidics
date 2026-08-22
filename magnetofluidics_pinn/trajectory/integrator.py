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

from magnetofluidics_pinn.types import FieldSample, ParticleState


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
    - `ValueError`: If `initial_states` is empty or `n_steps` is not strictly
      positive.
    """
    if not initial_states:
        raise ValueError("initial_states must contain at least one particle.")
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive.")
    raise NotImplementedError("Implementation scheduled for Step 2.")
