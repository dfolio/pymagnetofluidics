"""Plotting functions for flow fields and particle trajectories.

Each function returns a `matplotlib.figure.Figure` instance instead of
calling `plt.show()` internally, leaving display or saving decisions to the
caller.
"""

from __future__ import annotations

import matplotlib.figure
from torch import nn

from magnetofluidics_pinn.types import Domain, ParticleState


def plot_streamlines(
    flow_network: nn.Module, domain: Domain, resolution: int = 200
) -> matplotlib.figure.Figure:
    """Plot the flow streamlines predicted by a trained network.

    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure.
    - `domain`: Vessel geometry the field is evaluated over.
    - `resolution`: Number of grid points per axis used for evaluation.

    Returns:
    - A `matplotlib.figure.Figure` showing the streamlines over the domain.

    Raises:
    - `ValueError`: If `resolution` is not strictly positive.
    """
    if resolution <= 0:
        raise ValueError("resolution must be strictly positive.")
    raise NotImplementedError("Implementation scheduled for Step 2.")


def plot_trajectories(
    trajectories: list[list[ParticleState]], domain: Domain
) -> matplotlib.figure.Figure:
    """Plot one or several particle trajectories over the domain.

    Args:
    - `trajectories`: Per-particle list of
      [`ParticleState`][magnetofluidics_pinn.types.ParticleState] instances,
      as returned by
      [`integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory].
    - `domain`: Vessel geometry the trajectories are drawn over.

    Returns:
    - A `matplotlib.figure.Figure` showing the domain outline and every
      particle path.

    Raises:
    - `ValueError`: If `trajectories` is empty.
    """
    if not trajectories:
        raise ValueError("trajectories must contain at least one particle path.")
    raise NotImplementedError("Implementation scheduled for Step 2.")
