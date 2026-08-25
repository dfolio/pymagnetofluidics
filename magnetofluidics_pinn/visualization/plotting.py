"""Plotting functions for flow fields and particle trajectories.

Each function returns a `matplotlib.figure.Figure` instance instead of
calling `plt.show()` internally, leaving display or saving decisions to the
caller.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from magnetofluidics_pinn.device_utils import resolve_module_device
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
    - `NotImplementedError`: If `domain.kind` is not `"channel"`.
    """
    if resolution <= 0:
        raise ValueError("resolution must be strictly positive.")
    if domain.kind != "channel":
        raise NotImplementedError(
            f"Streamline plotting currently only supports domain.kind == 'channel'; got {domain.kind!r}."
        )

    network_device = resolve_module_device(flow_network)
    radial_axis = torch.linspace(0.0, domain.radius, resolution, device=network_device)
    axial_axis = torch.linspace(0.0, domain.length, resolution, device=network_device)
    radial_grid, axial_grid = torch.meshgrid(radial_axis, axial_axis, indexing="ij")
    grid_points = torch.stack([radial_grid.reshape(-1), axial_grid.reshape(-1)], dim=1)

    with torch.no_grad():
        prediction = flow_network(grid_points)
    velocity_r = prediction[:, 0].reshape(resolution, resolution).cpu().numpy()
    velocity_z = prediction[:, 1].reshape(resolution, resolution).cpu().numpy()

    figure, axes = plt.subplots(figsize=(8.0, 4.0))
    axes.streamplot(
        axial_axis.cpu().numpy(),
        radial_axis.cpu().numpy(),
        velocity_z,
        velocity_r,
        density=1.2,
        color="steelblue",
    )
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel("Axial position z")
    axes.set_ylabel("Radial position r")
    axes.set_title("Predicted flow streamlines")
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius)
    figure.tight_layout()
    return figure


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
    - `NotImplementedError`: If `domain.kind` is not `"channel"`.
    """
    if not trajectories:
        raise ValueError("trajectories must contain at least one particle path.")
    if domain.kind != "channel":
        raise NotImplementedError(
            f"Trajectory plotting currently only supports domain.kind == 'channel'; got {domain.kind!r}."
        )

    figure, axes = plt.subplots(figsize=(8.0, 4.0))
    for trajectory in trajectories:
        radial_positions = np.array([state.position[0].item() for state in trajectory])
        axial_positions = np.array([state.position[1].item() for state in trajectory])
        axes.plot(axial_positions, radial_positions, marker="o", markersize=2.0)

    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel("Axial position z")
    axes.set_ylabel("Radial position r")
    axes.set_title("Particle trajectories")
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius)
    figure.tight_layout()
    return figure
