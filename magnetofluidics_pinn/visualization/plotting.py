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

from magnetofluidics_pinn.training.trainer import TrainingHistory as TrainingHistory

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

# CHANGE: Added functional visualization and summary routine for TrainingHistory
def plot_training_history(
    history: TrainingHistory,
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Plot loss convergence histories across the Adam and L-BFGS training phases.

    Concatenates Adam epoch steps with L-BFGS evaluation steps, drawing a stage
    demarcation boundary if L-BFGS refinement was executed.

    Args:
    - `history`: A `TrainingHistory` instance containing `.adam` and `.lbfgs`
      `LossHistory` tuples.
    - `log_scale`: Whether to set the y-axis of each subplot to a logarithmic
      scale. Defaults to `True`.

    Returns:
    - A `matplotlib.figure.Figure` instance containing the multi-panel loss curves.

    Raises:
    - `ValueError`: If both Adam and L-BFGS histories are empty.
    """
    n_adam = len(history.adam.step)
    n_lbfgs = len(history.lbfgs.step)

    if n_adam == 0 and n_lbfgs == 0:
        raise ValueError("Cannot plot empty TrainingHistory: both phases contain zero steps.")

    # Construct continuous step indexing across both optimization stages
    adam_steps = np.asarray(history.adam.step, dtype=np.int64)
    lbfgs_steps = np.asarray(history.lbfgs.step, dtype=np.int64) + n_adam
    combined_steps = np.concatenate([adam_steps, lbfgs_steps]) if n_lbfgs > 0 else adam_steps

    def merge_series(field: str) -> np.ndarray:
        """Pure helper to concatenate series across training phases."""
        adam_vals = getattr(history.adam, field)
        lbfgs_vals = getattr(history.lbfgs, field)
        return (
            np.concatenate([np.asarray(adam_vals), np.asarray(lbfgs_vals)])
            if n_lbfgs > 0
            else np.asarray(adam_vals)
        )

    loss_metrics = (
        ("Composite Total Loss", merge_series("total"), "black", "--"),
        ("Stokes PDE Residual", merge_series("pde"), "tab:blue", "-"),
        ("Wall No-Slip Loss", merge_series("wall"), "tab:orange", "-"),
        ("Inlet Velocity Loss", merge_series("inlet"), "tab:green", "-"),
        ("Outlet Pressure Loss", merge_series("outlet"), "tab:purple", "-"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    def render_loss_axis(
        ax: plt.Axes,
        title: str,
        series: np.ndarray,
        color: str,
        linestyle: str,
    ) -> None:
        """Renders an individual loss curve with optional stage demarcation."""
        ax.plot(combined_steps, series, color=color, linestyle=linestyle, linewidth=1.5, label=title)
        if n_lbfgs > 0 and n_adam > 0:
            ax.axvline(
                x=n_adam,
                color="tab:red",
                linestyle=":",
                linewidth=1.2,
                label="Adam → L-BFGS" if ax is axes_flat[0] else None,
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Optimization Step (Adam Epochs + L-BFGS Iterations)")
        ax.set_ylabel("Loss Magnitude")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    # Render panels 1-5 for individual loss metrics
    for _ax, (_title, _series, _col, _ls) in zip(axes_flat[:5], loss_metrics):
        render_loss_axis(_ax, _title, _series, _col, _ls)

    # Panel 6: Multi-loss overlay comparison
    overlay_ax = axes_flat[5]
    for _title, _series, _col, _ls in loss_metrics:
        overlay_ax.plot(combined_steps, _series, color=_col, linestyle=_ls, linewidth=1.2, label=_title)
    if n_lbfgs > 0 and n_adam > 0:
        overlay_ax.axvline(x=n_adam, color="tab:red", linestyle=":", linewidth=1.2)
    if log_scale:
        overlay_ax.set_yscale("log")
    overlay_ax.set_xlabel("Optimization Step")
    overlay_ax.set_ylabel("Loss Magnitude")
    overlay_ax.set_title("All Losses Overlaid")
    overlay_ax.grid(True, linestyle=":", alpha=0.6)
    overlay_ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("PINN Loss Convergence Dynamics", fontsize=13, y=1.00)
    fig.tight_layout()
    return fig


