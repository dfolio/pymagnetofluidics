"""Plotting functions for flow fields, particle trajectories, and verification metrics.

Each function returns a `matplotlib.figure.Figure` instance instead of
calling `plt.show()` internally, leaving display or saving decisions to the
caller.
"""

from __future__ import annotations

import matplotlib.figure
# REMOVED: import matplotlib.pyplot as plt  <-- Avoid global state machine side effects
import numpy as np
import torch
from torch import nn

from magnetofluidics_pinn.device_utils import resolve_module_device
from magnetofluidics_pinn.training.trainer import (LOSS_COMPONENT_NAMES as LOSS_COMPONENT_NAMES,
                                                   TrainingHistory as TrainingHistory)
from magnetofluidics_pinn.types import Domain, ParticleState
# NEW: verification-metrics plotting functions consume `ProfileEvaluation`,
# the shared data container `verification.metrics.evaluate_velocity_profiles`
# produces, so that a trained network is evaluated on the profile grid once
# regardless of how many plots are derived from it.
from magnetofluidics_pinn.verification.metrics import ProfileEvaluation


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
    
    # CHANGED: Use pure OO API instead of plt.subplots() to prevent global side-effects
    figure = matplotlib.figure.Figure(figsize=(8.0, 4.0))
    axes = figure.subplots()
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
        
    # CHANGED: Use pure OO API
    figure = matplotlib.figure.Figure(figsize=(8.0, 4.0))
    axes = figure.subplots()
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
    
    adam_steps = np.asarray(history.adam.step, dtype=np.int64)
    lbfgs_steps = np.asarray(history.lbfgs.step, dtype=np.int64) + n_adam
    combined_steps = np.concatenate([adam_steps, lbfgs_steps]) if n_lbfgs > 0 else adam_steps
    
    def merge_series(field: str) -> np.ndarray:
        adam_vals = getattr(history.adam, field)
        lbfgs_vals = getattr(history.lbfgs, field)
        return (
            np.concatenate([np.asarray(adam_vals), np.asarray(lbfgs_vals)])
            if n_lbfgs > 0 else np.asarray(adam_vals)
        )
    
    panel_titles: dict[str, str] = {
        "total"     : "Composite Total Loss", "momentum_r": "Radial-Momentum Residual",
        "momentum_z": "Axial-Momentum Residual", "continuity": "Continuity (Mass-Conservation) Residual",
        "wall"      : "Wall No-Slip Loss", "inlet": "Inlet Velocity Loss", "outlet": "Outlet Pressure Loss",
        "positivity": "Axial-Velocity Positivity Penalty", "conservation": "Global Flow-Rate Conservation Loss",
    }
    palette = ("black", "tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown",
               "tab:pink")
    field_order = ("total",) + LOSS_COMPONENT_NAMES
    loss_metrics = tuple(
        (panel_titles[field], merge_series(field), color, "--" if field == "total" else "-")
        for field, color in zip(field_order, palette)
    )
    n_panels = len(loss_metrics) + 1
    n_cols = 3
    n_rows = np.ceil(n_panels / n_cols).astype(int)
    
    n_panels = len(loss_metrics) + 1
    n_cols = 3
    n_rows = np.ceil(n_panels / n_cols).astype(int)
    
    # CHANGED: Replaced plt.subplots with pure OO Figure initialization to avoid global state
    fig = matplotlib.figure.Figure(figsize=(5.0 * n_cols, 3.6 * n_rows))
    axes = fig.subplots(n_rows, n_cols)
    
    # axes is a NumPy array here, so flatten() behaves exactly as before
    axes_flat = axes.flatten()
    
    def render_loss_axis(ax, title, series, color, linestyle) -> None:
        ax.plot(combined_steps, series, color=color, linestyle=linestyle, linewidth=1.5, label=title)
        if n_lbfgs > 0 and n_adam > 0:
            ax.axvline(x=n_adam, color="tab:red", linestyle=":", linewidth=1.2,
                       label="Adam -> L-BFGS" if ax is axes_flat[0] else None)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Optimization Step (Adam Epochs + L-BFGS Iterations)")
        ax.set_ylabel("Loss Magnitude")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
    
    for _ax, (_title, _series, _col, _ls) in zip(axes_flat[: len(loss_metrics)], loss_metrics):
        render_loss_axis(_ax, _title, _series, _col, _ls)
    overlay_ax = axes_flat[len(loss_metrics)]
    
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
    
    for _ax in axes_flat[n_panels:]:
        _ax.axis("off")
    
    fig.suptitle("PINN Loss Convergence Dynamics", fontsize=13, y=1.00)
    fig.tight_layout()
    return fig


# ============================================================================
# NEW: verification-metrics plotting functions (added alongside
# `magnetofluidics_pinn.verification` — see that subpackage's module
# docstrings for the metrics each of these visualizes). Each function is a
# pure renderer: it consumes already-computed data and returns a `Figure`,
# following exactly the same convention as `plot_streamlines` and
# `plot_training_history` above.
# ============================================================================

def plot_velocity_profile_comparison(profile_evaluation: ProfileEvaluation) -> matplotlib.figure.Figure:
    """Plot predicted axial-velocity profiles against the analytical Hagen-Poiseuille profile.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].

    Returns:
    - A `matplotlib.figure.Figure` overlaying one predicted $u_z(r)$ curve
      per axial station on top of the shared analytical profile.

    Raises:
    - `ValueError`: If `profile_evaluation` holds no axial stations.
    """
    if not profile_evaluation.z_stations:
        raise ValueError("profile_evaluation must contain at least one z station.")
    
    radial_grid = profile_evaluation.radial_grid.cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(
            radial_grid, profile_evaluation.predicted_uz[station_index].cpu().numpy(),
            marker="o", markersize=2.5, linewidth=1.0, label=f"Predicted, z={z_station:.2e}",
        )
    axes.plot(
        radial_grid, profile_evaluation.analytical_uz.cpu().numpy(),
        color="black", linestyle="--", linewidth=1.5, label="Analytical (Hagen-Poiseuille)",
    )
    axes.set_xlabel("Radial position r")
    axes.set_ylabel("Axial velocity u_z")
    axes.set_title("Predicted vs. analytical axial-velocity profiles")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_profile_error(profile_evaluation: ProfileEvaluation) -> matplotlib.figure.Figure:
    r"""Plot the axial-velocity profile error against the analytical solution.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].

    Returns:
    - A `matplotlib.figure.Figure` with one error curve
      $u_z^\\text{predicted}(r) - u_z^*(r)$ per axial station.

    Raises:
    - `ValueError`: If `profile_evaluation` holds no axial stations.
    """
    if not profile_evaluation.z_stations:
        raise ValueError("profile_evaluation must contain at least one z station.")
    
    radial_grid = profile_evaluation.radial_grid.cpu().numpy()
    error = (profile_evaluation.predicted_uz - profile_evaluation.analytical_uz.unsqueeze(0)).cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(radial_grid, error[station_index], linewidth=1.0, label=f"z={z_station:.2e}")
    axes.axhline(0.0, color="black", linewidth=0.75)
    axes.set_xlabel("Radial position r")
    axes.set_ylabel("u_z error (predicted - analytical)")
    axes.set_title("Axial-velocity profile error")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_radial_leakage(profile_evaluation: ProfileEvaluation) -> matplotlib.figure.Figure:
    r"""Plot the predicted radial-velocity (leakage) profile at each axial station.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].

    Returns:
    - A `matplotlib.figure.Figure` with one $u_r(r)$ curve per axial
      station; a perfectly mass-conserving field would show this
      identically zero.

    Raises:
    - `ValueError`: If `profile_evaluation` holds no axial stations.
    """
    if not profile_evaluation.z_stations:
        raise ValueError("profile_evaluation must contain at least one z station.")
    
    radial_grid = profile_evaluation.radial_grid.cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(
            radial_grid, profile_evaluation.predicted_ur[station_index].cpu().numpy(),
            linewidth=1.0, label=f"z={z_station:.2e}",
        )
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel("Radial position r")
    axes.set_ylabel("Radial velocity u_r")
    axes.set_title("Radial leakage across the domain")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_flow_rate_deviation(
        axial_positions: torch.Tensor, flow_rate_curve: torch.Tensor, reference_flow_rate: float,
) -> matplotlib.figure.Figure:
    r"""Plot the predicted flow rate's deviation from its analytical reference.

    Args:
    - `axial_positions`: Tensor of shape `(n_stations,)`, the axial
      stations $Q(z)$ was evaluated at.
    - `flow_rate_curve`: Tensor of shape `(n_stations,)`, e.g. from
      [`verification.metrics.compute_flow_rate_curve`][magnetofluidics_pinn.verification.metrics.compute_flow_rate_curve].
    - `reference_flow_rate`: The analytical reference flow rate
      $Q_\\text{ref}$, e.g. from
      [`physics.conservation.poiseuille_reference_flow_rate`][magnetofluidics_pinn.physics.conservation.poiseuille_reference_flow_rate].

    Returns:
    - A `matplotlib.figure.Figure` plotting $Q(z) - Q_\\text{ref}$ against
      $z$.

    Raises:
    - `ValueError`: If `axial_positions` and `flow_rate_curve` do not have
      the same shape, or if `reference_flow_rate` is not strictly positive.
    """
    if axial_positions.shape != flow_rate_curve.shape:
        raise ValueError(
            "axial_positions and flow_rate_curve must have the same shape; got "
            f"{tuple(axial_positions.shape)} and {tuple(flow_rate_curve.shape)}."
        )
    if reference_flow_rate <= 0.0:
        raise ValueError(f"reference_flow_rate must be strictly positive; got {reference_flow_rate!r}.")
    
    z_values = axial_positions.detach().cpu().numpy()
    deviation = (flow_rate_curve.detach() - reference_flow_rate).cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    axes.plot(z_values, deviation, marker="o", markersize=3.0, linewidth=1.0, color="tab:red")
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel("Axial position z")
    axes.set_ylabel("Q(z) - Q_ref")
    axes.set_title("Global flow-rate deviation from the analytical reference")
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_pressure_gradient_fit(
        axial_positions: np.ndarray,
        predicted_pressure: np.ndarray,
        fitted_slope: float,
        fitted_intercept: float,
        analytical_slope: float,
        reference_pressure_at_outlet: float,
        domain_length: float,
) -> matplotlib.figure.Figure:
    """Plot the network's predicted axial pressure profile against its fit and the analytical gradient.

    Args:
    - `axial_positions`, `predicted_pressure`: Arrays of shape
      `(n_axial_points,)`, e.g. from
      [`verification.metrics.evaluate_axis_pressure_profile`][magnetofluidics_pinn.verification.metrics.evaluate_axis_pressure_profile].
    - `fitted_slope`, `fitted_intercept`: The linear fit's coefficients,
      e.g. from
      [`verification.metrics.compute_pressure_gradient_error`][magnetofluidics_pinn.verification.metrics.compute_pressure_gradient_error]'s
      `"predicted_pressure_gradient"` and `"fitted_intercept"` entries.
    - `analytical_slope`: The analytical pressure gradient, e.g. from
      [`verification.analytical.analytical_pressure_gradient`][magnetofluidics_pinn.verification.analytical.analytical_pressure_gradient].
    - `reference_pressure_at_outlet`: The outlet gauge pressure the
      analytical line is anchored to.
    - `domain_length`: Dimensionless channel length, where the analytical
      line's gauge is anchored.

    Returns:
    - A `matplotlib.figure.Figure` with the predicted pressure samples, the
      fitted line, and the analytical line.

    Raises:
    - `ValueError`: If `axial_positions` and `predicted_pressure` do not
      have the same shape.
    """
    if axial_positions.shape != predicted_pressure.shape:
        raise ValueError(
            "axial_positions and predicted_pressure must have the same shape; got "
            f"{axial_positions.shape} and {predicted_pressure.shape}."
        )
    analytical_pressure = reference_pressure_at_outlet + analytical_slope * (axial_positions - domain_length)
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    axes.scatter(axial_positions, predicted_pressure, s=10.0, color="tab:blue", label="Predicted p(r=0, z)")
    axes.plot(
        axial_positions, fitted_slope * axial_positions + fitted_intercept,
        color="tab:orange", linewidth=1.2, label="Linear fit",
    )
    axes.plot(axial_positions, analytical_pressure, color="black", linestyle="--", linewidth=1.2, label="Analytical")
    axes.set_xlabel("Axial position z")
    axes.set_ylabel("Pressure p (r=0)")
    axes.set_title("Axial pressure profile: predicted fit vs. analytical")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_residual_heatmap(
        radial_grid: torch.Tensor, axial_grid: torch.Tensor, residual_magnitude: torch.Tensor, component_name: str,
) -> matplotlib.figure.Figure:
    """Plot a held-out PDE residual's magnitude as a heatmap over the (r, z) domain.

    Args:
    - `radial_grid`, `axial_grid`: 1-D tensors of shape `(n_radial,)` and
      `(n_axial,)`, the grid `residual_magnitude` was evaluated over.
    - `residual_magnitude`: Tensor of shape `(n_radial, n_axial)`, e.g. one
      entry of
      [`verification.metrics.evaluate_residual_grid`][magnetofluidics_pinn.verification.metrics.evaluate_residual_grid]'s
      return value.
    - `component_name`: Name of the residual component being plotted (used
      in the title and colorbar label only).

    Returns:
    - A `matplotlib.figure.Figure` with a pseudocolor mesh of
      `residual_magnitude`.

    Raises:
    - `ValueError`: If `residual_magnitude`'s shape does not match
      `(radial_grid.numel(), axial_grid.numel())`.
    """
    if residual_magnitude.shape != (radial_grid.shape[0], axial_grid.shape[0]):
        raise ValueError(
            f"residual_magnitude must have shape ({radial_grid.shape[0]}, {axial_grid.shape[0]}); "
            f"got {tuple(residual_magnitude.shape)}."
        )
    figure = matplotlib.figure.Figure(figsize=(7.0, 3.5))
    axes = figure.subplots()
    mesh = axes.pcolormesh(
        axial_grid.cpu().numpy(), radial_grid.cpu().numpy(), residual_magnitude.detach().cpu().numpy(),
        shading="auto", cmap="magma",
    )
    figure.colorbar(mesh, ax=axes, label=f"|{component_name}| residual")
    axes.set_xlabel("Axial position z")
    axes.set_ylabel("Radial position r")
    axes.set_title(f"Held-out {component_name} residual magnitude")
    figure.tight_layout()
    return figure
