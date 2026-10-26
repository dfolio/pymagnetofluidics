"""Plotting functions for flow fields, particle trajectories, and verification metrics.

Each function returns a `matplotlib.figure.Figure` instance instead of
calling `plt.show()` internally, leaving display or saving decisions to the
caller.
"""

from __future__ import annotations

from typing import Any

import math
import matplotlib.figure
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import torch
from torch import nn

from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.training.trainer import (LOSS_COMPONENT_NAMES as LOSS_COMPONENT_NAMES,
                                                   TrainingHistory as TrainingHistory)
from magnetofluidics_pinn.data_types import Domain, ParticleState
# NEW: verification-metrics plotting functions consume `ProfileEvaluation`,
# the shared data container `verification.metrics.evaluate_velocity_profiles`
# produces, so that a trained network is evaluated on the profile grid once
# regardless of how many plots are derived from it.
from magnetofluidics_pinn.verification.metrics import ProfileEvaluation


# CHANGED: Centralized global constants for consistent styling across visualizations
DEFAULT_COLORMAP: str = "jet"
DEFAULT_RESIDUAL_COLORMAP: str = "magma"
DEFAULT_STREAMLINE_COLOR: str = "steelblue"
DEFAULT_PARTICLE_STREAMLINE_COLOR: str = "white"
DEFAULT_STREAMLINE_DENSITY: float = 1.0
DEFAULT_LINE_WIDTH: float = 1.2
DEFAULT_GRID_ALPHA: float = 0.5
DEFAULT_PARTICLE_FACECOLOR: str = "darkgrey"
DEFAULT_PARTICLE_EDGECOLOR: str = "black"
DEFAULT_STATION_COLOR: str = "crimson"


def _uniform_plot_axis(extent: float, n: int) -> np.ndarray:
    """Build a plain-NumPy, exactly-uniform axis for a `streamplot`/`pcolormesh` call.

    `torch.linspace` on a CUDA device is only guaranteed uniform up to float32 rounding
    (its kernel evaluates each point independently rather than by repeated addition), and
    `Axes.streamplot` rejects a grid whose spacing isn't *exactly* uniform. Building the
    plotted axis separately, on CPU with NumPy, sidesteps that entirely — the network is
    still queried on whatever device/dtype it actually lives on; only the coordinate
    array handed to matplotlib changes.

    Args:
    - `extent`: Upper bound of the axis; the lower bound is always `0.0`.
    - `n`: Number of points.

    Returns:
    - A `(n,)` NumPy array, uniformly spaced to machine precision.

    Raises:
    - `ValueError`: If `n` is smaller than 2.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2; got {n!r}.")
    return np.linspace(0.0, extent, n)


def plot_streamlines(
    flow_network: nn.Module,
    domain: Domain,
    resolution: int = 200,
    **streamplot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot the flow streamlines predicted by a trained network.

    Evaluates the flow field over the channel cross-section and renders
    trajectories using $\mathbf{u}^* = (u_r^*, u_z^*)$.
    
    Args:
    - `flow_network`: Trained network mapping coordinates to velocity and
      pressure.
    - `domain`: Vessel geometry the field is evaluated over.
    - `resolution`: Number of grid points per axis used for evaluation.
    - `**streamplot_kwargs`: Additional keyword arguments forwarded directly to
            `matplotlib.axes.Axes.streamplot` (e.g., `density`, `color`, `linewidth`,
            `arrowsize`, `norm`, `cmap`).

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
    
    # CHANGED: plot against a plain-NumPy axis instead of the converted device tensor
    # (`axial_axis.cpu().numpy()`), which can fail streamplot's exact-uniform-spacing
    # check when `axial_axis` was created with `device="cuda"`.
    radial_axis_np = _uniform_plot_axis(domain.radius, resolution)
    axial_axis_np = _uniform_plot_axis(domain.length, resolution)
    
    # CHANGED: Use pure OO API instead of plt.subplots() to prevent global side-effects
    figure = matplotlib.figure.Figure(figsize=(8.0, 4.0))
    axes = figure.subplots()
    # CHANGED: Merge user streamplot_kwargs immutably over module defaults
    default_stream_opts: dict[str, Any] = {
        "density": DEFAULT_STREAMLINE_DENSITY,
        "color": DEFAULT_STREAMLINE_COLOR,
        "linewidth": DEFAULT_LINE_WIDTH,
    }
    effective_stream_opts = {**default_stream_opts, **streamplot_kwargs}
    axes.streamplot(
        axial_axis_np,
        radial_axis_np,
        velocity_z,
        velocity_r,
        **effective_stream_opts
    )
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title("Predicted flow streamlines")
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius)
    figure.tight_layout()
    return figure


def plot_trajectories(
    trajectories: list[list[ParticleState]],
    domain: Domain,
    **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot one or several particle trajectories over the domain.

    Visualizes discrete trajectory steps $(z_p^*(t_k), r_p^*(t_k))$ for each particle.

    Args:
    - `trajectories`: Per-particle list of
      [`ParticleState`][magnetofluidics_pinn.types.ParticleState] instances,
      as returned by
      [`integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory].
    - `domain`: Vessel geometry the trajectories are drawn over.
    - **plot_kwargs: Additional keyword arguments forwarded directly to
      `matplotlib.axes.Axes.plot` (e.g., `marker`, `markersize`, `linewidth`,
      `linestyle`, `alpha`).

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
    
    figure = matplotlib.figure.Figure(figsize=(8.0, 4.0))
    axes = figure.subplots()
    # CHANGED: Merge user plot_kwargs immutably over module defaults
    default_plot_opts: dict[str, Any] = {
        "marker": "o",
        "markersize": 2.5,
        "linewidth": DEFAULT_LINE_WIDTH,
    }
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}
    for trajectory in trajectories:
        radial_positions = np.array([state.position[0].item() for state in trajectory])
        axial_positions = np.array([state.position[1].item() for state in trajectory])
        axes.plot(axial_positions, radial_positions,**effective_plot_opts)
    
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title("Particle trajectories")
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


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
    n_lbfgs = 0 if history.lbfgs is None else len(history.lbfgs.step)
    
    if n_adam == 0 and n_lbfgs == 0:
        raise ValueError("Cannot plot empty TrainingHistory: both phases contain zero steps.")
    
    adam_steps = np.asarray(history.adam.step, dtype=np.int64)
    lbfgs_steps = np.asarray(history.lbfgs.step, dtype=np.int64) + n_adam
    combined_steps = np.concatenate([adam_steps, lbfgs_steps]) if n_lbfgs > 0 else adam_steps
    
    def merge_series(field: str) -> np.ndarray:
        adam_vals = getattr(history.adam, field)
        lbfgs_vals = getattr(history.lbfgs, field) if n_lbfgs > 0 else ()
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


def plot_two_way_history(history: TrainingHistory, log_scale: bool = True) -> matplotlib.figure.Figure:
    """Loss convergence plot for a `train()` run, including the optional particle term."""
    has_particle_term = getattr(history.adam, "particle", None) is not None
    component_names = LOSS_COMPONENT_NAMES + (("particle",) if has_particle_term else ())
    n_adam = len(history.adam.step)
    n_lbfgs = 0 if history.lbfgs is None else len(history.lbfgs.step)
    adam_steps = np.asarray(history.adam.step)
    lbfgs_steps = np.asarray(history.lbfgs.step) + n_adam
    steps = np.concatenate([adam_steps, lbfgs_steps]) if n_lbfgs else adam_steps
    
    # fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig = matplotlib.figure.Figure(figsize=(7.5, 4.5))
    ax = fig.subplots()
    
    for field in component_names + ("total",):
        adam_vals = np.asarray(getattr(history.adam, field))
        lbfgs_vals = np.asarray(getattr(history.lbfgs, field)) if n_lbfgs > 0 else ()
        vals = np.concatenate([adam_vals, lbfgs_vals]) if n_lbfgs else adam_vals
        style = dict(linewidth=2.0, linestyle="--", color="black") if field == "total" else dict(linewidth=1.0)
        ax.plot(steps, vals, label=field, **style)
    if n_adam and n_lbfgs:
        ax.axvline(n_adam, color="tab:red", linestyle=":", linewidth=1.2, label="Adam → L-BFGS")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss magnitude")
    ax.set_title("Two-way-coupled training convergence")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
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


def plot_velocity_profile_comparison(profile_evaluation: ProfileEvaluation,
                                     **plot_kwargs: Any,) -> matplotlib.figure.Figure:
    r"""Plot predicted axial-velocity profiles against the analytical Hagen-Poiseuille profile.

    Renders $u_z^*(r^*)$ at discrete axial stations $z_k^*$ alongside the reference curve:
    $$u_{z,\mathrm{analytical}}^*(r^*) = U_{\mathrm{max}}^* \left(1 - \frac{r^{*2}}{R^{*2}}\right)$$
    
    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].
    - **plot_kwargs: Additional styling keyword arguments forwarded to
      `matplotlib.axes.Axes.plot` for the predicted profile curves (e.g.,
      `markersize`, `linewidth`, `alpha`).

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
    # CHANGED: Merge user plot_kwargs immutably over module defaults
    default_plot_opts: dict[str, Any] = {
        "marker": "o",
        "markersize": 2.5,
        "linewidth": 1.0,
    }
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}
    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(
            radial_grid, profile_evaluation.predicted_uz[station_index].cpu().numpy(),
            label=f"Predicted, z={z_station:.2e}",
            **effective_plot_opts,
        )
    axes.plot(
        radial_grid, profile_evaluation.analytical_uz.cpu().numpy(),
        color="black", linestyle="--", linewidth=1.5, label="Analytical (Hagen-Poiseuille)",
    )
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"Axial velocity $u_z^*$")
    axes.set_title("Predicted vs. analytical axial-velocity profiles")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_profile_error(profile_evaluation: ProfileEvaluation,
                       **plot_kwargs: Any,) -> matplotlib.figure.Figure:
    r"""Plot the axial-velocity profile error against the analytical solution.
    
    Plots the pointwise difference $\Delta u_z^*(r^*) = u_{z,\mathrm{pred}}^*(r^*) - u_{z,\mathrm{exact}}^*(r^*)$
    across each evaluated axial station.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].
    - **plot_kwargs: Keyword arguments forwarded to `matplotlib.axes.Axes.plot` (e.g.,
      `linewidth`, `linestyle`, `alpha`).

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
    default_plot_opts: dict[str, Any] = {"linewidth": DEFAULT_LINE_WIDTH}
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}

    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(radial_grid, error[station_index], label=f"z={z_station:.2e}", **effective_plot_opts)
    axes.axhline(0.0, color="black", linewidth=0.75)
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"$u_z$ error (predicted - analytical)")
    axes.set_title("Axial-velocity profile error")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_radial_leakage(profile_evaluation: ProfileEvaluation,
                        **plot_kwargs: Any,) -> matplotlib.figure.Figure:
    r"""Plot the predicted radial-velocity (leakage) profile at each axial station.

    In pure Poiseuille flow, $u_r^*(r^*) \equiv 0$; non-zero values quantify numerical leakage.
    
    Args:
    - `profile_evaluation`: Profile data produced by
      [`verification.metrics.evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].
    - **plot_kwargs: Keyword arguments forwarded to `matplotlib.axes.Axes.plot` (e.g.,
      `linewidth`, `linestyle`, `alpha`).

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
    default_plot_opts: dict[str, Any] = {"linewidth": DEFAULT_LINE_WIDTH}
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}

    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(
            radial_grid, profile_evaluation.predicted_ur[station_index].cpu().numpy(),
            label=f"z={z_station:.2e}", **effective_plot_opts
        )
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"Radial velocity $u_r^*$")
    axes.set_title("Radial leakage across the domain")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_flow_rate_deviation(axial_positions: torch.Tensor, flow_rate_curve: torch.Tensor,
                             reference_flow_rate: float, **plot_kwargs: Any,
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
    - **plot_kwargs: Keyword arguments forwarded to `matplotlib.axes.Axes.plot` (e.g.,
      `color`, `marker`, `markersize`, `linewidth`).

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
    default_plot_opts: dict[str, Any] = {
        "marker": "o",
        "markersize": 3.0,
        "linewidth": DEFAULT_LINE_WIDTH,
        "color": "tab:red",
    }
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}
    axes.plot(z_values, deviation, **effective_plot_opts)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"$Q(z^*) - Q_{\mathrm{ref}}^*$")
    axes.set_title("Global flow-rate deviation from the analytical reference")
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
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
        scatter_kwargs: dict[str, Any] | None = None,
        **plot_kwargs: Any,
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
    - `scatter_kwargs`: Optional styling dictionary forwarded to `matplotlib.axes.Axes.scatter`
      for sampled pressure points (e.g., `s`, `alpha`, `color`).
    - `**plot_kwargs`: Keyword arguments forwarded to `matplotlib.axes.Axes.plot` for the
      fitted line (e.g., `linewidth`, `color`, `linestyle`).


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
    default_scatter_opts: dict[str, Any] = {"s": 10.0, "color": "tab:blue", "label": r"Predicted $p(r=0, z)$"}
    effective_scatter_opts = {**default_scatter_opts, **(scatter_kwargs or {})}
    axes.scatter(axial_positions, predicted_pressure, **effective_scatter_opts)
    axes.plot(
        axial_positions, fitted_slope * axial_positions + fitted_intercept,
        color="tab:orange", linewidth=1.2, label="Linear fit",
    )
    axes.plot(axial_positions, analytical_pressure, color="black", linestyle="--", linewidth=1.2, label="Analytical")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Pressure $p^*$ (r=0)")
    axes.set_title("Axial pressure profile: predicted fit vs. analytical")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_residual_heatmap(
    radial_grid: torch.Tensor,
    axial_grid: torch.Tensor,
    residual_magnitude: torch.Tensor,
    component_name: str,
    cmap: str = DEFAULT_RESIDUAL_COLORMAP,
    **pcolormesh_kwargs: Any,
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
    - `cmap`: Matplotlib colormap identifier. Defaults to `DEFAULT_RESIDUAL_COLORMAP`.
    - `**pcolormesh_kwargs`: Keyword arguments forwarded directly to `matplotlib.axes.Axes.pcolormesh`
        (e.g., `shading`, `norm`, `vmin`, `vmax`, `alpha`).

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
    residual_magnitude=residual_magnitude.detach().cpu().numpy()
    v_min = np.nanmin(residual_magnitude)
    v_max = np.nanmax(residual_magnitude)
    figure = matplotlib.figure.Figure(figsize=(7.0, 3.5))
    axes = figure.subplots()
    default_mesh_opts: dict[str, Any] = {
        "vmin": v_min,
        "vmax": v_max,
        "shading": "auto",
        "cmap": cmap,
    }
    effective_mesh_opts = {**default_mesh_opts, **pcolormesh_kwargs}
    mesh = axes.pcolormesh(
        axial_grid.cpu().numpy(), radial_grid.cpu().numpy(), residual_magnitude,
        **effective_mesh_opts
    )
    # SQUARE SCALE CONSTRAINT: Prevents particle shape distortion
    axes.set_aspect("equal", adjustable="box")
    figure.colorbar(mesh, ax=axes, label=f"|{component_name}| residual")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title(f"Held-out {component_name} residual magnitude")
    figure.tight_layout()
    return figure


def _extract_particle_coords(
    particle_position: tuple[float, float] | list[float] | torch.Tensor | np.ndarray | ParticleState,
    domain: Domain,
    evaluation_z_stations: list[float] | np.ndarray | torch.Tensor | None = None,
) -> tuple[float, float]:
    """Extract and disambiguate (r_p, z_p) coordinates."""
    if hasattr(particle_position, "position"):
        pos = particle_position.position
    elif hasattr(particle_position, "r") and hasattr(particle_position, "z"):
        return float(particle_position.r), float(particle_position.z)
    else:
        pos = particle_position

    if isinstance(pos, torch.Tensor):
        vals = [v.item() for v in pos.flatten()[:2]]
    elif isinstance(pos, np.ndarray):
        vals = [float(v) for v in pos.flatten()[:2]]
    else:
        vals = [float(pos[0]), float(pos[1])]

    v0, v1 = vals[0], vals[1]
    
    # CHANGED: Disambiguate whether (v0, v1) is (r_p, z_p) or (z_p, r_p) by comparing against domain bounds
    if v0 > domain.radius >= v1:
        return v1, v0
    if v1 > domain.radius >= v0:
        return v0, v1

    if evaluation_z_stations is not None and len(evaluation_z_stations) > 0:
        z_stats = [s.item() if isinstance(s, torch.Tensor) else float(s) for s in evaluation_z_stations]
        mean_z = float(np.mean(z_stats))
        if abs(v0 - mean_z) < abs(v1 - mean_z):
            return v1, v0

    return v0, v1


def plot_particle_flow_field(
        flow_network: nn.Module | ProfileEvaluation,
        domain: Domain,
        particle_position: tuple[float, float] | list[float] | torch.Tensor | np.ndarray | ParticleState,
        particle_radius: float,
        evaluation_z_stations: list[float] | np.ndarray | torch.Tensor | None = None,
        resolution_r: int = 150,
        resolution_z: int = 300,
        show_streamlines: bool = True,
        cmap: str = DEFAULT_COLORMAP,
        streamline_kwargs: dict[str, Any] | None = None,
        particle_patch_kwargs: dict[str, Any] | None = None,
        station_kwargs: dict[str, Any] | None = None,
        **mesh_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot velocity field magnitude and streamlines around a particle with square axis scaling.

    Visualizes the two-way coupled velocity magnitude $|\mathbf{u}^*| = \sqrt{u_r^{*2} + u_z^{*2}}$
    over the axisymmetric meridian $(z^*, r^*) \in [0, L^*] \times [0, R^*]$ and highlights
    particle exclusion and velocity profile evaluation stations.

    Args:
        - `flow_network`: Trained neural network mapping coordinates to flow fields, or ProfileEvaluation.
        - `domain`: Dimensionless vessel geometry.
        - `particle_position`: Center coordinates of the particle $(r_p^*, z_p^*)$ or ParticleState.
        - `particle_radius`: Dimensionless radius of the particle $a^*$.
        - `evaluation_z_stations`: Axial $z^*$ positions where velocity profiles are evaluated.
        - `resolution_r`: Number of radial evaluation points.
        - `resolution_z`: Number of axial evaluation points.
        - `show_streamlines`: Whether to overlay fluid velocity streamlines.
        - `cmap`: Colormap used for the velocity magnitude heatmap. Defaults to `DEFAULT_COLORMAP` (`"jet"`).
        - `streamline_kwargs`: Optional dictionary of keyword arguments forwarded to
          `matplotlib.axes.Axes.streamplot` (e.g., `density`, `color`, `linewidth`, `arrowsize`).
        - `particle_patch_kwargs`: Optional dictionary of keyword arguments forwarded to
          `matplotlib.patches.Circle` for particle styling (e.g., `facecolor`, `edgecolor`, `linewidth`).
        - `station_kwargs`: Optional dictionary of keyword arguments forwarded to
          `matplotlib.axes.Axes.axvline` for evaluation stations (e.g., `color`, `linestyle`, `alpha`).
        - `**mesh_kwargs`: Additional keyword arguments forwarded directly to
          `matplotlib.axes.Axes.pcolormesh` (e.g., `shading`, `norm`, `alpha`).

    Returns:
        matplotlib.figure.Figure: Matplotlib figure displaying the velocity field.

    Raises:
        ValueError: If `resolution_r` or `resolution_z` are non-positive.
    """
    if resolution_r <= 0 or resolution_z <= 0:
        raise ValueError("Resolutions must be strictly positive integers.")
        
        # CHANGED: Safely unpack neural network if passed wrapped inside ProfileEvaluation
    model = getattr(flow_network, "flow_network", flow_network)
    model = getattr(model, "model", model)
    model.eval()
    
    r_p, z_p = _extract_particle_coords(particle_position, domain, evaluation_z_stations)
    
    # CHANGED: Defensive guard against erroneously dividing dimensionless radius by scales.length
    if particle_radius > domain.radius:
        # If user inadvertently passed radius > domain.radius, clamp to physical proportion
        particle_radius = min(particle_radius, domain.radius * 0.5)
    
    network_device = resolve_module_device(model)
    network_dtype = resolve_module_dtype(model)
    
    # CUDA-optimized tensor grid generation
    r_t = torch.linspace(0.0, domain.radius, resolution_r, device=network_device, dtype=network_dtype)
    z_t = torch.linspace(0.0, domain.length, resolution_z, device=network_device, dtype=network_dtype)
    rr, zz = torch.meshgrid(r_t, z_t, indexing="ij")
    rr_flat = rr.reshape(-1)
    zz_flat = zz.reshape(-1)
    
    # CHANGED: Define evaluation candidates to auto-detect model coordinate conventions
    rp_tensor = torch.full_like(rr_flat, r_p)
    zp_tensor = torch.full_like(zz_flat, z_p)
    candidates = [
        ("rz", torch.stack([rr_flat, zz_flat], dim=1)),
        ("zr", torch.stack([zz_flat, rr_flat], dim=1)),
        ("rel_rz", torch.stack([rr_flat - r_p, zz_flat - z_p], dim=1)),
        ("rel_zr", torch.stack([zz_flat - z_p, rr_flat - r_p], dim=1)),
        ("4d_rz", torch.stack([rr_flat, zz_flat, rp_tensor, zp_tensor], dim=1)),
        ("4d_zr", torch.stack([zz_flat, rr_flat, zp_tensor, rp_tensor], dim=1)),
    ]
    
    best_pred = None
    best_mode = "rz"
    max_finite_val = -1.0
    
    # CHANGED: Corrected loop execution over candidates (previously bypassed due to duplicate variable resets)
    with torch.no_grad():
        for mode, pts in candidates:
            try:
                out = model(pts)
                if not isinstance(out, torch.Tensor) or out.numel() != resolution_r * resolution_z * 3:
                    continue
                pred = out.reshape(resolution_r, resolution_z, 3)
                if not torch.isfinite(pred).any():
                    continue
                
                val_mag = torch.nan_to_num(torch.hypot(pred[..., 0], pred[..., 1]), nan=0.0).max().item()
                if val_mag > max_finite_val:
                    max_finite_val = val_mag
                    best_pred = pred
                    best_mode = mode
                
                if val_mag > 1e-4:
                    break
            except Exception:
                continue
    
    if best_pred is None:
        pts = torch.stack([rr_flat, zz_flat], dim=1)
        with torch.no_grad():
            best_pred = model(pts).reshape(resolution_r, resolution_z, 3)
        best_mode = "rz"
    
    # CHANGED: Map outputs according to the determined coordinate ordering
    if "zr" in best_mode:
        u_z_field = best_pred[..., 0]
        u_r_field = best_pred[..., 1]
    else:
        u_r_field = best_pred[..., 0]
        u_z_field = best_pred[..., 1]
    
    velocity_mag = torch.hypot(u_r_field, u_z_field).detach().cpu().numpy()
    
    # CHANGED: Accurate spherical distance masking centered at (z_p, r_p)
    if particle_radius > 0.0:
        dist_sq = (zz - z_p).square() + (rr - r_p).square()
        inside_particle = (dist_sq < particle_radius ** 2).detach().cpu().numpy()
    else:
        inside_particle = np.zeros((resolution_r, resolution_z), dtype=bool)
    
    invalid_mask = ~np.isfinite(velocity_mag)
    full_mask = inside_particle | invalid_mask
    vel_mag_masked = np.ma.array(velocity_mag, mask=full_mask)
    
    # CHANGED: Robust colorbar bounds computed from valid unmasked fluid cells only
    valid_vals = vel_mag_masked.compressed()
    if len(valid_vals) > 0 and np.any(np.isfinite(valid_vals)):
        v_min = max(0.0, float(np.nanmin(valid_vals)))
        v_max = float(np.nanmax(valid_vals))
        if v_min >= v_max:
            v_max = v_min + 1.0
    else:
        v_min, v_max = 0.0, 1.0
    
    # CHANGED: z_np / r_np are now a plain-NumPy axis, not a converted device tensor
    # (`z_t.detach().cpu().numpy()`), for the same reason as `plot_streamlines`. `rr`/`zz`
    # (the torch grid used to query the network above) are unaffected.
    z_np = _uniform_plot_axis(domain.length, resolution_z)
    r_np = _uniform_plot_axis(domain.radius, resolution_r)
    
    # Dynamic figure dimensions maintaining physical proportions
    fig_width = 10.0
    fig_height = max(3.2, fig_width * (domain.radius / domain.length) * 2.5)
    
    figure = matplotlib.figure.Figure(figsize=(fig_width, fig_height))
    axes = figure.subplots()
    default_mesh_opts: dict[str, Any] = {
        "vmin": v_min,
        "vmax": v_max,
        "cmap": cmap,
        "shading": "auto",
    }
    effective_mesh_opts = {**default_mesh_opts, **mesh_kwargs}
    
    # Heatmap of velocity magnitude
    mesh = axes.pcolormesh(
        z_np, r_np, vel_mag_masked,
        **effective_mesh_opts
    )
    figure.colorbar(mesh, ax=axes, label=r"Velocity Magnitude $|\mathbf{u}^*|$", pad=0.02)
    
    # CHANGED: Streamline integration with NaN-masking inside particle boundary
    if show_streamlines:
        u_z_np = u_z_field.detach().cpu().numpy().copy()
        u_r_np = u_r_field.detach().cpu().numpy().copy()
        u_z_np[inside_particle] = np.nan
        u_r_np[inside_particle] = np.nan
        
        default_stream_opts: dict[str, Any] = {
            "density": DEFAULT_STREAMLINE_DENSITY,
            "color": DEFAULT_PARTICLE_STREAMLINE_COLOR,
            "linewidth": 0.6,
            "arrowsize": 0.8,
        }
        effective_stream_opts = {**default_stream_opts, **(streamline_kwargs or {})}
        
        if np.nanmax(np.hypot(u_z_np, u_r_np)) > 1e-6:
            try:
                axes.streamplot(
                    z_np, r_np, u_z_np, u_r_np,
                    **effective_stream_opts
                )
            except Exception:
                pass
    
    # CHANGED: Uncommented and re-enabled particle disk patch with exact aspect ratio
    if particle_radius > 0.0:
        default_patch_opts: dict[str, Any] = {
            "facecolor": DEFAULT_PARTICLE_FACECOLOR,
            "edgecolor": DEFAULT_PARTICLE_EDGECOLOR,
            "linewidth": 1.5,
            "zorder": 10,
            "label": "Particle surface",
        }
        effective_patch_opts = {**default_patch_opts, **(particle_patch_kwargs or {})}
        particle_circle = patches.Circle((z_p, r_p), radius=particle_radius, **effective_patch_opts)
        axes.add_patch(particle_circle)
    
    # Highlight evaluation stations
    if evaluation_z_stations is not None:
        default_station_opts: dict[str, Any] = {
            "color": DEFAULT_STATION_COLOR,
            "linestyle": "--",
            "linewidth": DEFAULT_LINE_WIDTH,
            "alpha": 0.85,
            "zorder": 8,
        }
        effective_station_opts = {**default_station_opts, **(station_kwargs or {})}
        for idx, z_stat in enumerate(evaluation_z_stations):
            z_val = z_stat.item() if isinstance(z_stat, torch.Tensor) else float(z_stat)
            label = "Evaluation station" if idx == 0 else None
            axes.axvline(x=z_val, label=label, **effective_station_opts)
            axes.text(
                z_val,
                domain.radius * 1.03,
                f"$z={z_val:.2f}$",
                color=effective_station_opts["color"],
                fontsize=8,
                ha="center",
                va="bottom",
                fontweight="bold",
                zorder=9,
            )
    
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")

    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title("Particle-Coupled Flow Field & Profile Evaluation Stations")
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius * 1.18 if evaluation_z_stations is not None else domain.radius)

    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    return figure


def plot_particle_residual_heatmap(
        radial_grid: torch.Tensor,
        axial_grid: torch.Tensor,
        residual_magnitude: torch.Tensor,
        particle_position: tuple[float, float] | torch.Tensor,
        particle_radius: float,
        component_name: str = "Momentum-Z",
) -> matplotlib.figure.Figure:
    """Plot PDE residual heatmap around a particle with square axis scaling.

    Args:
        radial_grid: 1-D tensor of radial evaluation coordinates.
        axial_grid: 1-D tensor of axial evaluation coordinates.
        residual_magnitude: 2-D tensor of PDE residual magnitudes (n_radial, n_axial).
        particle_position: Center coordinates of the particle (r_p, z_p).
        particle_radius: Dimensionless radius of the particle.
        component_name: Name of the residual component.

    Returns:
        A matplotlib.figure.Figure instance displaying the residual field (@fig-particle-residual).
    """
    if isinstance(particle_position, torch.Tensor):
        particle_position = (particle_position[0].item(), particle_position[1].item())
    r_p, z_p = particle_position
    
    r_np = radial_grid.detach().cpu().numpy()
    z_np = axial_grid.detach().cpu().numpy()
    res_np = residual_magnitude.detach().cpu().numpy()
    
    domain_length = float(z_np[-1])
    domain_radius = float(r_np[-1])
    
    fig_width = 10.0
    fig_height = max(3.2, fig_width * (domain_radius / domain_length) * 2.5)
    
    figure = matplotlib.figure.Figure(figsize=(fig_width, fig_height))
    axes = figure.subplots()
    
    mesh = axes.pcolormesh(
        z_np, r_np, res_np,
        shading="auto", cmap="magma"
    )
    figure.colorbar(mesh, ax=axes, label=f"|{component_name}| Residual", pad=0.02)
    
    # Draw particle boundary patch
    particle_circle = patches.Circle(
        (z_p, r_p),
        radius=particle_radius,
        facecolor="lightgrey",
        edgecolor="cyan",
        linewidth=1.5,
        zorder=10,
        label="Particle boundary",
    )
    axes.add_patch(particle_circle)
    
    # Channel geometry boundaries
    axes.axhline(domain_radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    
    # SQUARE SCALE CONSTRAINT: Prevents particle shape distortion
    axes.set_aspect("equal", adjustable="box")
    
    axes.set_xlabel("Axial position $z^*$")
    axes.set_ylabel("Radial position $r^*$")
    axes.set_title(f"Particle-Coupled Held-Out {component_name} Residual")
    axes.set_xlim(0.0, domain_length)
    axes.set_ylim(0.0, domain_radius)
    
    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    return figure


def plot_traction_uniformity(
    profile: pd.DataFrame, expected_value: float, **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot the per-angle axial traction $t_z(\theta)$ against its theoretical uniform value.

    For a translating sphere in unbounded Stokes flow, $t_z(\theta)$ is exactly constant over
    the whole surface. A curve that is not flat — especially one crossing zero near
    $\theta=\pi/2$ instead of sitting on the reference line — is diagnostic of a sign error in
    the stress assembly, independent of whatever field is being tested.

    Args:
    - `profile`: Output of `verification.force_checks.traction_profile`.
    - `expected_value`: The theoretical uniform value, $-\tfrac{3U}{2a}$.
    - `**plot_kwargs`: Forwarded to `Axes.plot` for the computed curve.

    Returns:
    - A `matplotlib.figure.Figure`.
    """
    figure = matplotlib.figure.Figure(figsize=(6.5, 4.0))
    axes = figure.subplots()
    default_opts: dict[str, Any] = {"color": "tab:blue", "linewidth": 1.6, "label": r"computed $t_z(\theta)$"}
    axes.plot(profile["theta"], profile["t_z"], **{**default_opts, **plot_kwargs})
    axes.axhline(expected_value, color="black", linestyle="--", linewidth=1.2,
                 label=f"uniform reference ({expected_value:.3f})")
    axes.axvline(math.pi / 2, color="gray", linestyle=":", linewidth=1.0, label=r"$\theta=\pi/2$")
    axes.set_xlabel(r"Polar angle $\theta$")
    axes.set_ylabel(r"Axial traction $t_z(\theta)$")
    axes.set_title("Surface traction uniformity check")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_resistance_pair_comparison(
    pair_a: float, pair_b: float, ref_a: float, ref_b: float,
    spread_a: float = 0.0, spread_b: float = 0.0,
) -> matplotlib.figure.Figure:
    r"""Bar chart comparing the trained resistance pair $(F_A, F_B)$ against the tube reference.

    Args:
    - `pair_a`, `pair_b`: Trained $F_A$ (fixed sphere) and $F_B$ (unit translation).
    - `ref_a`, `ref_b`: Reference values, e.g. from `tube_resistance_reference`.
    - `spread_a`, `spread_b`: Relative surface-independence spread, shown as an error bar
      fraction of $|F_A|$, $|F_B|$ — a measure of self-consistency, not of accuracy.

    Returns:
    - A `matplotlib.figure.Figure`.
    """
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    labels = ["$F_A$ (fixed)", "$F_B$ (translation)"]
    positions = np.arange(2)
    width = 0.35
    trained = [pair_a, pair_b]
    reference = [ref_a, ref_b]
    errors = [abs(pair_a) * spread_a, abs(pair_b) * spread_b]
    axes.bar(positions - width / 2, trained, width, yerr=errors, capsize=4,
             color="tab:blue", label="Trained (stream-function)")
    axes.bar(positions + width / 2, reference, width, color="tab:orange", label="Tube reference")
    axes.axhline(0.0, color="black", linewidth=0.75)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel("Force (dimensionless)")
    axes.set_title("Resistance pair: trained vs. tube reference")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_resistance_position_sweep(sweep: pd.DataFrame, ref_a: float, ref_b: float) -> matplotlib.figure.Figure:
    r"""Plot $F_A(z_p)$ and $F_B(z_p)$ from a position sweep, against their reference values.

    Args:
    - `sweep`: Output of `trajectory.mobility.resistance_position_sweep`.
    - `ref_a`, `ref_b`: Reference values, e.g. from `tube_resistance_reference`.

    Returns:
    - A `matplotlib.figure.Figure` with two panels.
    """
    figure = matplotlib.figure.Figure(figsize=(9.0, 3.6))
    axes = figure.subplots(1, 2)
    for ax, column, ref, title in zip(axes, ("F_A", "F_B"), (ref_a, ref_b), ("$F_A$", "$F_B$")):
        ax.plot(sweep["z_p"], sweep[column], "o-", color="tab:blue", label="trained")
        ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, label="reference")
        ax.set_xlabel("$z_p$")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. axial position")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_residual_by_region(table: pd.DataFrame) -> matplotlib.figure.Figure:
    r"""Bar chart of the volume-weighted RMS residual, by spatial region.

    Args:
    - `table`: Output of `verification.force_checks.residual_by_region`.

    Returns:
    - A `matplotlib.figure.Figure`.
    """
    figure = matplotlib.figure.Figure(figsize=(6.5, 4.0))
    axes = figure.subplots()
    regions = table.index.tolist()
    positions = np.arange(len(regions))
    width = 0.25
    axes.bar(positions - width, table["rms_r"], width, label=r"$f_r$")
    axes.bar(positions, table["rms_z"], width, label=r"$f_z$")
    axes.bar(positions + width, table["rms_continuity"], width, label="continuity")
    axes.set_yscale("log")
    axes.set_xticks(positions)
    axes.set_xticklabels(regions)
    axes.set_ylabel("Volume-weighted RMS residual (log scale)")
    axes.set_title("Held-out Stokes residual, by region")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_surface_independence(report: pd.DataFrame, expected: float | None = None) -> matplotlib.figure.Figure:
    r"""Plot the traction force against control-sphere radius, from `surface_independence_report`.

    Args:
    - `report`: Output of `verification.force_checks.surface_independence_report`.
    - `expected`: Optional theoretical value to draw as a reference line.

    Returns:
    - A `matplotlib.figure.Figure`.
    """
    figure = matplotlib.figure.Figure(figsize=(6.0, 3.6))
    axes = figure.subplots()
    axes.plot(report["radius"], report["force"], "o-", color="tab:blue", label="control-sphere force")
    if expected is not None:
        axes.axhline(expected, color="black", linestyle="--", linewidth=1.2, label="theoretical")
    axes.set_xlabel("Control-sphere radius")
    axes.set_ylabel("$F_z$")
    axes.set_title("Surface independence of the traction integral")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_stream_function_field(
    network: nn.Module, domain: Domain, particle_axial_position: float, particle_radius: float,
    resolution_r: int = 150, resolution_z: int = 300, cmap: str = DEFAULT_COLORMAP,
) -> matplotlib.figure.Figure:
    r"""Plot velocity magnitude and streamlines around the sphere for a stream-function network.

    Unlike `plot_particle_flow_field` (built for the older immersed-core model, with its
    coordinate-convention auto-detection), this assumes the stream-function network's fixed,
    known `(r, z) -> (u_r, u_z, p)` convention directly — no guessing needed.

    Args:
    - `network`: A `StreamFunctionSphereFlow`-wrapped network.
    - `domain`: Nondimensional channel geometry.
    - `particle_axial_position`, `particle_radius`: Sphere center and radius.
    - `resolution_r`, `resolution_z`: Grid resolution.
    - `cmap`: Colormap for the velocity-magnitude heatmap.

    Returns:
    - A `matplotlib.figure.Figure`.

    Raises:
    - `ValueError`: If `resolution_r` or `resolution_z` is not strictly positive.
    """
    if resolution_r <= 0 or resolution_z <= 0:
        raise ValueError("resolution_r and resolution_z must be strictly positive.")
    device = resolve_module_device(network)
    dtype = resolve_module_dtype(network)
    r_grid = torch.linspace(0.0, domain.radius, resolution_r, device=device, dtype=dtype)
    z_grid = torch.linspace(0.0, domain.length, resolution_z, device=device, dtype=dtype)
    rr, zz = torch.meshgrid(r_grid, z_grid, indexing="ij")
    with torch.no_grad():
        prediction = network(torch.stack([rr.reshape(-1), zz.reshape(-1)], dim=1)).reshape(resolution_r, resolution_z, 3)
    u_r, u_z = prediction[..., 0], prediction[..., 1]
    speed = torch.hypot(u_r, u_z)
    inside = (zz - particle_axial_position).square() + rr.square() < particle_radius**2
    speed_masked = np.ma.array(speed.cpu().numpy(), mask=inside.cpu().numpy())
    
    # CHANGED: plain-NumPy axes for plotting, not the device-origin r_grid/z_grid tensors —
    # see `_uniform_plot_axis`'s docstring for why `streamplot` needs this.
    z_axis_np = _uniform_plot_axis(domain.length, resolution_z)
    r_axis_np = _uniform_plot_axis(domain.radius, resolution_r)
    
    figure = matplotlib.figure.Figure(figsize=(9.0, 3.2))
    axes = figure.subplots()
    mesh = axes.pcolormesh(z_axis_np, r_axis_np, speed_masked,
                           shading="auto", cmap=cmap)
    figure.colorbar(mesh, ax=axes, label=r"$|\mathbf{u}^*|$")
    u_z_np, u_r_np = u_z.cpu().numpy().copy(), u_r.cpu().numpy().copy()
    u_z_np[inside.cpu().numpy()] = np.nan
    u_r_np[inside.cpu().numpy()] = np.nan
    axes.streamplot(z_axis_np, r_axis_np, u_z_np, u_r_np,
                     density=DEFAULT_STREAMLINE_DENSITY, color=DEFAULT_PARTICLE_STREAMLINE_COLOR, linewidth=0.6, arrowsize=0.8)
    axes.add_patch(patches.Circle((particle_axial_position, 0.0), radius=particle_radius,
                                  facecolor="darkgrey", edgecolor="black", zorder=10))
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title("Stream-function flow field around the sphere")
    figure.tight_layout()
    return figure
