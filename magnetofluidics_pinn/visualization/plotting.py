"""Plotting functions for flow fields, particle trajectories, and verification metrics.

Each function returns a `matplotlib.figure.Figure` instance instead of
calling `plt.show()` internally, leaving display or saving decisions to the
caller.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import math
import matplotlib.figure
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch import nn

from magnetofluidics_pinn.device_utils import (
    resolve_module_device,
    resolve_module_dtype)
from magnetofluidics_pinn.training.trainer import (
    LOSS_COMPONENT_NAMES as LOSS_COMPONENT_NAMES,
    TrainingHistory as TrainingHistory)
from magnetofluidics_pinn.data_types import Domain, ParticleState
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


def _to_numpy(values: Any) -> np.ndarray:
    """Convert a tensor or array-like to a plain NumPy array.

    Args:
    - `values`: Tensor (any device), NumPy array, or array-like.

    Returns:
    - NumPy `ndarray`; tensors are detached and moved to the host first.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _render_particle_flow_axes(
    axes: Any,
    domain: Domain,
    velocity_r: np.ndarray,
    velocity_z: np.ndarray,
    inside_particle: np.ndarray,
    particle_center: tuple[float, float],
    particle_radius: float,
    resolution_r: int,
    resolution_z: int,
    evaluation_z_stations: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    cmap: str = DEFAULT_COLORMAP,
    title: str = "Particle-Coupled Flow Field & Profile Evaluation Stations",
    attach_colorbar: bool = True,
    show_streamlines: bool = True,
    mesh_kwargs: dict[str, Any] | None = None,
    streamline_kwargs: dict[str, Any] | None = None,
    particle_patch_kwargs: dict[str, Any] | None = None,
    station_kwargs: dict[str, Any] | None = None,
) -> Any:
    r"""Draw one particle-coupled flow frame onto an existing axes.

    Renders the velocity magnitude $|\mathbf{u}^*| = \sqrt{u_r^{*2} + u_z^{*2}}$
    over the axisymmetric meridian $(z^*, r^*) \in [0, L^*] \times [0, R^*]$ as a
    pseudocolor mesh, overlays NaN-masked streamlines, the particle disk, and
    the evaluation stations, and enforces the square aspect ratio.

    Args:
    - `axes`: Target `matplotlib.axes.Axes` (assumed already cleared).
    - `domain`: Dimensionless vessel geometry.
    - `velocity_r`: `velocity_z`: Arrays of shape `(resolution_r, resolution_z)`
      holding the field components $u_r^*$, $u_z^*$ on the grid.
    - `inside_particle`: Boolean array of shape `(resolution_r, resolution_z)`,
      `True` inside the particle, where the field is not interpreted.
    - `particle_center`: Particle center $(r_p^*, z_p^*)$.
    - `particle_radius`: Dimensionless particle radius $a^*$.
    - `resolution_r`: `resolution_z`: Grid resolution per axis.
    - `evaluation_z_stations`: Optional axial stations to highlight.
    - `cmap`: Colormap for the velocity magnitude mesh.
    - `title`: Axes title.
    - `attach_colorbar`: Whether to attach a colorbar for the mesh. The
      animation removes the previous colorbar before each re-render; the static
      plot never re-renders and always attaches one.
    - `show_streamlines`: Whether to overlay masked fluid streamlines.
    - `mesh_kwargs`, `streamline_kwargs`, `particle_patch_kwargs`,
      `station_kwargs`: Optional styling dictionaries merged immutably over
      the module defaults and forwarded to `pcolormesh`, `streamplot`,
      `patches.Circle`, and `axvline` respectively.

    Returns:
    - The pseudocolor mesh artist.

    Raises:
    - `ValueError`: If the component arrays or the mask do not match the
      declared grid resolution.
    """
    expected_shape = (resolution_r, resolution_z)
    if velocity_r.shape != expected_shape or velocity_z.shape != expected_shape:
        raise ValueError(
            f"velocity components must have shape {expected_shape}; got "
            f"{velocity_r.shape} and {velocity_z.shape}."
        )
    if inside_particle.shape != expected_shape:
        raise ValueError(
            f"inside_particle must have shape {expected_shape}; got {inside_particle.shape}."
        )

    r_p, z_p = particle_center
    z_np = _uniform_plot_axis(domain.length, resolution_z)
    r_np = _uniform_plot_axis(domain.radius, resolution_r)

    velocity_mag = np.hypot(velocity_r, velocity_z)
    invalid_mask = ~np.isfinite(velocity_mag)
    full_mask = inside_particle | invalid_mask
    vel_mag_masked = np.ma.array(velocity_mag, mask=full_mask)

    valid_vals = vel_mag_masked.compressed()
    if len(valid_vals) > 0 and np.any(np.isfinite(valid_vals)):
        v_min = max(0.0, float(np.nanmin(valid_vals)))
        v_max = float(np.nanmax(valid_vals))
        if v_min >= v_max:
            v_max = v_min + 1.0
    else:
        v_min, v_max = 0.0, 1.0

    default_mesh_opts: dict[str, Any] = {"vmin"   : v_min, "vmax": v_max, "cmap": cmap,
                                         "shading": "auto"}
    effective_mesh_opts = {**default_mesh_opts, **(mesh_kwargs or {})}
    mesh = axes.pcolormesh(z_np, r_np, vel_mag_masked, **effective_mesh_opts)
    if attach_colorbar:
        axes.figure.colorbar(mesh, ax=axes,
                             label=r"Velocity Magnitude $|\mathbf{u}^*|$", pad=0.02)

    if show_streamlines:
        u_z_plot = velocity_z.copy()
        u_r_plot = velocity_r.copy()
        u_z_plot[inside_particle] = np.nan
        u_r_plot[inside_particle] = np.nan
        default_stream_opts: dict[str, Any] = {
            "density"  : DEFAULT_STREAMLINE_DENSITY,
            "color"    : DEFAULT_PARTICLE_STREAMLINE_COLOR,
            "linewidth": 0.6,
            "arrowsize": 0.8,
        }
        effective_stream_opts = {**default_stream_opts, **(streamline_kwargs or {})}
        if np.nanmax(np.hypot(u_z_plot, u_r_plot)) > 1.0e-6:
            try:
                axes.streamplot(z_np, r_np, u_z_plot, u_r_plot, **effective_stream_opts)
            except Exception:
                pass

    if particle_radius > 0.0:
        default_patch_opts: dict[str, Any] = {
            "facecolor": DEFAULT_PARTICLE_FACECOLOR,
            "edgecolor": DEFAULT_PARTICLE_EDGECOLOR,
            "linewidth": 1.5,
            "zorder"   : 10,
            "label"    : "Particle surface",
        }
        effective_patch_opts = {**default_patch_opts, **(particle_patch_kwargs or {})}
        axes.add_patch(
            patches.Circle((z_p, r_p), radius=particle_radius, **effective_patch_opts))

    has_stations = evaluation_z_stations is not None and len(evaluation_z_stations) > 0
    if has_stations:
        default_station_opts: dict[str, Any] = {
            "color"    : DEFAULT_STATION_COLOR,
            "linestyle": "--",
            "linewidth": DEFAULT_LINE_WIDTH,
            "alpha"    : 0.85,
            "zorder"   : 8,
        }
        effective_station_opts = {**default_station_opts, **(station_kwargs or {})}
        for idx, z_stat in enumerate(evaluation_z_stations):
            z_val = z_stat.item() if isinstance(z_stat, torch.Tensor) else float(z_stat)
            label = "Evaluation station" if idx == 0 else None
            axes.axvline(x=z_val, label=label, **effective_station_opts)
            axes.text(
                z_val, domain.radius * 1.03, f"$z={z_val:.2f}$",
                color=effective_station_opts["color"], fontsize=8,
                ha="center", va="bottom", fontweight="bold", zorder=9,
            )

    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title(title)
    axes.set_xlim(0.0, domain.length)
    axes.set_ylim(0.0, domain.radius * 1.18 if has_stations else domain.radius)
    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return mesh


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
        "density"  : DEFAULT_STREAMLINE_DENSITY,
        "color"    : DEFAULT_STREAMLINE_COLOR,
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
        "marker"    : "o",
        "markersize": 2.5,
        "linewidth" : DEFAULT_LINE_WIDTH,
    }
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}
    for trajectory in trajectories:
        radial_positions = np.array([state.position[0].item() for state in trajectory])
        axial_positions = np.array([state.position[1].item() for state in trajectory])
        axes.plot(axial_positions, radial_positions, **effective_plot_opts)

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
        raise ValueError(
            "Cannot plot empty TrainingHistory: both phases contain zero steps.")

    adam_steps = np.asarray(history.adam.step, dtype=np.int64)
    lbfgs_steps = np.asarray(history.lbfgs.step, dtype=np.int64) + n_adam
    combined_steps = np.concatenate(
        [adam_steps, lbfgs_steps]) if n_lbfgs > 0 else adam_steps

    def merge_series(field: str) -> np.ndarray:
        adam_vals = getattr(history.adam, field)
        lbfgs_vals = getattr(history.lbfgs, field) if n_lbfgs > 0 else ()
        return (
            np.concatenate([np.asarray(adam_vals), np.asarray(lbfgs_vals)])
            if n_lbfgs > 0 else np.asarray(adam_vals)
        )

    panel_titles: dict[str, str] = {
        "total"       : "Composite Total Loss",
        "momentum_r"  : "Radial-Momentum Residual",
        "momentum_z"  : "Axial-Momentum Residual",
        "continuity"  : "Continuity (Mass-Conservation) Residual",
        "wall"        : "Wall No-Slip Loss", "inlet": "Inlet Velocity Loss",
        "outlet"      : "Outlet Pressure Loss",
        "positivity"  : "Axial-Velocity Positivity Penalty",
        "conservation": "Global Flow-Rate Conservation Loss",
    }
    palette = ("black", "tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:green",
               "tab:purple", "tab:brown",
               "tab:pink")
    field_order = ("total",) + LOSS_COMPONENT_NAMES
    loss_metrics = tuple(
        (panel_titles[field], merge_series(field), color,
         "--" if field == "total" else "-")
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
        ax.plot(combined_steps, series, color=color, linestyle=linestyle, linewidth=1.5,
                label=title)
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

    for _ax, (_title, _series, _col, _ls) in zip(axes_flat[: len(loss_metrics)],
                                                 loss_metrics):
        render_loss_axis(_ax, _title, _series, _col, _ls)
    overlay_ax = axes_flat[len(loss_metrics)]

    for _title, _series, _col, _ls in loss_metrics:
        overlay_ax.plot(combined_steps, _series, color=_col, linestyle=_ls,
                        linewidth=1.2, label=_title)
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


def plot_two_way_history(history: TrainingHistory,
                         log_scale: bool = True) -> matplotlib.figure.Figure:
    """Loss convergence plot for a `train()` run, including the optional particle term."""
    has_particle_term = getattr(history.adam, "particle", None) is not None
    component_names = LOSS_COMPONENT_NAMES + (
        ("particle",) if has_particle_term else ())
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
        style = dict(linewidth=2.0, linestyle="--",
                     color="black") if field == "total" else dict(linewidth=1.0)
        ax.plot(steps, vals, label=field, **style)
    if n_adam and n_lbfgs:
        ax.axvline(n_adam, color="tab:red", linestyle=":", linewidth=1.2,
                   label="Adam → L-BFGS")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss magnitude")
    ax.set_title("Two-way-coupled training convergence")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def _stream_history_records(history: Any) -> tuple[
    list[str], list[Sequence[float]], list[Sequence[float]]]:
    """Extract component names and per-phase records from a stream-function training history.

    Args:
    - `history`: Training history exposing `names`, `adam`, and `lbfgs`.

    Returns:
    - Tuple `(names, adam_records, lbfgs_records)`.

    Raises:
    - `ValueError`: If the history lacks a total entry or holds no records.
    """
    names = list(getattr(history, "names", ()) or ())
    if "total" not in names:
        raise ValueError("history.names must contain a 'total' entry.")
    adam_records = list(getattr(history, "adam", ()) or ())
    lbfgs_records = list(getattr(history, "lbfgs", ()) or ())
    if not adam_records and not lbfgs_records:
        raise ValueError(
            "Cannot plot an empty training history: both phases contain zero records.")
    return names, adam_records, lbfgs_records


def plot_stream_loss_history(
    history: Any,
    component_colors: Mapping[str, str] | None = None,
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a stream-function training history by component, phase-separated.

    Draws every component of the Adam phase, then the L-BFGS total continued
    on the same abscissa, with a vertical separator at the phase boundary.

    Args:
    - `history`: Training history exposing `names`, `adam`, and `lbfgs` records
      (each record indexable by the component's position in `names`).
    - `component_colors`: Optional mapping from component name to color;
      unmatched components cycle through the module palette.
    - `log_scale`: Whether to use a logarithmic ordinate.

    Returns:
    - A `matplotlib.figure.Figure` with the per-component loss curves.

    Raises:
    - `ValueError`: If the history is empty or lacks a total entry.
    """
    names, adam_records, lbfgs_records = _stream_history_records(history)
    total_index = names.index("total")
    palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown", "tab:pink", "tab:cyan", "tab:gray")
    colors = dict(component_colors or {})
    lookup = [colors.get(name, palette[index % len(palette)])
              for index, name in enumerate(names)]

    figure = matplotlib.figure.Figure(figsize=(7.5, 4.0))
    axes = figure.subplots()
    for index, name in enumerate(names):
        if name == "total":
            continue
        if adam_records:
            axes.semilogy([record[index] for record in adam_records],
                          color=lookup[index], linewidth=0.9, alpha=0.8,
                          label=f"Adam {name}")
    adam_totals = [record[total_index] for record in adam_records]
    lbfgs_totals = [record[total_index] for record in lbfgs_records]
    if adam_totals:
        axes.semilogy(adam_totals, color="black", linewidth=1.6, label="Adam total")
    if lbfgs_totals:
        offset = len(adam_totals) - 1 if adam_totals else 0
        axes.semilogy(range(offset, offset + len(lbfgs_totals)),
                      lbfgs_totals, color="black", linewidth=1.6, linestyle="--",
                      label="L-BFGS total")
    if adam_totals and lbfgs_totals:
        axes.axvline(len(adam_totals) - 1, color="gray", linestyle=":", linewidth=1.0)
    if log_scale:
        axes.set_yscale("log")
    axes.set_xlabel("Adam epoch / L-BFGS closure evaluation")
    axes.set_ylabel("Loss (mean-squared residual)")
    axes.set_title("Stream-function training loss by component")
    axes.legend(fontsize=7, ncol=2, loc="upper right")
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_history_totals_comparison(
    histories: Mapping[str, Any],
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Compare the total-loss curves of several stream-function training runs.

    Args:
    - `histories`: Mapping from run label to training history (each exposing
      `names`, `adam`, and `lbfgs`); at least one entry.
    - `log_scale`: Whether to use a logarithmic ordinate.

    Returns:
    - A `matplotlib.figure.Figure` with one total-loss curve per run.

    Raises:
    - `ValueError`: If `histories` is empty or a history is invalid.
    """
    if not histories:
        raise ValueError("histories must contain at least one run.")
    palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown")
    figure = matplotlib.figure.Figure(figsize=(7.0, 3.6))
    axes = figure.subplots()
    for index, (label, history) in enumerate(histories.items()):
        names, adam_records, lbfgs_records = _stream_history_records(history)
        total_index = names.index("total")
        totals = [record[total_index] for record in adam_records] + \
                 [record[total_index] for record in lbfgs_records]
        axes.semilogy(totals, color=palette[index % len(palette)], linewidth=1.2,
                      label=label)
    if log_scale:
        axes.set_yscale("log")
    axes.set_xlabel("Adam epoch / L-BFGS closure evaluation")
    axes.set_ylabel("Total loss")
    axes.set_title("Warm-start against cold-start convergence")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_velocity_profile_comparison(profile_evaluation: ProfileEvaluation,
                                     **plot_kwargs: Any, ) -> matplotlib.figure.Figure:
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
        "marker"    : "o",
        "markersize": 2.5,
        "linewidth" : 1.0,
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
        color="black", linestyle="--", linewidth=1.5,
        label="Analytical (Hagen-Poiseuille)",
    )
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"Axial velocity $u_z^*$")
    axes.set_title("Predicted vs. analytical axial-velocity profiles")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_profile_error(profile_evaluation: ProfileEvaluation,
                       **plot_kwargs: Any, ) -> matplotlib.figure.Figure:
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
    error = (
            profile_evaluation.predicted_uz - profile_evaluation.analytical_uz.unsqueeze(
            0)).cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    default_plot_opts: dict[str, Any] = {"linewidth": DEFAULT_LINE_WIDTH}
    effective_plot_opts = {**default_plot_opts, **plot_kwargs}

    for station_index, z_station in enumerate(profile_evaluation.z_stations):
        axes.plot(radial_grid, error[station_index], label=f"z={z_station:.2e}",
                  **effective_plot_opts)
    axes.axhline(0.0, color="black", linewidth=0.75)
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"$u_z$ error (predicted - analytical)")
    axes.set_title("Axial-velocity profile error")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_station_profile_comparison(
    radial_grid: torch.Tensor,
    station_profiles: Sequence[tuple[float, torch.Tensor, torch.Tensor]],
    labels: Sequence[str],
    **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Overlay the particle-perturbed and undisturbed $u_z(r)$ profiles at axial stations.

    Args:
    - `radial_grid`: 1-D tensor of radial positions $r^*$.
    - `station_profiles`: Per-station tuples
      $(z^*, u_{z,\mathrm{two-way}}, u_{z,\mathrm{baseline}})$.
    - `labels`: Per-station labels, matched positionally with `station_profiles`.
    - `**plot_kwargs`: Keyword arguments forwarded to `Axes.plot` for the
      predicted curves (e.g., `linewidth`, `markersize`).

    Returns:
    - A `matplotlib.figure.Figure` with one solid predicted and one dashed
      baseline curve per station.

    Raises:
    - `ValueError`: If the stations and labels do not match, or a profile does
      not match the radial grid.
    """
    if not station_profiles:
        raise ValueError("station_profiles must contain at least one station.")
    if len(labels) != len(station_profiles):
        raise ValueError(
            f"labels ({len(labels)}) and station_profiles ({len(station_profiles)}) must match."
        )
    r_np = _to_numpy(radial_grid)
    palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown")
    figure = matplotlib.figure.Figure(figsize=(7.5, 4.2))
    axes = figure.subplots()
    default_opts: dict[str, Any] = {"linewidth": 1.6}
    effective_opts = {**default_opts, **plot_kwargs}
    for index, ((z_station, predicted, baseline), label) in enumerate(
        zip(station_profiles, labels)):
        predicted_np = _to_numpy(predicted)
        baseline_np = _to_numpy(baseline)
        if predicted_np.shape != r_np.shape or baseline_np.shape != r_np.shape:
            raise ValueError(
                f"Station {z_station}: profile shapes {predicted_np.shape} and "
                f"{baseline_np.shape} must match the radial grid {r_np.shape}."
            )
        color = palette[index % len(palette)]
        axes.plot(r_np, predicted_np, color=color,
                  label=f"{label} (z*={z_station:.2f})", **effective_opts)
        axes.plot(r_np, baseline_np, color=color, linestyle="--", linewidth=1.0,
                  alpha=0.7,
                  label="undisturbed baseline" if index == 0 else None)
    axes.set_xlabel(r"Radial position $r^*$")
    axes.set_ylabel(r"Axial velocity $u_z^*$")
    axes.set_title("Particle-perturbed vs. undisturbed axial-velocity profiles")
    axes.legend(fontsize=7, ncol=2, loc="lower left")
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_radial_leakage(profile_evaluation: ProfileEvaluation,
                        **plot_kwargs: Any, ) -> matplotlib.figure.Figure:
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


def plot_flow_rate_deviation(axial_positions: torch.Tensor,
                             flow_rate_curve: torch.Tensor,
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
        raise ValueError(
            f"reference_flow_rate must be strictly positive; got {reference_flow_rate!r}.")

    z_values = axial_positions.detach().cpu().numpy()
    deviation = (flow_rate_curve.detach() - reference_flow_rate).cpu().numpy()
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    default_plot_opts: dict[str, Any] = {
        "marker"    : "o",
        "markersize": 3.0,
        "linewidth" : DEFAULT_LINE_WIDTH,
        "color"     : "tab:red",
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


def plot_flow_rate_reconnection(
    z_upstream: torch.Tensor,
    q_upstream: torch.Tensor,
    z_downstream: torch.Tensor,
    q_downstream: torch.Tensor,
    reference_flow_rate: float,
    particle_axial_position: float,
    particle_radius: float,
    **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot the flow-rate reconnection $Q(z^*)$ around the excluded particle interval.

    Args:
    - `z_upstream`: `q_upstream`: Upstream branch of the $Q(z^*)$ sweep.
    - `z_downstream`: `q_downstream`: Downstream branch of the sweep.
    - `reference_flow_rate`: Analytical reference $Q_{\mathrm{ref}}^*$.
    - `particle_axial_position`: `particle_radius`: Sphere location and radius,
      defining the excluded interval drawn as a shaded span.
    - `**plot_kwargs`: Keyword arguments forwarded to `Axes.plot`.

    Returns:
    - A `matplotlib.figure.Figure` with both branches, the reference line, and
      the excluded interval.

    Raises:
    - `ValueError`: If either branch's positions and values differ in length,
      or the reference flow rate is not strictly positive.
    """
    if reference_flow_rate <= 0.0:
        raise ValueError(
            f"reference_flow_rate must be strictly positive; got {reference_flow_rate!r}.")
    z_up = _to_numpy(z_upstream)
    q_up = _to_numpy(q_upstream)
    z_down = _to_numpy(z_downstream)
    q_down = _to_numpy(q_downstream)
    if z_up.shape != q_up.shape or z_down.shape != q_down.shape:
        raise ValueError(
            "Each branch's axial positions and flow rates must have equal shapes.")

    figure = matplotlib.figure.Figure(figsize=(8.0, 3.2))
    axes = figure.subplots()
    default_opts: dict[str, Any] = {"color"     : "tab:blue", "marker": ".",
                                    "markersize": 3.0, "linewidth": 1.0}
    effective_opts = {**default_opts, **plot_kwargs}
    axes.plot(z_up, q_up, label="$Q(z)$, two-way", **effective_opts)
    axes.plot(z_down, q_down, **effective_opts)
    axes.axhline(reference_flow_rate, color="black", linestyle="--", linewidth=1.0,
                 label=r"$Q_{\mathrm{ref}}^*$ (analytical)")
    axes.axvspan(particle_axial_position - particle_radius,
                 particle_axial_position + particle_radius,
                 color="lightgray", alpha=0.5, label="Particle (excluded)")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"$Q(z)^*$")
    axes.set_title("Flow-rate reconnection around the particle")
    axes.legend(fontsize=8, loc="lower center")
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
    analytical_pressure = reference_pressure_at_outlet + analytical_slope * (
            axial_positions - domain_length)
    figure = matplotlib.figure.Figure(figsize=(6.0, 4.0))
    axes = figure.subplots()
    default_scatter_opts: dict[str, Any] = {"s"    : 10.0, "color": "tab:blue",
                                            "label": r"Predicted $p(r=0, z)$"}
    effective_scatter_opts = {**default_scatter_opts, **(scatter_kwargs or {})}
    axes.scatter(axial_positions, predicted_pressure, **effective_scatter_opts)
    axes.plot(
        axial_positions, fitted_slope * axial_positions + fitted_intercept,
        color="tab:orange", linewidth=1.2, label="Linear fit",
    )
    axes.plot(axial_positions, analytical_pressure, color="black", linestyle="--",
              linewidth=1.2, label="Analytical")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Pressure $p^*$ (r=0)")
    axes.set_title("Axial pressure profile: predicted fit vs. analytical")
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, linestyle=":", alpha=DEFAULT_GRID_ALPHA)
    figure.tight_layout()
    return figure


def plot_pressure_through_particle(
    axial_positions: np.ndarray,
    predicted_pressure: np.ndarray | np.ma.MaskedArray,
    analytical_pressure: np.ndarray,
    particle_axial_position: float,
    particle_radius: float,
    **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot the axial pressure through the particle against the particle-free baseline.

    Args:
    - `axial_positions`: Axial stations $z^*$ of the sweep.
    - `predicted_pressure`: Predicted $p^*(r{\approx}0, z)$, optionally masked
      over the excluded particle interval.
    - `analytical_pressure`: Particle-free analytical baseline.
    - `particle_axial_position`: `particle_radius`: Sphere location and radius,
      defining the excluded interval drawn as a shaded span.
    - `**plot_kwargs`: Keyword arguments forwarded to `Axes.plot` for the
      predicted curve.

    Returns:
    - A `matplotlib.figure.Figure` with the masked predicted curve, the
      baseline, and the excluded interval.

    Raises:
    - `ValueError`: If the input arrays differ in length.
    """
    z_np = _to_numpy(axial_positions)
    p_pred = predicted_pressure if isinstance(predicted_pressure, np.ma.MaskedArray) \
        else _to_numpy(predicted_pressure)
    p_ana = _to_numpy(analytical_pressure)
    if not (len(z_np) == len(p_pred) == len(p_ana)):
        raise ValueError(
            "axial_positions, predicted_pressure, and analytical_pressure must have equal lengths.")

    figure = matplotlib.figure.Figure(figsize=(8.0, 3.2))
    axes = figure.subplots()
    default_opts: dict[str, Any] = {"color": "tab:red", "linewidth": 1.2}
    effective_opts = {**default_opts, **plot_kwargs}
    axes.plot(z_np, p_pred, label=r"Predicted $p(r{\approx}0, z)$, two-way",
              **effective_opts)
    axes.plot(z_np, p_ana, color="black", linestyle="--", linewidth=1.0,
              label="Analytical (particle-free) baseline")
    axes.axvspan(particle_axial_position - particle_radius,
                 particle_axial_position + particle_radius,
                 color="lightgray", alpha=0.5, label="Particle (excluded)")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Pressure $p^*(r{\approx}0, z)$")
    axes.set_title("Axial pressure through the particle vs. the undisturbed baseline")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
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
    residual_magnitude = residual_magnitude.detach().cpu().numpy()
    v_min = np.nanmin(residual_magnitude)
    v_max = np.nanmax(residual_magnitude)
    figure = matplotlib.figure.Figure(figsize=(7.0, 3.5))
    axes = figure.subplots()
    default_mesh_opts: dict[str, Any] = {
        "vmin"   : v_min,
        "vmax"   : v_max,
        "shading": "auto",
        "cmap"   : cmap,
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
    particle_position: tuple[float, float] | list[
        float] | torch.Tensor | np.ndarray | ParticleState,
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
        z_stats = [s.item() if isinstance(s, torch.Tensor) else float(s) for s in
                   evaluation_z_stations]
        mean_z = float(np.mean(z_stats))
        if abs(v0 - mean_z) < abs(v1 - mean_z):
            return v1, v0

    return v0, v1


def plot_particle_flow_field(
    flow_network: nn.Module | ProfileEvaluation,
    domain: Domain,
    particle_position: tuple[float, float] | list[
        float] | torch.Tensor | np.ndarray | ParticleState,
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

    Visualizes the two-way coupled velocity magnitude
    $|\mathbf{u}^*| = \sqrt{u_r^{*2} + u_z^{*2}}$ over the axisymmetric meridian
    $(z^*, r^*) \in [0, L^*] \times [0, R^*]$ and highlights particle exclusion
    and velocity profile evaluation stations. The rendering is delegated to
    `_render_particle_flow_axes`, shared with the animation.

    Args:
        - `flow_network`: Trained neural network mapping coordinates to flow fields, or `ProfileEvaluation`.
        - `domain`: Dimensionless vessel geometry.
        - `particle_position`: Center coordinates of the particle $(r_p^*, z_p^*)$ or `ParticleState`.
        - `particle_radius`: Dimensionless radius of the particle $a^*$.
        - `evaluation_z_stations`: Axial $z^*$ positions where velocity profiles are evaluated.
        - `resolution_r`: Number of radial evaluation points.
        - `resolution_z`: Number of axial evaluation points.
        - `show_streamlines`: Whether to overlay fluid velocity streamlines.
        - `cmap`: Colormap used for the velocity magnitude heatmap.
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

    model = getattr(flow_network, "flow_network", flow_network)
    model = getattr(model, "model", model)
    model.eval()

    r_p, z_p = _extract_particle_coords(particle_position, domain,
                                        evaluation_z_stations)
    if particle_radius > domain.radius:
        particle_radius = min(particle_radius, domain.radius * 0.5)

    network_device = resolve_module_device(model)
    network_dtype = resolve_module_dtype(model)

    r_t = torch.linspace(0.0, domain.radius, resolution_r, device=network_device,
                         dtype=network_dtype)
    z_t = torch.linspace(0.0, domain.length, resolution_z, device=network_device,
                         dtype=network_dtype)
    rr, zz = torch.meshgrid(r_t, z_t, indexing="ij")
    rr_flat = rr.reshape(-1)
    zz_flat = zz.reshape(-1)

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
    with torch.no_grad():
        for mode, pts in candidates:
            try:
                out = model(pts)
                if not isinstance(out,
                                  torch.Tensor) or out.numel() != resolution_r * resolution_z * 3:
                    continue
                pred = out.reshape(resolution_r, resolution_z, 3)
                if not torch.isfinite(pred).any():
                    continue
                val_mag = torch.nan_to_num(torch.hypot(pred[..., 0], pred[..., 1]),
                                           nan=0.0).max().item()
                if val_mag > max_finite_val:
                    max_finite_val = val_mag
                    best_pred = pred
                    best_mode = mode
                if val_mag > 1.0e-4:
                    break
            except Exception:
                continue

    if best_pred is None:
        pts = torch.stack([rr_flat, zz_flat], dim=1)
        with torch.no_grad():
            best_pred = model(pts).reshape(resolution_r, resolution_z, 3)
        best_mode = "rz"

    if "zr" in best_mode:
        u_z_field = best_pred[..., 0]
        u_r_field = best_pred[..., 1]
    else:
        u_r_field = best_pred[..., 0]
        u_z_field = best_pred[..., 1]

    if particle_radius > 0.0:
        dist_sq = (zz - z_p).square() + (rr - r_p).square()
        inside_particle = (dist_sq < particle_radius ** 2).detach().cpu().numpy()
    else:
        inside_particle = np.zeros((resolution_r, resolution_z), dtype=bool)

    fig_width = 10.0
    fig_height = max(3.2, fig_width * (domain.radius / domain.length) * 2.5)
    figure = matplotlib.figure.Figure(figsize=(fig_width, fig_height))
    axes = figure.subplots()
    _render_particle_flow_axes(
        axes, domain,
        velocity_r=u_r_field.detach().cpu().numpy(),
        velocity_z=u_z_field.detach().cpu().numpy(),
        inside_particle=inside_particle,
        particle_center=(r_p, z_p),
        particle_radius=particle_radius,
        resolution_r=resolution_r,
        resolution_z=resolution_z,
        evaluation_z_stations=evaluation_z_stations,
        cmap=cmap,
        show_streamlines=show_streamlines,
        mesh_kwargs=mesh_kwargs or None,
        streamline_kwargs=streamline_kwargs,
        particle_patch_kwargs=particle_patch_kwargs,
        station_kwargs=station_kwargs,
    )
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
    default_opts: dict[str, Any] = {"color": "tab:blue", "linewidth": 1.6,
                                    "label": r"computed $t_z(\theta)$"}
    axes.plot(profile["theta"], profile["t_z"], **{**default_opts, **plot_kwargs})
    axes.axhline(expected_value, color="black", linestyle="--", linewidth=1.2,
                 label=f"uniform reference ({expected_value:.3f})")
    axes.axvline(math.pi / 2, color="gray", linestyle=":", linewidth=1.0,
                 label=r"$\theta=\pi/2$")
    axes.set_xlabel(r"Polar angle $\theta$")
    axes.set_ylabel(r"Axial traction $t_z(\theta)$")
    axes.set_title("Surface traction uniformity check")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_quadrature_convergence(
    node_counts: Sequence[float],
    force_values: Sequence[float],
    **plot_kwargs: Any,
) -> matplotlib.figure.Figure:
    r"""Plot the surface-traction force estimate against the quadrature-node count.

    Args:
    - `node_counts`: Quadrature-node counts, plotted on a logarithmic abscissa.
    - `force_values`: Force estimates $f_z^*$ at each node count.
    - `**plot_kwargs`: Keyword arguments forwarded to `Axes.plot`.

    Returns:
    - A `matplotlib.figure.Figure` with the convergence curve.

    Raises:
    - `ValueError`: If the sequences differ in length or hold fewer than two
      points.
    """
    counts = _to_numpy(node_counts)
    forces = _to_numpy(force_values)
    if counts.shape != forces.shape:
        raise ValueError(
            f"node_counts and force_values must have the same shape; got {counts.shape} and {forces.shape}."
        )
    if counts.size < 2:
        raise ValueError("Convergence needs at least two quadrature levels.")

    figure = matplotlib.figure.Figure(figsize=(6.0, 3.6))
    axes = figure.subplots()
    default_opts: dict[str, Any] = {"marker"   : "o", "color": "tab:blue",
                                    "linewidth": 1.2}
    effective_opts = {**default_opts, **plot_kwargs}
    axes.plot(counts, forces, **effective_opts)
    axes.set_xscale("log")
    axes.set_xlabel("Number of quadrature nodes")
    axes.set_ylabel(r"$f_z^*$")
    axes.set_title("Drag-force quadrature convergence")
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
    axes.bar(positions + width / 2, reference, width, color="tab:orange",
             label="Tube reference")
    axes.axhline(0.0, color="black", linewidth=0.75)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel("Force (dimensionless)")
    axes.set_title("Resistance pair: trained vs. tube reference")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


def plot_resistance_position_sweep(sweep: pd.DataFrame, ref_a: float,
                                   ref_b: float) -> matplotlib.figure.Figure:
    r"""Plot $F_A(z_p)$ and $F_B(z_p)$ from a position sweep, against their reference values.

    Args:
    - `sweep`: Output of `trajectory.mobility.resistance_position_sweep`.
    - `ref_a`, `ref_b`: Reference values, e.g. from `tube_resistance_reference`.

    Returns:
    - A `matplotlib.figure.Figure` with two panels.
    """
    figure = matplotlib.figure.Figure(figsize=(9.0, 3.6))
    axes = figure.subplots(1, 2)
    for ax, column, ref, title in zip(axes, ("F_A", "F_B"), (ref_a, ref_b),
                                      ("$F_A$", "$F_B$")):
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


def plot_surface_independence(report: pd.DataFrame,
                              expected: float | None = None) -> matplotlib.figure.Figure:
    r"""Plot the traction force against control-sphere radius, from `surface_independence_report`.

    Args:
    - `report`: Output of `verification.force_checks.surface_independence_report`.
    - `expected`: Optional theoretical value to draw as a reference line.

    Returns:
    - A `matplotlib.figure.Figure`.
    """
    figure = matplotlib.figure.Figure(figsize=(6.0, 3.6))
    axes = figure.subplots()
    axes.plot(report["radius"], report["force"], "o-", color="tab:blue",
              label="control-sphere force")
    if expected is not None:
        axes.axhline(expected, color="black", linestyle="--", linewidth=1.2,
                     label="theoretical")
    axes.set_xlabel("Control-sphere radius")
    axes.set_ylabel("$F_z$")
    axes.set_title("Surface independence of the traction integral")
    axes.legend(fontsize=8)
    axes.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return figure


# Mark following function as depracted?
def plot_stream_function_field(
    network: nn.Module, domain: Domain, particle_axial_position: float,
    particle_radius: float,
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
    r_grid = torch.linspace(0.0, domain.radius, resolution_r, device=device,
                            dtype=dtype)
    z_grid = torch.linspace(0.0, domain.length, resolution_z, device=device,
                            dtype=dtype)
    rr, zz = torch.meshgrid(r_grid, z_grid, indexing="ij")
    with torch.no_grad():
        prediction = network(
            torch.stack([rr.reshape(-1), zz.reshape(-1)], dim=1)).reshape(resolution_r,
                                                                          resolution_z,
                                                                          3)
    u_r, u_z = prediction[..., 0], prediction[..., 1]
    speed = torch.hypot(u_r, u_z)
    inside = (
                     zz - particle_axial_position).square() + rr.square() < particle_radius ** 2
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
                    density=DEFAULT_STREAMLINE_DENSITY,
                    color=DEFAULT_PARTICLE_STREAMLINE_COLOR, linewidth=0.6,
                    arrowsize=0.8)
    axes.add_patch(
        patches.Circle((particle_axial_position, 0.0), radius=particle_radius,
                       facecolor="darkgrey", edgecolor="black", zorder=10))
    axes.axhline(domain.radius, color="black", linewidth=1.5)
    axes.axhline(0.0, color="black", linewidth=0.75, linestyle="--")
    axes.set_xlabel(r"Axial position $z^*$")
    axes.set_ylabel(r"Radial position $r^*$")
    axes.set_title("Stream-function flow field around the sphere")
    figure.tight_layout()
    return figure


def plot_particle_flow_animation(
    domain: Domain,
    frame_field: Callable[[int], nn.Module],
    positions: Sequence[float],
    velocities: Sequence[float],
    times: Sequence[float],
    particle_radius: float,
    *,
    n_frames: int = 60,
    fps: int = 20,
    resolution_r: int = 48,
    resolution_z: int = 160,
    output_path: str | Path | None = None,
    cmap: str = DEFAULT_COLORMAP,
    show_streamlines: bool = True,
    streamline_kwargs: dict[str, Any] | None = None,
    particle_patch_kwargs: dict[str, Any] | None = None,
) -> Path:
    r"""Animate the particle-coupled velocity field along a trajectory and save it as a looping GIF.

    Evaluates the flow field $\mathbf{u}^*(r^*, z^*)$ across selected trajectory
    time steps using PyTorch on CUDA, renders each frame through a deterministic
    `FigureCanvasAgg` canvas with fixed global velocity limits, and encodes
    a looping GIF using Pillow with clean frame disposal.

    Args:
        - `domain`: Dimensionless vessel geometry.
        - `frame_field`: Zero-argument-per-index callable returning the field
          mapping `(r, z)` to `(u_r, u_z, p)` at trajectory state `index`. Its
          device and dtype, resolved from the first frame, place the evaluation
          grid.
        - `positions`: `velocities`: `times`: Per-step trajectory states
          $z_p^*$, $U_p^*$, $t^*$; equal-length sequences.
        - `particle_radius`: Dimensionless particle radius $a^*$.
        - `n_frames`: Frame budget; the trajectory is strided to at most this
          many frames, and the final state is always included.
        - `fps`: Frames per second of the written GIF.
        - `resolution_r`: `resolution_z`: Grid resolution per axis.
        - `output_path`: Destination GIF path; defaults to
          `animations/particle_flow.gif`, created if missing.
        - `cmap`: Colormap for the velocity magnitude mesh.
        - `streamline_kwargs`: `particle_patch_kwargs`: Optional styling
          dictionaries forwarded to the rendering core.

    Returns:
        - The path of the written GIF.

    Raises:
        - `ValueError`: On a non-channel domain, non-positive resolutions,
          budget, or fps, mismatched trajectory sequences, or a field that
          does not return at least two components per point.
    """
    if domain.kind != "channel":
        raise NotImplementedError(
            f"Animation currently only supports domain.kind == 'channel'; got {domain.kind!r}."
        )
    if resolution_r < 2 or resolution_z < 2:
        raise ValueError("Resolutions must be at least 2 per axis.")
    if n_frames < 1 or fps < 1:
        raise ValueError("n_frames and fps must be strictly positive.")
    if not (len(positions) == len(velocities) == len(times)) or len(positions) < 2:
        raise ValueError(
            "positions, velocities, and times must be equal-length sequences of at least two states."
        )
    if particle_radius <= 0.0:
        raise ValueError(
            f"particle_radius must be strictly positive; got {particle_radius!r}.")

    z_positions = [float(p) for p in positions]
    u_velocities = [float(v) for v in velocities]
    t_times = [float(t) for t in times]
    total_states = len(z_positions)

    # Uniform frame indexing across the trajectory
    target_count = min(n_frames, total_states)
    frame_indices = np.unique(
        np.round(np.linspace(0, total_states - 1, target_count)).astype(int)
    ).tolist()

    first_field = frame_field(frame_indices[0])
    network_device = resolve_module_device(first_field)
    network_dtype = resolve_module_dtype(first_field)

    # 1. Coordinate grids for both CUDA inference and Matplotlib surface mesh
    z_grid_np = np.linspace(0.0, domain.length, resolution_z)
    r_grid_np = np.linspace(0.0, domain.radius, resolution_r)
    # Shapes: (resolution_r, resolution_z)
    Z_mesh, R_mesh = np.meshgrid(z_grid_np, r_grid_np)

    r_t = torch.as_tensor(r_grid_np, device=network_device, dtype=network_dtype)
    z_t = torch.as_tensor(z_grid_np, device=network_device, dtype=network_dtype)
    rr, zz = torch.meshgrid(r_t, z_t, indexing="ij")
    grid_points = torch.stack([rr.reshape(-1), zz.reshape(-1)], dim=1)

    def _evaluate_frame(index: int) -> dict[str, Any]:
        """Evaluate the field at trajectory state `index` into a plain record.

        Args:
        - `index`: Position index into the trajectory sequences.

        Returns:
        - A record holding the host-side velocity arrays, the particle mask,
          and the state scalars of one frame.

        Raises:
        - `ValueError`: If `frame_field` does not return at least two
          components per point.
        """
        field = frame_field(index)
        field.eval()
        # CHANGED: Use inference_mode for optimal PyTorch 2.x execution efficiency
        with torch.no_grad():
            prediction = field(grid_points)
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 2 or \
            prediction.shape[1] < 2:
            raise ValueError(
                "frame_field must return a tensor of shape (n_points, >= 2); got "
                f"{tuple(prediction.shape) if isinstance(prediction, torch.Tensor) else type(prediction)!r}."
            )

        u_r = prediction[:, 0].reshape(resolution_r, resolution_z).detach().cpu().numpy()
        u_z = prediction[:, 1].reshape(resolution_r, resolution_z).detach().cpu().numpy()
        v_mag = np.hypot(u_r, u_z)
        inside = (((zz - z_positions[index]) ** 2 + rr ** 2) < particle_radius ** 2).cpu().numpy()

        return {
            "u_r": u_r,
            "u_z": u_z,
            "v_mag": v_mag,
            "inside": inside,
            "z_p": z_positions[index],
            "u_p": u_velocities[index],
            "t": t_times[index],
        }

    # Evaluate every frame before any rendering, so a failure inside
    # `frame_field` raises its own traceback at the evaluation site.
    frames = tuple(_evaluate_frame(index) for index in frame_indices)

    # CHANGED: Compute global maximum strictly over fluid points to avoid core contamination
    fluid_maxima = [
        float(np.max(f["v_mag"][~f["inside"]]))
        for f in frames
        if np.any(~f["inside"])
    ]
    global_vmax = max(fluid_maxima) if fluid_maxima else 1.0
    global_vmax = max(global_vmax, 1.0e-3)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=global_vmax)

    fig_width = 10.0
    fig_height = max(3.2, fig_width * (domain.radius / domain.length) * 2.5)
    figure = matplotlib.figure.Figure(figsize=(fig_width, fig_height))
    canvas = FigureCanvasAgg(figure)

    # Static plot and colorbar axes configuration
    axes = figure.add_axes([0.08, 0.15, 0.78, 0.72])
    cax = figure.add_axes([0.88, 0.15, 0.025, 0.72])

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(scalar_mappable, cax=cax, label=r"Velocity Magnitude $|\mathbf{u}^*|$")

    def _draw(frame: Mapping[str, Any]) -> None:
        """Render one pre-evaluated frame onto the axes.

        Args:
        - `frame`: Record produced by `_evaluate_frame`.
        """
        axes.clear()

        mesh = _render_particle_flow_axes(
            axes, domain,
            velocity_r=frame["u_r"], velocity_z=frame["u_z"],
            inside_particle=frame["inside"],
            particle_center=(0.0, frame["z_p"]),
            particle_radius=particle_radius,
            resolution_r=resolution_r, resolution_z=resolution_z,
            evaluation_z_stations=None, cmap=cmap,
            title=(f"Released particle: $z_p={frame['z_p']:.2f}$, "
                   f"$U_p={frame['u_p']:.3f}$, $t={frame['t']:.2f}$"),
            attach_colorbar=False, show_streamlines=show_streamlines,
            streamline_kwargs=streamline_kwargs,
            particle_patch_kwargs=particle_patch_kwargs,
        )
        # CHANGED: Synchronize mesh color scale with global normalization
        if mesh is not None and hasattr(mesh, "set_clim"):
            mesh.set_clim(0.0, global_vmax)
        # Synchronize color limits across any QuadMesh created by helper
        for coll in axes.collections:
            if hasattr(coll, "set_norm"):
                coll.set_norm(norm)

        axes.set_xlim(0.0, domain.length)
        axes.set_ylim(0.0, domain.radius)

    destination = Path(output_path) if output_path is not None else Path(
        "animations") / "particle_flow.gif"
    destination.parent.mkdir(parents=True, exist_ok=True)

    rendered_images: list[Image.Image] = []
    for frame in frames:
        _draw(frame)
        canvas.draw()
        rgba_buffer = np.asarray(canvas.buffer_rgba())
        # Drop alpha channel to prevent transparency inversion and color artifacts
        rgb_image = Image.fromarray(rgba_buffer[:, :, :3], mode="RGB")
        rendered_images.append(rgb_image)

    # Encode animated GIF with disposal=2 to prevent frame ghosting
    frame_duration_ms = max(10, int(1000 / fps))
    rendered_images[0].save(
        str(destination),
        save_all=True,
        append_images=rendered_images[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    # NEW: Save a representative static PNG frame (mid-trajectory) for PDF rendering
    static_png_path = destination.with_suffix(".png")
    representative_frame_idx = len(rendered_images) // 2
    rendered_images[representative_frame_idx].save(str(static_png_path))

    return destination
