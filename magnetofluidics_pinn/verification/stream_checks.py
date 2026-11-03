r"""Checks specific to the stream-function sphere model.

- `lifting_residual`: the analytic lifting alone (network switched off) must be an exact Stokes solution in the
  `"unbounded"` mode; in the `"taper"` and `"wall"` modes it reports the residual the network has to absorb.
- `far_field_recovery`: far from the sphere the flow must return to the undisturbed Poiseuille profile.
- `axis_pressure_slope_error`: far from the sphere the axis pressure must have the Poiseuille slope $-4U/R^2$.
- `worst_residual_points`: locates the collocation points that dominate the momentum loss.
- `evaluate_native_residual_grid`: native Stokes residual on a structured grid, for heat maps.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.device_utils import resolve_device, resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.networks.lifting import (
    LiftingGeometry,
    LiftingMode,
    fields_from_reduced_stream,
    lifting_pressure,
    reduced_lifting,
)
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual_sz
from magnetofluidics_pinn.sampling.exterior import sample_exterior_points

_COMPONENTS: tuple[str, ...] = ("momentum_r", "momentum_z", "continuity")


def _summarize(residual: torch.Tensor) -> pd.DataFrame:
    """Reduce a residual tensor `(n, 3)` to max and RMS per component plus a `total` row.

    Args:
    - `residual`: Tensor `(n, 3)`.

    Returns:
    - A `DataFrame` indexed by component with columns `max_abs` and `rms`.
    """
    detached = residual.detach()
    rows = {name: {"max_abs": detached[:, k].abs().max().item(), "rms": detached[:, k].pow(2).mean().sqrt().item()}
            for k, name in enumerate(_COMPONENTS)}
    rows["total"] = {"max_abs": detached.abs().max().item(), "rms": detached.pow(2).mean().sqrt().item()}
    return pd.DataFrame.from_dict(rows, orient="index")


def lifting_residual(
    geometry: LiftingGeometry,
    mode: LiftingMode = "unbounded",
    n_points: int = 4000,
    random_seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    exclusion_factor: float = 1.05,
) -> pd.DataFrame:
    r"""Evaluate the Stokes residual of the analytic lifting alone (network switched off).

    Points are uniform in the meridian half-plane $(r, z) \in [0, R]\times[0, L]$, outside the ball $\rho < f a$.

    Args:
    - `geometry`: Geometry constants, see `networks.lifting.LiftingGeometry`.
    - `mode`: `"unbounded"` (must give round-off), `"taper"` or `"wall"`.
    - `n_points`: Number of points, strictly positive.
    - `random_seed`: Seed of a device-local generator.
    - `dtype`: `float64` is recommended: the point of the check is round-off level.
    - `device`: Device; auto-resolved if `None`.
    - `exclusion_factor`: Points closer than this multiple of $a$ to the sphere centre are rejected ($f > 1$).

    Returns:
    - A `DataFrame` indexed by `momentum_r`, `momentum_z`, `continuity`, `total` with columns `max_abs` and `rms`.

    Raises:
    - `ValueError`: If `n_points <= 0` or `exclusion_factor <= 1`.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be strictly positive; got {n_points!r}.")
    if exclusion_factor <= 1.0:
        raise ValueError(f"exclusion_factor must exceed 1; got {exclusion_factor!r}.")
    resolved = resolve_device(device)
    generator = torch.Generator(device=resolved).manual_seed(random_seed)
    r = geometry.radius * torch.rand(4 * n_points, 1, generator=generator, device=resolved, dtype=dtype)
    z = geometry.length * torch.rand(4 * n_points, 1, generator=generator, device=resolved, dtype=dtype)
    keep = (r.square() + (z - geometry.particle_z).square()) > (exclusion_factor * geometry.particle_radius) ** 2
    points = torch.cat([r, z], dim=1)[keep[:, 0]][:n_points]
    sz = torch.cat([points[:, 0:1].square(), points[:, 1:2]], dim=1).requires_grad_(True)

    def fields(sz_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, zz = sz_in[:, 0:1], sz_in[:, 1:2]
        g, u_z = fields_from_reduced_stream(reduced_lifting(s, zz, geometry, mode), sz_in)
        return g, u_z, lifting_pressure(s, zz, geometry, mode)

    return _summarize(stokes_residual_sz(fields, sz))


def far_field_recovery(
    network: nn.Module,
    domain: Domain,
    peak_velocity: float,
    z_stations: tuple[float, ...],
    n_radial: int = 64,
) -> pd.DataFrame:
    r"""Compare the axial-velocity profile at far stations with the undisturbed Poiseuille profile.

    Args:
    - `network`: Flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Nondimensional channel.
    - `peak_velocity`: Ambient centreline velocity $U > 0$.
    - `z_stations`: Axial stations, chosen far from the sphere (several tube radii).
    - `n_radial`: Radial nodes from the axis to the wall, at least 3; the axis node is included.

    Returns:
    - A `DataFrame` indexed by station with columns `max_abs_error`, `relative_l2_error` and `axis_velocity_error`
      ($u_z(0, z)/U - 1$).

    Raises:
    - `ValueError`: If `z_stations` is empty, `n_radial < 3`, or `peak_velocity` is not finite and positive.
    """
    if not z_stations:
        raise ValueError("z_stations must contain at least one station.")
    if n_radial < 3:
        raise ValueError(f"n_radial must be at least 3; got {n_radial!r}.")
    if not (math.isfinite(peak_velocity) and peak_velocity > 0.0):
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    radial = torch.linspace(0.0, domain.radius, n_radial, device=device, dtype=dtype)
    reference = peak_velocity * (1.0 - (radial / domain.radius).square())
    rows = {}
    for station in z_stations:
        coordinates = torch.stack([radial, torch.full_like(radial, station)], dim=1)
        with torch.no_grad():
            u_z = network(coordinates)[:, 1]
        error = u_z - reference
        rows[station] = {
            "max_abs_error": error.abs().max().item(),
            "relative_l2_error": (torch.linalg.norm(error) / torch.linalg.norm(reference)).item(),
            "axis_velocity_error": (u_z[0] / peak_velocity - 1.0).item(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def axis_pressure_slope_error(
    network: nn.Module,
    domain: Domain,
    peak_velocity: float,
    windows: tuple[tuple[float, float], ...],
    n_points: int = 64,
) -> pd.DataFrame:
    r"""Compare the fitted axis-pressure slope on far windows with the Poiseuille slope $-4U/R^2$.

    Args:
    - `network`: Flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Nondimensional channel.
    - `peak_velocity`: Ambient centreline velocity $U > 0$.
    - `windows`: Axial intervals `(z_start, z_end)` with `z_start < z_end`, inside $[0, L]$ and far from the sphere.
    - `n_points`: Samples per window, at least 2.

    Returns:
    - A `DataFrame` indexed by window start with columns `fitted_slope`, `analytical_slope` and `relative_error`.

    Raises:
    - `ValueError`: If a window is invalid, `n_points < 2`, or `peak_velocity` is not finite and positive.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2; got {n_points!r}.")
    if not (math.isfinite(peak_velocity) and peak_velocity > 0.0):
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    if not windows or any(not (0.0 <= lo < hi <= domain.length) for lo, hi in windows):
        raise ValueError("every window must satisfy 0 <= z_start < z_end <= L.")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    analytical = -4.0 * peak_velocity / domain.radius**2
    rows = {}
    for lo, hi in windows:
        z = torch.linspace(lo, hi, n_points, device=device, dtype=dtype)
        with torch.no_grad():
            pressure = network(torch.stack([torch.zeros_like(z), z], dim=1))[:, 2]
        slope = float(np.polyfit(z.cpu().numpy(), pressure.cpu().numpy(), deg=1)[0])
        rows[lo] = {"fitted_slope": slope, "analytical_slope": analytical,
                    "relative_error": abs(slope - analytical) / abs(analytical)}
    return pd.DataFrame.from_dict(rows, orient="index")


def worst_residual_points(
    network: nn.Module,
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    n_points: int = 20_000,
    n_worst: int = 10,
    random_seed: int = 0,
    shell_fraction: float = 0.25,
    shell_factor: float = 2.5,
    axis_clearance_fraction: float = 0.0,
) -> pd.DataFrame:
    r"""Locate the collocation points that dominate the momentum loss.

    Draws one batch with `sample_exterior_points`, evaluates the native residual and returns the `n_worst` points of largest
    $R_r^2 + R_z^2$. Use it on the batches of the epochs where the loss spikes, to see whether the culprit sits near the
    axis ($r \lesssim 10^{-5}$), on the sphere surface or on the taper edge.

    Args:
    - `network`: A `StreamFunctionSphereFlow`.
    - `domain`: Nondimensional channel.
    - `particle_axial_position`, `particle_radius`: Sphere centre and radius.
    - `n_points`: Batch size, strictly positive.
    - `n_worst`: Number of rows returned, strictly positive.
    - `random_seed`: Sampling seed (pass `random_seed + epoch` to reproduce a training batch).
    - `shell_fraction`, `shell_factor`, `axis_clearance_fraction`: Sampler options, see `sample_exterior_points`.

    Returns:
    - A `DataFrame` sorted by decreasing squared momentum residual, with columns `r`, `z`, `rho_over_a`, `momentum_r`,
      `momentum_z`, `squared_momentum` and `share_of_total` (fraction of the batch's summed squared momentum).

    Raises:
    - `ValueError`: If `n_points <= 0` or `n_worst <= 0`, or the network has no `stokes_residual_sz` method.
    """
    if n_points <= 0 or n_worst <= 0:
        raise ValueError("n_points and n_worst must be strictly positive.")
    if not hasattr(network, "stokes_residual_sz"):
        raise ValueError("network must expose `stokes_residual_sz` (a StreamFunctionSphereFlow).")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    points = sample_exterior_points(
        domain, particle_axial_position, particle_radius, n_points, random_seed, shell_fraction, shell_factor,
        device, dtype, axis_clearance_fraction,
    )
    residual = network.stokes_residual_sz(points).detach()
    squared = residual[:, 0].square() + residual[:, 1].square()
    order = torch.argsort(squared, descending=True)[:n_worst]
    rho = torch.hypot(points[:, 0], points[:, 1] - particle_axial_position) / particle_radius
    return pd.DataFrame({
        "r": points[order, 0].cpu().numpy(), "z": points[order, 1].cpu().numpy(),
        "rho_over_a": rho[order].cpu().numpy(), "momentum_r": residual[order, 0].cpu().numpy(),
        "momentum_z": residual[order, 1].cpu().numpy(), "squared_momentum": squared[order].cpu().numpy(),
        "share_of_total": (squared[order] / squared.sum()).cpu().numpy(),
    })


def evaluate_native_residual_grid(
    network: nn.Module,
    radial_grid: torch.Tensor,
    axial_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""Evaluate the absolute native Stokes residual on a structured $(r, z)$ grid, for heat maps.

    Unlike `verification.metrics.evaluate_residual_grid`, the grid may include the axis $r = 0$.

    Args:
    - `network`: A `StreamFunctionSphereFlow` exposing `stokes_residual_sz`.
    - `radial_grid`, `axial_grid`: 1-D tensors.

    Returns:
    - A dict mapping `"momentum_r"`, `"momentum_z"`, `"continuity"` to tensors `(n_radial, n_axial)` of absolute values.

    Raises:
    - `ValueError`: If a grid is not 1-D, or the network has no `stokes_residual_sz` method.
    """
    if radial_grid.ndim != 1 or axial_grid.ndim != 1:
        raise ValueError("radial_grid and axial_grid must both be 1-D tensors.")
    if not hasattr(network, "stokes_residual_sz"):
        raise ValueError("network must expose `stokes_residual_sz` (a StreamFunctionSphereFlow).")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    radial = radial_grid.to(device=device, dtype=dtype)
    axial = axial_grid.to(device=device, dtype=dtype)
    mesh_r, mesh_z = torch.meshgrid(radial, axial, indexing="ij")
    residual = network.stokes_residual_sz(torch.stack([mesh_r.reshape(-1), mesh_z.reshape(-1)], dim=1)).detach()
    shape = (radial.shape[0], axial.shape[0])
    return {name: residual[:, k].abs().reshape(shape) for k, name in enumerate(_COMPONENTS)}
