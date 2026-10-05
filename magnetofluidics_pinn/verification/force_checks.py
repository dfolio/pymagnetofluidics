r"""Force diagnostics that do not depend on the sphere-surface traction alone.

In exact Stokes flow $\div\boldsymbol\sigma = \mathbf 0$, so the force on a body can be
computed on *any* enclosing surface. This module exposes that property as a
test (surface independence) and adds a global momentum balance between two
tube cross-sections [@happel1983low]. Both expose discretization or training
errors that a single surface integral hides.
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd
import torch
from torch import nn

from magnetofluidics_pinn.config import FluidConfig  # NEW
from magnetofluidics_pinn.physics.fluid_residuals import (
    stokes_residual,
    compute_flow_derivatives
)
from magnetofluidics_pinn.sampling.exterior import sample_exterior_points  # NEW
from magnetofluidics_pinn.types import Domain  # NEW
from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype

FlowField = Callable[[torch.Tensor], torch.Tensor]


def _resolve_placement(
    network: FlowField, device: torch.device | str | None, dtype: torch.dtype | None
) -> tuple[torch.device, torch.dtype]:
    """Resolve the device and dtype for new tensors.

    Args:
    - `network`: A module or a plain callable.
    - `device`: Explicit device or `None`.
    - `dtype`: Explicit dtype or `None`.

    Returns:
    - The resolved `(device, dtype)`. Modules use their own parameters; callables fall
      back to CPU and `float64` unless overridden.
    """
    is_module = isinstance(network, nn.Module)
    resolved_device = torch.device(device) if device is not None else (
        resolve_module_device(network) if is_module else torch.device("cpu"))
    resolved_dtype = dtype if dtype is not None else (
        resolve_module_dtype(network) if is_module else torch.float64)
    return resolved_device, resolved_dtype


def _stress_components(network: FlowField, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Evaluate $\sigma_{zz} = -p + 2\partial_z u_z$ and $\sigma_{zr} = \partial_r u_z + \partial_z u_r$.

    Args:
    - `network`: Callable mapping `(n, 2)` to `(n, 3)`.
    - `coordinates`: Tensor `(n, 2)` of `(r, z)` points.

    Returns:
    - Detached tensors `(sigma_zz, sigma_zr)`, each `(n, 1)`.
    """
    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        derivs = compute_flow_derivatives(network, points, compute_second_order=False)
        sigma_zz = -derivs.pressure + 2.0 * derivs.d_velocity_z_dz
        sigma_zr = derivs.d_velocity_z_dr + derivs.d_velocity_r_dz
    return sigma_zz.detach(), sigma_zr.detach()


def sphere_traction_force(
    network: FlowField,
    center_z: float,
    radius: float,
    n_quadrature_points: int = 361,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    r"""Integrate the axial traction over a *control* sphere of arbitrary radius.

    $$ F_z = \int_0^\pi \big(\sigma_{zz}\cos\theta + \sigma_{zr}\sin\theta\big)\,2\pi a^2\sin\theta\,\dd\theta. $$

    For a Stokes field this equals the force on the enclosed body for every radius
    that encloses it and stays inside the fluid domain.

    Args:
    - `network`: Flow field mapping `(r, z)` to `(u_r, u_z, p)`.
    - `center_z`: Axial center of the control sphere.
    - `radius`: Control-sphere radius.
    - `n_quadrature_points`: Polar-angle nodes (trapezoidal rule).
    - `device`: Device override.
    - `dtype`: Dtype override.

    Returns:
    - The axial force as a Python float.

    Raises:
    - `ValueError`: If `radius <= 0`, a scalar is not finite, or `n_quadrature_points < 3`.
    """
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius must be finite and strictly positive; got {radius!r}.")
    if not math.isfinite(center_z):
        raise ValueError(f"center_z must be finite; got {center_z!r}.")
    if n_quadrature_points < 3:
        raise ValueError(f"n_quadrature_points must be at least 3; got {n_quadrature_points!r}.")
    resolved_device, resolved_dtype = _resolve_placement(network, device, dtype)
    theta = torch.linspace(0.0, math.pi, n_quadrature_points, device=resolved_device, dtype=resolved_dtype)
    coordinates = torch.stack([radius * torch.sin(theta), center_z + radius * torch.cos(theta)], dim=1)
    sigma_zz, sigma_zr = _stress_components(network, coordinates)
    traction = sigma_zz.squeeze(1) * torch.cos(theta) + sigma_zr.squeeze(1) * torch.sin(theta)
    integrand = traction * 2.0 * math.pi * radius * coordinates[:, 0]
    return torch.trapezoid(integrand, theta).item()


def surface_independence_report(
    network: FlowField,
    center_z: float,
    radii: tuple[float, ...],
    n_quadrature_points: int = 361,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> pd.DataFrame:
    r"""Evaluate the force on several enclosing spheres and report their spread.

    Args:
    - `network`: Flow field mapping `(r, z)` to `(u_r, u_z, p)`.
    - `center_z`: Axial center of the spheres.
    - `radii`: Control radii, all strictly positive.
    - `n_quadrature_points`: Polar nodes per sphere.
    - `device`: Device override.
    - `dtype`: Dtype override.

    Returns:
    - A `DataFrame` with columns `radius`, `force`, and `relative_deviation`
      $= \abs{F - \bar F}/\abs{\bar F}$ (absolute deviation if $\bar F = 0$).

    Raises:
    - `ValueError`: If `radii` is empty.
    """
    if not radii:
        raise ValueError("radii must contain at least one control radius.")
    forces = tuple(
        sphere_traction_force(network, center_z, radius, n_quadrature_points, device, dtype)
        for radius in radii
    )
    mean_force = sum(forces) / len(forces)
    scale = abs(mean_force) if mean_force != 0.0 else 1.0
    return pd.DataFrame({
        "radius": radii,
        "force": forces,
        "relative_deviation": tuple(abs(force - mean_force) / scale for force in forces),
    })


def cross_section_momentum_force(
    network: FlowField,
    channel_radius: float,
    z_upstream: float,
    z_downstream: float,
    n_radial: int = 401,
    n_axial: int = 401,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    r"""Infer the axial force on a body from a momentum balance between two cross-sections.

    With no inertia, the force on the fluid vanishes. The fluid between the sections
    is acted on by the two end tractions, the wall shear, and the body:

    $$ F_\text{body} = \int_0^R \sigma_{zz}\big|_{z_2} 2\pi r\,\dd r
    - \int_0^R \sigma_{zz}\big|_{z_1} 2\pi r\,\dd r
    + \int_{z_1}^{z_2} \sigma_{zr}\big|_{r=R}\, 2\pi R\,\dd z. $$

    Args:
    - `network`: Flow field mapping `(r, z)` to `(u_r, u_z, p)`.
    - `channel_radius`: Tube radius $R$.
    - `z_upstream`: Upstream section $z_1$.
    - `z_downstream`: Downstream section $z_2 > z_1$.
    - `n_radial`: Radial nodes per section.
    - `n_axial`: Axial nodes along the wall.
    - `device`: Device override.
    - `dtype`: Dtype override.

    Returns:
    - The axial force the fluid exerts on the enclosed body.

    Raises:
    - `ValueError`: If `channel_radius <= 0`, `z_upstream >= z_downstream`, or a node count is below 3.
    """
    if not (math.isfinite(channel_radius) and channel_radius > 0.0):
        raise ValueError(f"channel_radius must be finite and strictly positive; got {channel_radius!r}.")
    if not z_upstream < z_downstream:
        raise ValueError("z_upstream must be smaller than z_downstream.")
    if n_radial < 3 or n_axial < 3:
        raise ValueError("n_radial and n_axial must be at least 3.")
    resolved_device, resolved_dtype = _resolve_placement(network, device, dtype)
    r_grid = torch.linspace(0.0, channel_radius, n_radial, device=resolved_device, dtype=resolved_dtype)

    def section_force(z_value: float) -> torch.Tensor:
        points = torch.stack([r_grid, torch.full_like(r_grid, z_value)], dim=1)
        sigma_zz, _ = _stress_components(network, points)
        return torch.trapezoid(sigma_zz.squeeze(1) * 2.0 * math.pi * r_grid, r_grid)

    z_grid = torch.linspace(z_upstream, z_downstream, n_axial, device=resolved_device, dtype=resolved_dtype)
    wall_points = torch.stack([torch.full_like(z_grid, channel_radius), z_grid], dim=1)
    _, sigma_zr = _stress_components(network, wall_points)
    wall_force = torch.trapezoid(sigma_zr.squeeze(1) * 2.0 * math.pi * channel_radius, z_grid)
    return (section_force(z_downstream) - section_force(z_upstream) + wall_force).item()


def global_momentum_balance(
    network: FlowField,
    center_z: float,
    particle_radius: float,
    channel_radius: float,
    section_offset: float,
    control_radius: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Compare the sphere-surface force with the cross-section momentum force.

    Args:
    - `network`: Flow field mapping `(r, z)` to `(u_r, u_z, p)`.
    - `center_z`: Axial center of the sphere.
    - `particle_radius`: Sphere radius.
    - `channel_radius`: Tube radius.
    - `section_offset`: Distance from `center_z` to each cross-section.
    - `control_radius`: Control-sphere radius; defaults to `particle_radius`.
    - `device`: Device override.
    - `dtype`: Dtype override.

    Returns:
    - A dict with `surface_force`, `momentum_force`, and `relative_difference`.

    Raises:
    - `ValueError`: If `section_offset <= control_radius`.
    """
    radius = particle_radius if control_radius is None else control_radius
    if section_offset <= radius:
        raise ValueError("section_offset must exceed the control radius.")
    surface = sphere_traction_force(network, center_z, radius, device=device, dtype=dtype)
    momentum = cross_section_momentum_force(
        network, channel_radius, center_z - section_offset, center_z + section_offset,
        device=device, dtype=dtype,
    )
    scale = max(abs(surface), abs(momentum), 1.0e-12)
    return {"surface_force": surface, "momentum_force": momentum,
            "relative_difference": abs(surface - momentum) / scale}


def residual_by_region(
    network: FlowField,
    domain: Domain,
    fluid_config: FluidConfig,
    particle_axial_position: float,
    particle_radius: float,
    n_points: int = 20_000,
    random_seed: int = 987_654,
    residual_form: str = "standard",
    near_factor: float = 2.5,
    axis_band: float = 0.05,
    section_offset: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> pd.DataFrame:
    r"""Evaluate the Stokes residual on fresh points, split by region, with a force bound.

    Judging with `residual_form="standard"` is deliberate. The `r_weighted` form is the
    standard form times $r$ or $r^2$, so it can look small where the physics is not satisfied.
    Every statistic is volume-weighted (weight $r$), because the force identity
    $F_\text{momentum}-F_\text{surface}=\int f_z\,\dd V$ is a volume integral.

    Args:
    - `network`: Flow field mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Nondimensional channel.
    - `fluid_config`: Stokes fluid configuration.
    - `particle_axial_position`: Sphere center.
    - `particle_radius`: Sphere radius.
    - `n_points`: Number of fresh points (area-uniform in the meridian plane, outside the sphere).
    - `random_seed`: Seed, independent of any training seed.
    - `residual_form`: `"standard"` (default) or `"r_weighted"`.
    - `near_factor`: The "near sphere" region is $\rho < $ this factor times $a$.
    - `axis_band`: The "axis band" region is $r <$ this fraction of $R$.
    - `section_offset`: If given, adds a `slab` row on $\abs{z - z_p} <$ offset and fills
      `force_error_bound` $= V_\text{slab}\,\mathrm{rms}_V(f_z)$ for it.
    - `device`, `dtype`: Placement overrides for plain callables.

    Returns:
    - A `DataFrame` indexed by region with columns `count`, `rms_r`, `rms_z`, `rms_continuity`
      and `force_error_bound` (NaN except for the `slab` row).

    Raises:
    - `ValueError`: If an argument is out of range.
    """
    if n_points <= 0 or near_factor <= 1.0 or not (0.0 < axis_band < 1.0):
        raise ValueError("n_points > 0, near_factor > 1 and 0 < axis_band < 1 are required.")
    resolved_device, resolved_dtype = _resolve_placement(network, device, dtype)
    points = sample_exterior_points(
        domain, particle_axial_position, particle_radius, n_points, random_seed,
        shell_fraction=0.0, device=resolved_device, dtype=resolved_dtype,
    ).requires_grad_(True)
    with torch.enable_grad():
        residual = stokes_residual(network, points, fluid_config, residual_form=residual_form).detach()
    r, z = points[:, 0].detach(), points[:, 1].detach()
    rho = torch.hypot(r, z - particle_axial_position)
    masks = {"all": torch.ones_like(r, dtype=torch.bool),
             "near sphere": rho < near_factor * particle_radius,
             "axis band": r < axis_band * domain.radius}
    if section_offset is not None:
        masks["slab"] = (z - particle_axial_position).abs() < section_offset
    nan = float("nan")
    rows = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            rows[name] = {"count": 0, "rms_r": nan, "rms_z": nan, "rms_continuity": nan,
                          "force_error_bound": nan}
            continue
        weight = r[mask]
        rms = [torch.sqrt((weight * residual[mask, k].square()).sum() / weight.sum()).item() for k in range(3)]
        bound = nan
        if name == "slab":
            slab_volume = math.pi * domain.radius**2 * 2.0 * section_offset - 4.0 / 3.0 * math.pi * particle_radius**3
            bound = slab_volume * rms[1]
        rows[name] = {"count": int(mask.sum()), "rms_r": rms[0], "rms_z": rms[1],
                      "rms_continuity": rms[2], "force_error_bound": bound}
    return pd.DataFrame.from_dict(rows, orient="index")