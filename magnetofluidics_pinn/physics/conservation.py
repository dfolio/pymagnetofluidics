r"""Global mass-conservation diagnostics for the axisymmetric flow field.

[`physics.fluid_residuals.stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
only enforces the continuity equation pointwise, at a finite set of
collocation points, as a soft loss term. A trained field can satisfy that
residual to a small tolerance almost everywhere and *still* let small,
systematically-signed local violations accumulate, along the channel's
axial extent, into a macroscopically wrong volumetric flow rate: integrating
the axisymmetric continuity equation over a cross-section gives
$$\frac{\partial Q}{\partial z} = -2\pi R\, u_r(R, z), \qquad Q(z) = \int_0^R u_z(r, z)\, 2\pi r \dd{r}, $$
so $Q(z)$ is exactly constant only where the pointwise continuity residual
is exactly zero - not merely small on average. This module adds no new
physics beyond what `stokes_residual` already encodes: it is a
differentiable *diagnostic and constraint* function, integrating the
network's own predicted axial velocity across a cross-section, so that a
loss term can penalize the resulting flow rate directly for its
(in)constancy along $z$ - the macroscopic quantity a person actually
inspects when checking mass conservation - rather than trusting the
pointwise residual alone to imply it
[@hu2025pecann; @sigalingging2026massconserving].
"""

from __future__ import annotations

import math
from typing import Callable

import torch


def annular_flow_rate(
    network: Callable[[torch.Tensor], torch.Tensor],
    channel_radius: float,
    particle_radius: float,
    particle_axial_position: float,
    particle_velocity: float,
    axial_positions: torch.Tensor,
    n_radial_quadrature_points: int = 64,
) -> torch.Tensor:
    r"""Evaluate volumetric flow rate $Q(z)$ across cross-sections cutting through an obstacle.

    For cross-sections intersecting a sphere ($|z - z_p| \le a$), the fluid occupies
    the annular clearance $r \in [r_{\mathrm{core}}(z), R]$, where:
    $$
    r_{\mathrm{core}}(z) = \sqrt{\max\left(0, a^2 - (z - z_p)^2\right)}.
    $$
    The total volumetric transport combines fluid flux through the annulus and
    rigid-body convective transport of the obstacle core [@happel1983low]:
    $$
    Q(z) = \int_{r_{\mathrm{core}}(z)}^R u_z(r, z) \, 2\pi r \, dr + \pi r_{\mathrm{core}}(z)^2 U_p.
    $$
    For cross-sections completely outside the obstacle ($|z - z_p| > a$),
    $r_{\mathrm{core}}(z) = 0$, recovering the full-bore pipe flow rate.

    Args:
    - `network`: Flow field callable mapping `(n, 2)` coordinates to `(n, 3)`.
    - `channel_radius`: Dimensionless channel radius $R$.
    - `particle_radius`: Dimensionless sphere radius $a$.
    - `particle_axial_position`: Dimensionless sphere center $z_p$.
    - `particle_velocity`: Dimensionless translational axial velocity $U_p$.
    - `axial_positions`: 1-D tensor of shape `(n_stations,)` with $z$ coordinates.
    - `n_radial_quadrature_points`: Trapezoidal integration nodes across the radial span.

    Returns:
    - Tensor of shape `(n_stations,)` containing total volumetric flow rate $Q(z)$.

    Raises:
    - `ValueError`: If dimensions or radii are non-positive or invalid.
    """
    if channel_radius <= 0.0 or particle_radius <= 0.0:
        raise ValueError("channel_radius and particle_radius must be strictly positive.")
    if axial_positions.ndim != 1:
        raise ValueError(f"axial_positions must be 1-D; got shape {tuple(axial_positions.shape)}.")
    if n_radial_quadrature_points < 2:
        raise ValueError("n_radial_quadrature_points must be at least 2.")

    n_stations = axial_positions.shape[0]
    device = axial_positions.device
    dtype = axial_positions.dtype

    # Axial distance to particle center and radial extent of the solid sphere at each station
    z_diff = axial_positions - particle_axial_position
    r_core_sq = torch.clamp(particle_radius**2 - z_diff.square(), min=0.0)
    r_core = torch.sqrt(r_core_sq)

    # Parametrized radial grid mapping [0, 1] linearly to [r_core(z), R]
    s_grid = torch.linspace(0.0, 1.0, n_radial_quadrature_points, device=device, dtype=dtype)
    r_span = channel_radius - r_core  # Shape: (n_stations,)
    radial_mesh = r_core.unsqueeze(1) + s_grid.unsqueeze(0) * r_span.unsqueeze(1)  # (n_stations, n_nodes)
    axial_mesh = axial_positions.unsqueeze(1).expand(-1, n_radial_quadrature_points)

    coordinates = torch.stack([radial_mesh.reshape(-1), axial_mesh.reshape(-1)], dim=1)
    axial_velocity = network(coordinates)[:, 1:2].reshape(n_stations, n_radial_quadrature_points)

    # Integrand 2 * pi * r * u_z(r, z)
    integrand = axial_velocity * radial_mesh * (2.0 * math.pi)

    # Step size per station for composite trapezoidal rule: dr = (R - r_core) / (n_nodes - 1)
    dr = r_span / (n_radial_quadrature_points - 1)
    fluid_flux = torch.trapezoid(integrand, dx=1.0, dim=1) * dr

    # Add convective flux carried by the rigid obstacle core
    core_flux = math.pi * r_core_sq * particle_velocity
    return fluid_flux + core_flux


def axial_flow_rate(
    network: Callable[[torch.Tensor], torch.Tensor],
    radius: float,
    axial_positions: torch.Tensor,
    n_radial_quadrature_points: int = 64,
) -> torch.Tensor:
    r"""Evaluate the volumetric flow rate at a set of axial stations.

    Computes

    $$ Q(z) = \int_0^R u_z(r, z)\, 2\pi r \dd{r} $$

    at each requested $z$, via composite trapezoidal quadrature over a
    fixed radial grid, by directly querying `network`'s predicted axial
    velocity. Differentiable end-to-end with respect to `network`'s
    parameters (and to `axial_positions`), so its output is usable both
    inside a training loss (see `training.trainer`) and for post-hoc
    evaluation, e.g. in a verification notebook.

    This is a pure, orchestration-adjacent function in the sense of
    `device_utils`'s convention: it creates the radial quadrature grid on
    whatever device and dtype `axial_positions` already has, rather than
    resolving a device itself.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, 2)` to
      `(n_points, 3)`, the `(u_r, u_z, p)` flow field (trained, or
      in-training).
    - `radius`: Channel radius, in the same (dimensionless) units the
      network was trained on.
    - `axial_positions`: Tensor of shape `(n_stations,)` with the $z$
      coordinates at which $Q(z)$ is evaluated.
    - `n_radial_quadrature_points`: Number of radial quadrature nodes per
      station; higher values reduce quadrature error at the cost of a
      larger network evaluation batch.

    Returns:
    - Tensor of shape `(n_stations,)` with $Q(z)$ evaluated at each
      requested axial position.

    Raises:
    - `ValueError`: If `radius` is not strictly positive, if
      `axial_positions` is not a 1-D tensor, or if
      `n_radial_quadrature_points` is smaller than 2 (trapezoidal
      quadrature requires at least two nodes).
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be strictly positive; got {radius!r}.")
    if axial_positions.ndim != 1:
        raise ValueError(
            f"axial_positions must be a 1-D tensor; got shape {tuple(axial_positions.shape)}."
        )
    if n_radial_quadrature_points < 2:
        raise ValueError(
            "n_radial_quadrature_points must be at least 2 for trapezoidal "
            f"quadrature; got {n_radial_quadrature_points!r}."
        )

    n_stations = axial_positions.shape[0]
    radial_grid = torch.linspace(
        0.0, radius, n_radial_quadrature_points,
        dtype=axial_positions.dtype, device=axial_positions.device,
    )

    # Broadcast every station against every radial node into a single
    # (n_stations * n_radial_quadrature_points, 2) coordinate batch, so the
    # network is evaluated once rather than once per station.
    radial_mesh = radial_grid.unsqueeze(0).expand(n_stations, -1)
    axial_mesh = axial_positions.unsqueeze(1).expand(-1, n_radial_quadrature_points)
    coordinates = torch.stack([radial_mesh.reshape(-1), axial_mesh.reshape(-1)], dim=1)

    axial_velocity = network(coordinates)[:, 1:2].reshape(n_stations, n_radial_quadrature_points)
    integrand = axial_velocity * radial_mesh * (2.0 * math.pi)

    # Composite trapezoidal rule along the radial (quadrature) axis.
    return torch.trapezoid(integrand, radial_grid, dim=1)


def poiseuille_reference_flow_rate(radius: float, peak_velocity: float) -> float:
    r"""Closed-form volumetric flow rate of an exact Hagen-Poiseuille profile.

    For the parabolic profile imposed by
    [`inlet_velocity_condition`][magnetofluidics_pinn.boundary_conditions.flow_bc.inlet_velocity_condition],
    $u_z(r) = u_\mathrm{peak}\left(1 - (r/R)^2\right)$, the flow rate
    integrates in closed form to $Q_\mathrm{ref} = \pi R^2 u_\mathrm{peak} / 2$.
    Used as the conservation target for
    [`axial_flow_rate`][magnetofluidics_pinn.physics.conservation.axial_flow_rate]:
    since the inlet condition already pins the network to this exact
    profile at $z = 0$, mass conservation requires $Q(z) = Q_\mathrm{ref}$
    at *every* $z$, not only at the inlet.

    Args:
    - `radius`: Channel radius, dimensionless, matching the network's
      training domain.
    - `peak_velocity`: Dimensionless centerline velocity, matching the
      value passed to `inlet_velocity_condition`.

    Returns:
    - The reference flow rate $Q_\mathrm{ref} = \pi R^2 u_\mathrm{peak} / 2$.

    Raises:
    - `ValueError`: If `radius` or `peak_velocity` is not strictly positive.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be strictly positive; got {radius!r}.")
    if peak_velocity <= 0.0:
        raise ValueError(f"peak_velocity must be strictly positive; got {peak_velocity!r}.")
    return math.pi * radius**2 * peak_velocity / 2.0