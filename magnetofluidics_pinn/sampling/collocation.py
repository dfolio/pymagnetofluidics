"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

import math

import torch

from magnetofluidics_pinn.config import ParticleConfig
from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.types import CollocationPoints, Domain, ParticleState

# Fraction of the channel radius kept clear around the symmetry axis
# (r = 0) when drawing interior points, used whenever a caller does not
# override it via `sample_collocation_points`'s `axis_clearance_fraction`
# argument. The axisymmetric Stokes residual (see
# `physics.fluid_residuals.stokes_residual`) involves terms proportional to
# `1 / r` and `1 / r**2` under its default `residual_form="standard"`,
# which are singular exactly on the axis. A clearance that is too tight is
# not just a removable-singularity concern but a numerical-stability one:
# at 1e-4 * radius, `1 / r**2` amplifies an untrained network's (generally
# nonzero) `u_r` by a factor of order 1e8, so any collocation point that
# happens to land close to the axis can produce a residual outlier many
# orders of magnitude larger than the rest of the batch, destabilizing the
# mean-squared PDE loss used during training. 5e-2 keeps that amplification
# bounded (~4e2) while still leaving the axis itself well-sampled.
#
# CHANGED: this constant is now only the *default* clearance rather than
# the only one — see `axis_clearance_fraction` below. Under
# `residual_form="r_weighted"`
# (`physics.fluid_residuals.stokes_residual`/`navier_stokes_residual`), the
# 1/r, 1/r**2 terms are removed analytically before this instability can
# arise, so a caller using that residual form may safely pass a smaller
# value, down to and including `0.0` (flush with the axis).
_AXIS_CLEARANCE_FRACTION = 5.0e-2
# NEW. Default safety margin for the excluded-particle rejection sampler in
# `sample_collocation_points_with_particle`: draw this many candidate
# interior points per point ultimately requested, before discarding the
# ones that fall inside the particle. The particle disk's own area is at
# most `particle.radius**2 / domain.radius**2` of the channel's r-z cross
# section (equality only if the sphere spans the full radius); a factor of
# 4 comfortably covers every particle size this module's own validation
# allows (see `sample_collocation_points_with_particle`'s radius check)
# without materially over-drawing for a small, realistic particle.
_DEFAULT_OVERSAMPLING_FACTOR = 4.0


def boundary_face_sizes(n_boundary: int) -> tuple[int, int, int]:
    """Split a boundary point budget across the three faces of a channel.

    A straight channel has three boundary faces: the lateral wall
    (`r = domain.radius`), the inlet cross-section (`z = 0`), and the outlet
    cross-section (`z = domain.length`). This helper is shared by
    [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
    and by the training loop, which needs to know how `CollocationPoints.boundary`
    is laid out (wall points first, then inlet points, then outlet points) in
    order to apply the matching boundary condition to each subset.

    Args:
    - `n_boundary`: Total number of boundary collocation points to allocate.

    Returns:
    - A tuple `(n_wall, n_inlet, n_outlet)` of non-negative integers summing
      to `n_boundary`; any remainder from integer division is assigned to
      the outlet face.
    """
    n_wall = n_boundary // 3
    n_inlet = n_boundary // 3
    n_outlet = n_boundary - n_wall - n_inlet
    return n_wall, n_inlet, n_outlet


def sample_collocation_points(
        domain: Domain,
        n_interior: int,
        n_boundary: int,
        random_seed: int,
        n_initial: int | None = None,
        axis_clearance_fraction: float | None = None,  # NEW
        device: str | torch.device | None = None,
) -> CollocationPoints:
    """Sample interior, boundary, and (optionally) initial points.

    Args:
    - `domain`: Vessel geometry to sample from.
    - `n_interior`: Number of interior points to sample.
    - `n_boundary`: Number of boundary points to sample.
    - `random_seed`: Seed guaranteeing reproducible sampling.
    - `n_initial`: Number of initial-time points to sample; `None` for
      steady-state problems (e.g., Stokes flow).
    - `axis_clearance_fraction`: NEW. Fraction of `domain.radius` kept clear
      of the symmetry axis when drawing interior points, overriding this
      module's `_AXIS_CLEARANCE_FRACTION` default. `None` (the default)
      keeps that module default, reproducing every prior release's sampling
      exactly. `0.0` samples flush to the axis; only do so together with
      `residual_form="r_weighted"` on whichever residual function consumes
      the returned points (`physics.fluid_residuals.stokes_residual` /
      `navier_stokes_residual`) — this function itself does not know which
      residual form the caller intends to pair it with, so it cannot enforce
      that pairing; see `TrainingConfig.__post_init__` for where that
      cross-field check is actually made, for the `train()` entry point.
    - `device`: Device the returned tensors are created on directly, e.g.
      `"cuda"`, `"cuda:0"`, or `"cpu"`; resolved automatically when `None`.
      Tensors are created directly on this device (auto-selecting CUDA
      when available, matching `build_mlp` and `train`) rather than
      unconditionally on the CPU, avoiding a CPU -> GPU copy every epoch
      when training on a CUDA device. Note that a CPU generator and a CUDA
      generator seeded identically produce *different* draws (they use
      different underlying RNG algorithms): reproducibility is guaranteed
      per device type, not across device types.

    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance holding the sampled tensors, all on the resolved device.
      `boundary` concatenates, in order, the wall, inlet, and outlet
      subsets sized by
      [`boundary_face_sizes`][magnetofluidics_pinn.sampling.collocation.boundary_face_sizes].

    Raises:
    - `ValueError`: If `n_interior` or `n_boundary` is not strictly positive,
      if `domain.kind` is not `"channel"` (the only geometry this sampler
      currently supports), if `axis_clearance_fraction` is not `None` and
      does not lie in `[0.0, 1.0)`, or if `device` is not a valid device
      string.
    - `RuntimeError`: If `device` (or the auto-selected default) resolves to
      a CUDA device but no CUDA device is available.
    - `NotImplementedError`: If `n_initial` is not `None`; initial-time
      sampling is only meaningful for unsteady (Navier-Stokes) problems,
      scheduled for a later roadmap phase.
    """
    if n_interior <= 0 or n_boundary <= 0:
        raise ValueError("n_interior and n_boundary must be strictly positive.")
    if domain.kind != "channel":
        raise ValueError(
            "sample_collocation_points currently only supports "
            f"domain.kind == 'channel'; got {domain.kind!r}."
        )
    if n_initial is not None:
        raise NotImplementedError(
            "Initial-time collocation sampling is scheduled for the "
            "unsteady (Navier-Stokes) phase of the roadmap."
        )
    if axis_clearance_fraction is not None and not (0.0 <= axis_clearance_fraction < 1.0):
        raise ValueError(
            "axis_clearance_fraction must be None, or lie in [0.0, 1.0); "
            f"got {axis_clearance_fraction!r}."
        )
    resolved_axis_clearance_fraction = (
        _AXIS_CLEARANCE_FRACTION if axis_clearance_fraction is None else axis_clearance_fraction
    )
    resolved_device = resolve_device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(random_seed)
    
    # Interior points: uniform in z, but *volume*-uniform in r rather than
    # linearly uniform. Revolving an annulus at radius r around the axis
    # gives it measure proportional to r dr, so drawing r linearly from
    # Uniform(r_min, R) over-samples the region near the axis and
    # under-samples the region near the wall relative to how much of the
    # domain's actual volume each represents - exactly the wall-adjacent
    # region where accuracy matters most for mass conservation (see
    # `physics.fluid_residuals`). Inverting the CDF of the r dr measure,
    # F(r) = (r^2 - r_min^2) / (R^2 - r_min^2), gives the correct draw:
    # r = sqrt(r_min^2 + (R^2 - r_min^2) * xi), xi ~ Uniform(0, 1).
    r_min = domain.radius * resolved_axis_clearance_fraction  # CHANGED: was the module constant directly.
    xi = torch.rand(n_interior, 1, generator=generator, device=resolved_device)
    interior_r = torch.sqrt(r_min ** 2 + (domain.radius ** 2 - r_min ** 2) * xi)
    interior_z = domain.length * torch.rand(n_interior, 1, generator=generator, device=resolved_device)
    interior = torch.cat([interior_r, interior_z], dim=1)
    
    n_wall, n_inlet, n_outlet = boundary_face_sizes(n_boundary)
    
    wall_points = torch.cat(
        [
            torch.full((n_wall, 1), domain.radius, device=resolved_device),
            domain.length * torch.rand(n_wall, 1, generator=generator, device=resolved_device),
        ],
        dim=1,
    )
    inlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_inlet, 1, generator=generator, device=resolved_device),
            torch.zeros(n_inlet, 1, device=resolved_device),
        ],
        dim=1,
    )
    outlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_outlet, 1, generator=generator, device=resolved_device),
            torch.full((n_outlet, 1), domain.length, device=resolved_device),
        ],
        dim=1,
    )
    boundary = torch.cat([wall_points, inlet_points, outlet_points], dim=0)
    
    return CollocationPoints(interior=interior, boundary=boundary, initial=None)


def _sample_particle_shell_points(
        generator: torch.Generator,
        n_points: int,
        particle_radius: float,
        axial_center: float,
        shell_thickness: float,
        channel_radius: float,
        channel_length: float,
        min_radius: float,
        oversampling_factor: float,
        device: torch.device,
) -> torch.Tensor:
    r"""Draw volume-uniform points in the spherical shell $a < \rho < a + \delta$ around the particle.

    A 3-D volume-uniform draw has $\rho^3$ uniform on $[a^3, (a+\delta)^3]$ and $\cos\theta$ uniform on
    $[-1, 1]$; projecting to $(r, z) = (\rho\sin\theta, z_c + \rho\cos\theta)$ keeps the $r\,\mathrm{d}r$
    measure used by the global sampler. Candidates outside the channel, or closer to the axis than
    `min_radius`, are rejected.

    Args:
    - `generator`: Seeded generator on `device`.
    - `n_points`: Number of accepted points requested (`0` returns an empty `(0, 2)` tensor).
    - `particle_radius`, `axial_center`, `shell_thickness`: Shell geometry $a$, $z_c$, $\delta$.
    - `channel_radius`, `channel_length`: Domain bounds $R$, $L$.
    - `min_radius`: Axis clearance, as in the other samplers.
    - `oversampling_factor`: Candidates drawn per requested point; must exceed 1.
    - `device`: Target device.

    Returns:
    - Tensor of shape `(n_points, 2)` with `(r, z)` rows.

    Raises:
    - `ValueError`: If `n_points` is negative or `oversampling_factor <= 1`.
    - `RuntimeError`: If fewer than `n_points` candidates are accepted.
    """
    if n_points < 0:
        raise ValueError(f"n_points must be non-negative; got {n_points!r}.")
    if oversampling_factor <= 1.0:
        raise ValueError(f"oversampling_factor must be strictly greater than 1.0; got {oversampling_factor!r}.")
    if n_points == 0:
        return torch.empty((0, 2), device=device)
    n_candidates = math.ceil(oversampling_factor * n_points)
    inner_cubed = particle_radius ** 3
    outer_cubed = (particle_radius + shell_thickness) ** 3
    distance = (inner_cubed + (outer_cubed - inner_cubed)
                * torch.rand(n_candidates, 1, generator=generator, device=device)).pow(1.0 / 3.0)
    cosine = 2.0 * torch.rand(n_candidates, 1, generator=generator, device=device) - 1.0
    radial = distance * torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
    axial = axial_center + distance * cosine
    admissible = ((radial >= min_radius) & (radial < channel_radius)
                  & (axial > 0.0) & (axial < channel_length)).squeeze(1)
    points = torch.cat([radial, axial], dim=1)[admissible][:n_points]
    if points.shape[0] < n_points:
        raise RuntimeError(f"Insufficient shell candidates survived: needed {n_points}, got {points.shape[0]}.")
    return points


def sample_collocation_points_with_particle(
        domain: Domain,
        particle_state: ParticleState,
        particle_config: ParticleConfig,
        n_interior: int,
        n_boundary: int,
        n_surface_points: int,
        random_seed: int,
        axis_clearance_fraction: float | None = None,
        constriction_oversample_ratio: float = 0.35,  # NEW: 35% focused in bypass clearance
        constriction_axial_factor: float = 1.5,       # NEW: Axial span |z - z_p| <= 1.5 * a
        oversampling_factor: float = _DEFAULT_OVERSAMPLING_FACTOR,
        boundary_layer_fraction: float = 0.25,  # NEW: 25% of points focused near particle
        boundary_layer_radius_factor: float = 2.5,  # NEW: Shell thickness up to 2.5 * a
        device: str | torch.device | None = None,
) -> CollocationPoints:
    r"""Sample interior, boundary, and particle-surface points around an embedded sphere.
 
    CHANGED: Implements multizone collocation sampling. Allocates `constriction_oversample_ratio`
    (default 35%) of interior collocation points directly to the high-shear annular constriction
    cylinder $|z - z_p| \le 1.5 a, \, r \in [a, R]$, resolving the accelerated bypass jet and
    preventing mass-flow deficit across the obstacle [@hu2025pecann; @happel1983low].
 
    Args:
    - `domain`: Vessel geometry to sample from; must already be
      nondimensionalized.
    - `particle_state`: The embedded particle's state; must fit strictly inside `domain`
      (see Raises).
    - `particle_config`: The embedded particle's configuration; must fit strictly inside `domain`.
    - `n_interior`: Number of interior (fluid-only) points to sample.
    - `n_boundary`: Number of vessel-boundary (wall/inlet/outlet) points to
      sample; split exactly as
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
      splits it.
    - `n_surface_points`: Number of points to sample on the particle's
      own surface.
    - `random_seed`: Seed guaranteeing reproducible sampling.
    - `axis_clearance_fraction`: Same meaning as
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]'s
      argument of the same name; applies only to the *candidate* interior
      draw, before particle rejection.
    - `constriction_oversample_ratio`: Fraction of interior budget allocated to constriction gap.
    - `constriction_axial_factor`: Axial extent multiplier determining constriction zone span.
    - `oversampling_factor`: How many candidate interior points to draw
      per point ultimately requested, before discarding the ones that fall
      inside the particle. Must be strictly greater than `1.0`.
    - `device`: Device the returned tensors are created on directly.
 
    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance with `particle_surface` populated (never `None`).
 
    Raises:
    - `ValueError`: If `n_interior`, `n_boundary`, or `n_surface_points`
      is not strictly positive, if `domain.kind` is not `"channel"`, if
      `oversampling_factor` is not strictly greater than `1.0`, if
      `axis_clearance_fraction` is invalid (see
      `sample_collocation_points`), or if `particle_config` does not fit strictly
      inside `domain` — i.e. `particle_config.radius >= domain.radius` (the
      sphere would touch or exceed the channel wall) or the sphere's axial
      extent `[axial_position - radius, axial_position + radius]` is not
      strictly contained in `[0, domain.length]` (the sphere would touch
      or cross the inlet or outlet).
    - RuntimeError: If, after drawing `oversampling_factor * n_interior`
      candidates, fewer than `n_interior` survive particle rejection —
      raise `oversampling_factor`, or draw a smaller `n_interior`, rather
      than silently training on fewer points than requested.
    """
    if n_interior <= 0 or n_boundary <= 0 or n_surface_points <= 0:
        raise ValueError("n_interior, n_boundary, and n_surface_points must be strictly positive.")
    if domain.kind != "channel":
        raise ValueError(
            "sample_collocation_points_with_particle currently only supports "
            f"domain.kind == 'channel'; got {domain.kind!r}."
        )

    if particle_config.radius >= domain.radius:
        raise ValueError(
            f"particle.radius ({particle_config.radius!r}) must be strictly less than domain.radius ({domain.radius!r})."
        )
    if oversampling_factor <= 1.0:
        raise ValueError(f"oversampling_factor must be strictly greater than 1.0; got {oversampling_factor!r}.")
    if axis_clearance_fraction is not None and not (0.0 <= axis_clearance_fraction < 1.0):
        raise ValueError(
            "axis_clearance_fraction must be None, or lie in [0.0, 1.0); "
            f"got {axis_clearance_fraction!r}."
        )

    # obstacle_z_min = particle_state.axial_position - particle_config.radius
    # obstacle_z_max = particle_state.axial_position + particle_config.radius
    # if not (0.0 < obstacle_z_min and obstacle_z_max < domain.length):
    #     raise ValueError(
    #         "obstacle must fit strictly inside the channel's axial extent (0, domain.length); "
    #         f"got axial span [{obstacle_z_min!r}, {obstacle_z_max!r}] against domain.length="
    #         f"{domain.length!r}."
    #     )
    
    resolved_axis_clearance_fraction = (
        _AXIS_CLEARANCE_FRACTION if axis_clearance_fraction is None else axis_clearance_fraction
    )
    resolved_device = resolve_device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(random_seed)
    
    a = particle_config.radius
    z_p = particle_state.axial_position
    
    # 1. Budget Partition: Constriction Zone vs. Global Background
    n_constriction = int(n_interior * constriction_oversample_ratio)
    n_shell = int(n_interior * boundary_layer_fraction)  # NEW
    n_global = n_interior - n_constriction - n_shell
    if n_global < 0:  # NEW
        raise ValueError("constriction_oversample_ratio + boundary_layer_fraction must not exceed 1.")

    # 2. Constriction-Zone Collocation Sampling: |z - z_p| <= 1.5 * a, r in [a, R]
    z_gap_min = max(z_p - constriction_axial_factor * a, 0.0)
    z_gap_max = min(z_p + constriction_axial_factor * a, domain.length)
    z_gap_length = z_gap_max - z_gap_min
    
    n_constriction_candidates = math.ceil(oversampling_factor * n_constriction)
    r_min = domain.radius * resolved_axis_clearance_fraction
    xi_gap = torch.rand(n_constriction_candidates, 1, generator=generator, device=resolved_device)
    gap_cand_r = torch.sqrt(r_min ** 2 + (domain.radius ** 2 - r_min ** 2) * xi_gap)
    gap_cand_z = z_gap_min + z_gap_length * torch.rand(n_constriction_candidates, 1, generator=generator,
                                                       device=resolved_device)
    
    outside_gap_particle = (gap_cand_z - z_p).square() + gap_cand_r.square() >= a ** 2
    surviving_gap_r = gap_cand_r[outside_gap_particle][:n_constriction]
    surviving_gap_z = gap_cand_z[outside_gap_particle][:n_constriction]
    
    if surviving_gap_r.shape[0] < n_constriction:
        raise RuntimeError(
            f"Insufficient candidates survived constriction sampling: needed {n_constriction}, got {surviving_gap_r.shape[0]}."
        )
    
    # 3. Global Background Rejection Sampling across [0, domain.length]
    n_global_candidates = math.ceil(oversampling_factor * n_global)
    xi_glob = torch.rand(n_global_candidates, 1, generator=generator, device=resolved_device)
    glob_cand_r = torch.sqrt(r_min ** 2 + (domain.radius ** 2 - r_min ** 2) * xi_glob)
    glob_cand_z = domain.length * torch.rand(n_global_candidates, 1, generator=generator, device=resolved_device)
    
    outside_glob_particle = (glob_cand_z - z_p).square() + glob_cand_r.square() >= a ** 2
    surviving_glob_r = glob_cand_r[outside_glob_particle][:n_global]
    surviving_glob_z = glob_cand_z[outside_glob_particle][:n_global]
    
    if surviving_glob_r.shape[0] < n_global:
        raise RuntimeError(
            f"Insufficient candidates survived global sampling: needed {n_global}, got {surviving_glob_r.shape[0]}."
        )
    
    # Merge multizone coordinates
    shell_points = _sample_particle_shell_points(                             # NEW
          generator, n_shell, a, z_p, (boundary_layer_radius_factor - 1.0) * a,
          domain.radius, domain.length, r_min, oversampling_factor, resolved_device)
    interior_r = torch.cat([surviving_gap_r, surviving_glob_r], dim=0)
    interior_z = torch.cat([surviving_gap_z, surviving_glob_z], dim=0)
    interior = torch.cat([torch.stack([interior_r, interior_z], dim=1),
                          shell_points], dim=0)  # NEW: add shell points to interior
    
    # 4. Channel External Boundary Points (Wall, Inlet, Outlet)
    n_wall, n_inlet, n_outlet = boundary_face_sizes(n_boundary)
    wall_points = torch.cat(
        [
            torch.full((n_wall, 1), domain.radius, device=resolved_device),
            domain.length * torch.rand(n_wall, 1, generator=generator, device=resolved_device),
        ],
        dim=1,
    )
    inlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_inlet, 1, generator=generator, device=resolved_device),
            torch.zeros(n_inlet, 1, device=resolved_device),
        ],
        dim=1,
    )
    outlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_outlet, 1, generator=generator, device=resolved_device),
            torch.full((n_outlet, 1), domain.length, device=resolved_device),
        ],
        dim=1,
    )
    boundary = torch.cat([wall_points, inlet_points, outlet_points], dim=0)
    
    # 5. Particle Surface Collocation Points (Uniform meridian polar angle theta in [0, pi])
    theta_surf = math.pi * torch.rand(n_surface_points, 1, generator=generator, device=resolved_device)
    obstacle_surface = torch.cat(
        [
            a * torch.sin(theta_surf),
            z_p + a * torch.cos(theta_surf),
        ],
        dim=1,
    )
    
    return CollocationPoints(
        interior=interior,
        boundary=boundary,
        initial=None,
        particle_surface=obstacle_surface,
    )
