"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

import math

import torch

from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.types import CollocationPoints, Domain, SphericalObstacle

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
# NEW. Default safety margin for the excluded-obstacle rejection sampler in
# `sample_collocation_points_with_obstacle`: draw this many candidate
# interior points per point ultimately requested, before discarding the
# ones that fall inside the obstacle. The obstacle disk's own area is at
# most `obstacle.radius**2 / domain.radius**2` of the channel's r-z cross
# section (equality only if the sphere spans the full radius); a factor of
# 4 comfortably covers every obstacle size this module's own validation
# allows (see `sample_collocation_points_with_obstacle`'s radius check)
# without materially over-drawing for a small, realistic obstacle.
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
    # NEW: validate and resolve the axis-clearance override independently of
    # any caller (e.g. TrainingConfig) that might also validate it, since
    # this function is public and directly callable on its own.
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
    interior_r = torch.sqrt(r_min**2 + (domain.radius**2 - r_min**2) * xi)
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
 
 
def sample_collocation_points_with_obstacle(
    domain: Domain,
    obstacle: SphericalObstacle,
    n_interior: int,
    n_boundary: int,
    n_obstacle_surface: int,
    random_seed: int,
    axis_clearance_fraction: float | None = None,
    oversampling_factor: float = _DEFAULT_OVERSAMPLING_FACTOR,
    device: str | torch.device | None = None,
) -> CollocationPoints:
    r"""Sample interior, boundary, and obstacle-surface points around an embedded sphere.
 
    NEW. The two-way-coupled counterpart of
    [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]:
    wall, inlet, and outlet points are drawn exactly as that function
    already does (the vessel's own boundary is unaffected by an obstacle
    inside it), but interior points are drawn by *rejection sampling* —
    oversample from the same volume-correct distribution, then discard
    any point that falls inside the obstacle — and a new, third boundary
    face is added: points on the obstacle's own surface, where
    [`training.trainer.train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle]
    enforces the particle's rigid-body velocity via
    [`boundary_conditions.rigid_body_velocity_condition`][magnetofluidics_pinn.boundary_conditions.flow_bc.rigid_body_velocity_condition].
 
    **Rejection sampling never silently under-delivers.** Unlike a naive
    "oversample and slice" approach, this function raises rather than
    silently returning fewer than `n_interior` points if the requested
    `oversampling_factor` was not generous enough — a caller that received
    a short batch without an error would train on an unrequested, smaller
    (and non-reproducibly-sized, seed-to-seed) collocation set without
    ever knowing it.
 
    Args:
    - `domain`: Vessel geometry to sample from; must already be
      nondimensionalized.
    - `obstacle`: The embedded sphere; must fit strictly inside `domain`
      (see Raises).
    - `n_interior`: Number of interior (fluid-only) points to sample.
    - `n_boundary`: Number of vessel-boundary (wall/inlet/outlet) points to
      sample; split exactly as
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
      splits it.
    - `n_obstacle_surface`: Number of points to sample on the obstacle's
      own surface.
    - `random_seed`: Seed guaranteeing reproducible sampling.
    - `axis_clearance_fraction`: Same meaning as
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]'s
      argument of the same name; applies only to the *candidate* interior
      draw, before obstacle rejection.
    - `oversampling_factor`: How many candidate interior points to draw
      per point ultimately requested, before discarding the ones that fall
      inside the obstacle. Must be strictly greater than `1.0`.
    - `device`: Device the returned tensors are created on directly.
 
    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance with `obstacle_surface` populated (never `None`).
 
    Raises:
    - `ValueError`: If `n_interior`, `n_boundary`, or `n_obstacle_surface`
      is not strictly positive, if `domain.kind` is not `"channel"`, if
      `oversampling_factor` is not strictly greater than `1.0`, if
      `axis_clearance_fraction` is invalid (see
      `sample_collocation_points`), or if `obstacle` does not fit strictly
      inside `domain` — i.e. `obstacle.radius >= domain.radius` (the
      sphere would touch or exceed the channel wall) or the sphere's axial
      extent `[axial_position - radius, axial_position + radius]` is not
      strictly contained in `[0, domain.length]` (the sphere would touch
      or cross the inlet or outlet).
    - RuntimeError: If, after drawing `oversampling_factor * n_interior`
      candidates, fewer than `n_interior` survive obstacle rejection —
      raise `oversampling_factor`, or draw a smaller `n_interior`, rather
      than silently training on fewer points than requested.
    """
    if n_interior <= 0 or n_boundary <= 0 or n_obstacle_surface <= 0:
        raise ValueError("n_interior, n_boundary, and n_obstacle_surface must be strictly positive.")
    if domain.kind != "channel":
        raise ValueError(
            "sample_collocation_points_with_obstacle currently only supports "
            f"domain.kind == 'channel'; got {domain.kind!r}."
        )
    if oversampling_factor <= 1.0:
        raise ValueError(f"oversampling_factor must be strictly greater than 1.0; got {oversampling_factor!r}.")
    if axis_clearance_fraction is not None and not (0.0 <= axis_clearance_fraction < 1.0):
        raise ValueError(
            "axis_clearance_fraction must be None, or lie in [0.0, 1.0); "
            f"got {axis_clearance_fraction!r}."
        )
    if obstacle.radius >= domain.radius:
        raise ValueError(
            f"obstacle.radius ({obstacle.radius!r}) must be strictly less than domain.radius "
            f"({domain.radius!r}); a sphere reaching the channel wall is not representable here."
        )
    obstacle_z_min = obstacle.axial_position - obstacle.radius
    obstacle_z_max = obstacle.axial_position + obstacle.radius
    if not (0.0 < obstacle_z_min and obstacle_z_max < domain.length):
        raise ValueError(
            "obstacle must fit strictly inside the channel's axial extent (0, domain.length); "
            f"got axial span [{obstacle_z_min!r}, {obstacle_z_max!r}] against domain.length="
            f"{domain.length!r}."
        )
 
    resolved_axis_clearance_fraction = (
        _AXIS_CLEARANCE_FRACTION if axis_clearance_fraction is None else axis_clearance_fraction
    )
    resolved_device = resolve_device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(random_seed)
 
    # Interior points, by rejection sampling: draw candidates from the same
    # volume-correct (r dr) distribution `sample_collocation_points` uses,
    # then discard any candidate whose (z, r) falls inside the obstacle
    # disk `(z - z_p)^2 + r^2 < a^2` before keeping the first n_interior
    # survivors.
    n_candidates = math.ceil(oversampling_factor * n_interior)
    r_min = domain.radius * resolved_axis_clearance_fraction
    xi = torch.rand(n_candidates, 1, generator=generator, device=resolved_device)
    candidate_r = torch.sqrt(r_min**2 + (domain.radius**2 - r_min**2) * xi)
    candidate_z = domain.length * torch.rand(n_candidates, 1, generator=generator, device=resolved_device)
 
    outside_obstacle = (candidate_z - obstacle.axial_position) ** 2 + candidate_r**2 >= obstacle.radius**2
    surviving_r = candidate_r[outside_obstacle]
    surviving_z = candidate_z[outside_obstacle]
    if surviving_r.shape[0] < n_interior:
        raise RuntimeError(
            f"sample_collocation_points_with_obstacle drew {n_candidates} candidate interior "
            f"points (oversampling_factor={oversampling_factor!r}) but only "
            f"{surviving_r.shape[0]} survived obstacle rejection, fewer than the requested "
            f"n_interior={n_interior!r}. Raise oversampling_factor, or request fewer interior "
            "points, rather than silently training on an under-sized batch."
        )
    interior = torch.stack([surviving_r[:n_interior], surviving_z[:n_interior]], dim=1)
 
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
 
    # Obstacle surface: uniform in the meridian polar angle theta in [0,
    # pi], matching the parametrization
    # `physics.hydrodynamic_drag.surface_traction_force` integrates over
    # (r, z) = (a sin(theta), z_p + a cos(theta)). Randomly re-drawn (like
    # the interior and vessel-boundary points) rather than a fixed grid,
    # consistent with this module's per-epoch resampling philosophy for
    # the Adam training phase.
    theta = math.pi * torch.rand(n_obstacle_surface, 1, generator=generator, device=resolved_device)
    obstacle_surface = torch.cat(
        [
            obstacle.radius * torch.sin(theta),
            obstacle.axial_position + obstacle.radius * torch.cos(theta),
        ],
        dim=1,
    )
 
    return CollocationPoints(interior=interior, boundary=boundary, initial=None, obstacle_surface=obstacle_surface)
