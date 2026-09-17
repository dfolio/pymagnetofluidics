"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

import torch

from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.types import CollocationPoints, Domain

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
