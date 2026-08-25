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
# (r = 0) when drawing interior points. The axisymmetric Stokes residual
# (see `physics.fluid_residuals.stokes_residual`) involves terms
# proportional to `1 / r` and `1 / r**2`, which are singular exactly on the
# axis. A clearance that is too tight is not just a removable-singularity
# concern but a numerical-stability one: at 1e-4 * radius, `1 / r**2`
# amplifies an untrained network's (generally nonzero) `u_r` by a factor of
# order 1e8, so any collocation point that happens to land close to the
# axis can produce a residual outlier many orders of magnitude larger than
# the rest of the batch, destabilizing the mean-squared PDE loss used
# during training. 5e-2 keeps that amplification bounded (~4e2) while
# still leaving the axis itself well-sampled.
_AXIS_CLEARANCE_FRACTION = 5.0e-2


def boundary_face_sizes(n_boundary: int) -> tuple[int, int, int]:
    """Split a boundary point budget across the three faces of a channel.

    A straight channel has three boundary faces: the lateral wall
    (`r = domain.radius`), the inlet cross-section (`z = 0`), and the outlet
    cross-section (`z = domain.length`). This helper is shared by
    [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
    and by the training loop, which needs to know how
    `CollocationPoints.boundary` is laid out (wall-points-first, then inlet
    points, then outlet points) to apply the matching boundary condition to
    each subset.

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
    - `device`: Device the returned tensors are created on directly, e.g.
      `"cuda"`, `"cuda:0"`, or `"cpu"`; resolved automatically when `None`.
      # CHANGED: tensors are now created directly on the resolved device
      (auto-selecting CUDA when available, matching `build_mlp` and
      `train`), rather than unconditionally on the CPU. This avoids a
      CPU -> GPU copy every epoch when training on a CUDA device. Note
      that a CPU generator and a CUDA generator seeded identically produce
      *different* draws (they use different underlying RNG algorithms):
      reproducibility is guaranteed per device type, not across device
      types.

    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance holding the sampled tensors, all on the resolved device.
      `boundary` concatenates, in order, the wall, inlet, and outlet
      subsets sized by
      [`boundary_face_sizes`][magnetofluidics_pinn.sampling.collocation.boundary_face_sizes].

    Raises:
    - `ValueError`: If `n_interior` or `n_boundary` is not strictly positive,
      if `domain.kind` is not `"channel"` (the only geometry this sampler
      currently supports), or if `device` is not a valid device string.
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

    resolved_device = resolve_device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(random_seed)

    # Interior points: uniform in (r, z), keeping a small clearance around
    # the symmetry axis (see module docstring).
    r_min = domain.radius * _AXIS_CLEARANCE_FRACTION
    interior_r = r_min + (domain.radius - r_min) * torch.rand(
        n_interior, 1, generator=generator, device=resolved_device
    )
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

