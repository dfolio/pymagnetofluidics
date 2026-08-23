"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

import torch

from magnetofluidics_pinn.types import CollocationPoints, Domain

# Fraction of the channel radius kept clear around the symmetry axis
# (r = 0) when drawing interior points. The axisymmetric Stokes residual
# (see `physics.fluid_residuals.stokes_residual`) involves terms
# proportional to `1 / r` and `1 / r**2`, which are singular exactly on the
# axis; a small clearance avoids that removable singularity without
# materially affecting how well the interior is covered.
_AXIS_CLEARANCE_FRACTION = 1.0e-4


def boundary_face_sizes(n_boundary: int) -> tuple[int, int, int]:
    """Split a boundary point budget across the three faces of a channel.

    A straight channel has three boundary faces: the lateral wall
    (`r = domain.radius`), the inlet cross-section (`z = 0`), and the outlet
    cross-section (`z = domain.length`). This helper is shared by
    [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
    and by the training loop, which needs to know how `CollocationPoints.boundary`
    is laid out (wall-points-first, then inlet points, then outlet points)
    to apply the matching boundary condition to each subset.

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
) -> CollocationPoints:
    """Sample interior, boundary, and (optionally) initial points.

    Args:
    - `domain`: Vessel geometry to sample from.
    - `n_interior`: Number of interior points to sample.
    - `n_boundary`: Number of boundary points to sample.
    - `random_seed`: Seed guaranteeing reproducible sampling.
    - `n_initial`: Number of initial-time points to sample; `None` for
      steady-state problems (e.g., Stokes flow).

    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance holding the sampled tensors. `boundary` concatenates, in
      order, the wall, inlet, and outlet subsets sized by
      [`boundary_face_sizes`][magnetofluidics_pinn.sampling.collocation.boundary_face_sizes].
      Every tensor is created on the CPU; callers move them to the training
      device explicitly.

    Raises:
    - `ValueError`: If `n_interior` or `n_boundary` is not strictly positive,
      or if `domain.kind` is not `"channel"` (the only geometry this sampler
      currently supports).
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
    
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    
    # Interior points: uniform in (r, z), keeping a small clearance around
    # the symmetry axis (see module docstring).
    r_min = domain.radius * _AXIS_CLEARANCE_FRACTION
    interior_r = r_min + (domain.radius - r_min) * torch.rand(
        n_interior, 1, generator=generator
    )
    interior_z = domain.length * torch.rand(n_interior, 1, generator=generator)
    interior = torch.cat([interior_r, interior_z], dim=1)
    
    n_wall, n_inlet, n_outlet = boundary_face_sizes(n_boundary)
    
    wall_points = torch.cat(
        [
            torch.full((n_wall, 1), domain.radius),
            domain.length * torch.rand(n_wall, 1, generator=generator),
        ],
        dim=1,
    )
    inlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_inlet, 1, generator=generator),
            torch.zeros(n_inlet, 1),
        ],
        dim=1,
    )
    outlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_outlet, 1, generator=generator),
            torch.full((n_outlet, 1), domain.length),
        ],
        dim=1,
    )
    boundary = torch.cat([wall_points, inlet_points, outlet_points], dim=0)
    
    return CollocationPoints(interior=interior, boundary=boundary, initial=None)
