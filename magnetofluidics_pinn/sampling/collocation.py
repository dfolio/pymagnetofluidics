"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

import torch

from magnetofluidics_pinn.config import TrainingConfig
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
        n_interior: int | None = None,
        n_boundary: int | None = None,
        random_seed: int | None = None,
        device: str | torch.device | None = None,
        n_initial: int | None = None,
        training_config: TrainingConfig | None = None,
) -> CollocationPoints:
    """Sample interior, boundary, and (optionally) initial points.

    Accepts sampling parameters either explicitly or extracted from a
    [`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig] object.

    Args:
    - `domain`: Vessel geometry to sample from.
    - `n_interior`: Number of interior points. Extracted from `training_config`
      if `None`.
    - `n_boundary`: Number of boundary points. Extracted from `training_config`
      if `None`.
    - `random_seed`: Seed for sampling. Extracted from `training_config` if `None`.
    - `device`: Target computing device (`"cuda"`, `"cpu"`, or `torch.device`).
      Extracted from `training_config` or resolved to CUDA/CPU if `None`.
    - `n_initial`: Number of initial-time points to sample; `None` for
      steady-state problems (e.g., Stokes flow).
    - `training_config`: Optional configuration fallback.

    Returns:
    - A [`CollocationPoints`][magnetofluidics_pinn.types.CollocationPoints]
      instance with tensors allocated directly on `device`.

    Raises:
    - `ValueError`: If required point counts or seed are missing, non-positive,
      or if `domain.kind != "channel"`.
    - `NotImplementedError`: If `n_initial` is not `None`.
    """
    n_int = n_interior if n_interior is not None else (
        training_config.n_interior_points if training_config is not None else None
    )
    n_bnd = n_boundary if n_boundary is not None else (
        training_config.n_boundary_points if training_config is not None else None
    )
    seed = random_seed if random_seed is not None else (
        training_config.random_seed if training_config is not None else 42
    )
    target_device = (
        torch.device(device)
        if device is not None
        else (
            training_config.device
            if training_config is not None and training_config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    )

    if n_int is None or n_bnd is None:
        raise ValueError(
            "n_interior and n_boundary must be specified explicitly or via"
            "training_config."
        )
    if n_int <= 0 or n_bnd <= 0:
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
    generator = torch.Generator(device=target_device).manual_seed(seed)

    # Interior points: uniform in (r, z), keeping a small clearance around
    # the symmetry axis (see module docstring).
    r_min = domain.radius * _AXIS_CLEARANCE_FRACTION
    interior_r = r_min + (domain.radius - r_min) * torch.rand(
        n_int, 1, generator=generator, device=target_device
    )
    interior_z = domain.length * torch.rand(
        n_int, 1, generator=generator, device=target_device
    )
    interior = torch.cat([interior_r, interior_z], dim=1)

    n_wall, n_inlet, n_outlet = boundary_face_sizes(n_bnd)

    # CHANGED: Added device=target_device across all tensor factory calls
    wall_points = torch.cat(
        [
            torch.full((n_wall, 1), domain.radius, device=target_device),
            domain.length * torch.rand(n_wall, 1, generator=generator,
                                       device=target_device),
        ],
        dim=1,
    )
    inlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_inlet, 1, generator=generator,
                                       device=target_device),
            torch.zeros(n_inlet, 1, device=target_device),
        ],
        dim=1,
    )
    outlet_points = torch.cat(
        [
            domain.radius * torch.rand(n_outlet, 1, generator=generator,
                                       device=target_device),
            torch.full((n_outlet, 1), domain.length, device=target_device),
        ],
        dim=1,
    )
    boundary = torch.cat([wall_points, inlet_points, outlet_points], dim=0)

    return CollocationPoints(interior=interior, boundary=boundary, 
                             initial=None)
