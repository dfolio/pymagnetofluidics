"""Collocation-point generation.

Sampling is expressed as a pure function of the domain, the requested point
counts, and an explicit random seed, so that a given seed always reproduces
the same set of points.
"""

from __future__ import annotations

from magnetofluidics_pinn.types import CollocationPoints, Domain


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
      instance holding the sampled tensors.

    Raises:
    - `ValueError`: If `n_interior` or `n_boundary` is not strictly positive.
    """
    if n_interior <= 0 or n_boundary <= 0:
        raise ValueError("n_interior and n_boundary must be strictly positive.")
    raise NotImplementedError("Implementation scheduled for Step 2.")
