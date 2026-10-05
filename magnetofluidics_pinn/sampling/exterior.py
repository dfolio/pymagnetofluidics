r"""Collocation sampling for the exterior (fluid-only) domain around a sphere.

Fixes two issues in `sample_collocation_points_with_particle`: the unused
`boundary_layer_*` arguments, and the volume measure $r\,\dd r$ that starves
the near-axis region where the sphere sits. Points are uniform in the
meridian $(r, z)$ half-plane, plus a shell around the sphere.
"""

from __future__ import annotations

import math

import torch

from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.types import Domain

# Candidate oversampling used by rejection sampling.
_OVERSAMPLING = 2.0


def _take(candidates: torch.Tensor, keep: torch.Tensor, n_points: int, label: str) -> torch.Tensor:
    """Return the first `n_points` accepted candidates.

    Args:
    - `candidates`: Tensor `(m, 2)`.
    - `keep`: Boolean mask `(m,)`.
    - `n_points`: Number of points requested.
    - `label`: Name used in the error message.

    Returns:
    - Tensor `(n_points, 2)`.

    Raises:
    - `RuntimeError`: If fewer than `n_points` candidates are accepted.
    """
    accepted = candidates[keep]
    if accepted.shape[0] < n_points:
        raise RuntimeError(f"Only {accepted.shape[0]} of {n_points} {label} points survived rejection.")
    return accepted[:n_points]


def sample_exterior_points(
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    n_points: int,
    random_seed: int,
    shell_fraction: float = 0.25,
    shell_factor: float = 2.5,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Draw interior points outside the sphere, uniform in area plus a refined shell.

    The shell uses $\rho^2 \sim \mathcal U(a^2, (ka)^2)$ and $\theta \sim \mathcal U(0, \pi)$,
    which is uniform in the $(r, z)$ half-plane area $\rho\,\dd\rho\,\dd\theta$.

    Args:
    - `domain`: Nondimensional channel.
    - `particle_axial_position`: Sphere center $z_p$, with $0 < z_p < L$.
    - `particle_radius`: Sphere radius, with $0 < a < R$.
    - `n_points`: Total number of points.
    - `random_seed`: Seed for a device-local generator.
    - `shell_fraction`: Fraction in $[0, 1)$ of points placed in the shell.
    - `shell_factor`: Shell outer radius in units of $a$ ($> 1$); clipped to $0.95R$.
    - `device`: Target device; auto-resolved if `None`.
    - `dtype`: Floating-point dtype.

    Returns:
    - Detached tensor `(n_points, 2)` of `(r, z)` points.

    Raises:
    - `ValueError`: If an argument is out of range.
    - `RuntimeError`: If rejection sampling cannot provide enough points.
    """
    if n_points <= 0:
        raise ValueError("n_points must be strictly positive.")
    if not (0.0 <= shell_fraction < 1.0):
        raise ValueError("shell_fraction must lie in [0, 1).")
    if shell_factor <= 1.0:
        raise ValueError("shell_factor must exceed 1.")
    if not (0.0 < particle_radius < domain.radius):
        raise ValueError("particle_radius must satisfy 0 < a < domain.radius.")
    if not (0.0 < particle_axial_position < domain.length):
        raise ValueError("particle_axial_position must lie inside the channel.")

    resolved = resolve_device(device)
    generator = torch.Generator(device=resolved).manual_seed(random_seed)
    a, z_p = particle_radius, particle_axial_position
    outer = min(shell_factor * a, 0.95 * domain.radius)
    n_shell = int(round(n_points * shell_fraction)) if outer > a else 0
    n_bulk = n_points - n_shell

    def uniform(n: int) -> torch.Tensor:
        return torch.rand(n, 1, generator=generator, device=resolved, dtype=dtype)

    n_cand = math.ceil(_OVERSAMPLING * n_bulk) + 16
    bulk = torch.cat([domain.radius * uniform(n_cand), domain.length * uniform(n_cand)], dim=1)
    outside = (bulk[:, 1] - z_p).square() + bulk[:, 0].square() >= a**2
    bulk = _take(bulk, outside, n_bulk, "bulk")
    if n_shell == 0:
        return bulk.detach()

    n_cand = math.ceil(3.0 * n_shell) + 16
    rho = torch.sqrt(a**2 + (outer**2 - a**2) * uniform(n_cand))
    theta = math.pi * uniform(n_cand)
    shell = torch.cat([rho * torch.sin(theta), z_p + rho * torch.cos(theta)], dim=1)
    inside_channel = (shell[:, 1] > 0.0) & (shell[:, 1] < domain.length)
    shell = _take(shell, inside_channel, n_shell, "shell")
    return torch.cat([bulk, shell], dim=0).detach()


def sample_inlet_outlet_points(
    domain: Domain,
    n_points: int,
    random_seed: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw soft-constraint points on the inlet and outlet cross-sections.

    Wall and axis points are not needed, because the stream-function network
    satisfies those conditions exactly.

    Args:
    - `domain`: Nondimensional channel.
    - `n_points`: Total points, split evenly between the two faces; at least 2.
    - `random_seed`: Seed for a device-local generator.
    - `device`: Target device; auto-resolved if `None`.
    - `dtype`: Floating-point dtype.

    Returns:
    - The tuple `(inlet_points, outlet_points)`, each `(n, 2)` as `(r, z)`.

    Raises:
    - `ValueError`: If `n_points < 2`.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    resolved = resolve_device(device)
    generator = torch.Generator(device=resolved).manual_seed(random_seed)
    n_inlet = n_points // 2
    n_outlet = n_points - n_inlet

    def face(n: int, z_value: float) -> torch.Tensor:
        radial = domain.radius * torch.rand(n, 1, generator=generator, device=resolved, dtype=dtype)
        return torch.cat([radial, torch.full_like(radial, z_value)], dim=1)

    return face(n_inlet, 0.0), face(n_outlet, domain.length)
