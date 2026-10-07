r"""Analytical benchmarks and reference resistances for the two-way coupling.

`magnetofluidics_pinn.verification.benchmarks` supplies the independent
references the two-way-coupling notebook was missing: a closed-form Stokes
field that exercises `surface_traction_force` end to end, the classical tube
wall factors, and a shell-invariance diagnostic.

For an exact Stokes solution, $\nabla\cdot\boldsymbol\sigma = \mathbf{0}$ in
the fluid, so the force on any closed surface enclosing the particle is the
same. A trained field violates this by an amount that measures its local
accuracy near the particle, which is where the traction is evaluated
[@happel1983low; @kim2005microhydrodynamics].
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import torch

from magnetofluidics_pinn.config import ParticleConfig
from magnetofluidics_pinn.physics.hydrodynamic_drag import (
    brenner_poiseuille_wall_factor,
    surface_traction_force,
)
from magnetofluidics_pinn.types import ParticleState


def stokes_sphere_field(
        coordinates: torch.Tensor,
        particle_radius: float,
        axial_center: float,
        translation_velocity: float = 1.0,
) -> torch.Tensor:
    r"""Evaluate the unbounded Stokes field of a sphere translating along the axis.

    With $\rho$ the distance to the sphere centre, $s = a/\rho$,
    $\cos\theta = (z - z_c)/\rho$, $\sin\theta = r/\rho$ and $\mu = 1$:

    $$ u_r = \tfrac{3}{4} U s \sin\theta\cos\theta\,(1 - s^2), \qquad
       u_z = U\left[\tfrac{3}{4} s (1 + \cos^2\theta)
             + \tfrac{1}{4} s^3 (1 - 3\cos^2\theta)\right], \qquad
       p = \tfrac{3}{2} U \frac{s \cos\theta}{\rho}. $$

    The fluid is at rest at infinity. The field satisfies
    $\mathbf{u} = (0, U)$ on $\rho = a$, and the force the fluid exerts on
    the sphere is $F_z = -6\pi a U$ [@happel1983low].

    Args:
    - `coordinates`: Tensor of shape `(n_points, 2)` with `(r, z)` rows, all
      outside the sphere centre. Differentiable if it requires gradients.
    - `particle_radius`: Sphere radius $a > 0$.
    - `axial_center`: Axial position $z_c$ of the sphere centre.
    - `translation_velocity`: Sphere velocity $U$.

    Returns:
    - Tensor of shape `(n_points, 3)` with columns $(u_r, u_z, p)$.

    Raises:
    - `ValueError`: If `coordinates` is not `(n_points, 2)`, or if a scalar
      argument is not finite, or `particle_radius` is not strictly positive.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (n_points, 2); got {tuple(coordinates.shape)}.")
    if not (math.isfinite(particle_radius) and particle_radius > 0.0):
        raise ValueError(f"particle_radius must be finite and strictly positive; got {particle_radius!r}.")
    if not (math.isfinite(axial_center) and math.isfinite(translation_velocity)):
        raise ValueError("axial_center and translation_velocity must be finite.")

    radial = coordinates[:, 0:1]
    offset = coordinates[:, 1:2] - axial_center
    distance = torch.sqrt(radial.square() + offset.square())
    ratio = particle_radius / distance
    cosine, sine = offset / distance, radial / distance
    speed = translation_velocity

    u_r = 0.75 * speed * ratio * sine * cosine * (1.0 - ratio.square())
    u_z = speed * (
            0.75 * ratio * (1.0 + cosine.square())
            + 0.25 * ratio.pow(3) * (1.0 - 3.0 * cosine.square())
    )
    pressure = 1.5 * speed * ratio * cosine / distance
    return torch.cat([u_r, u_z, pressure], dim=1)


def haberman_sayre_factor(blockage_ratio: float) -> float:
    r"""Wall factor $K_H(\lambda)$ for a sphere translating on the axis of a tube.

    $$ K_H = \frac{1 - 0.75857\lambda^5}
       {1 - 2.1050\lambda + 2.0865\lambda^3 - 1.7068\lambda^5 + 0.72603\lambda^6},
       \qquad F = -6\pi\mu a U K_H, $$

    for fluid at rest far from the sphere [@haberman1958motion].

    Args:
    - `blockage_ratio`: $\lambda = a/R \in [0, 1)$.

    Returns:
    - The dimensionless factor $K_H \ge 1$.

    Raises:
    - `ValueError`: If `blockage_ratio` is outside $[0, 1)$ or the
      denominator is not positive.
    """
    if not (0.0 <= blockage_ratio < 1.0):
        raise ValueError(f"blockage_ratio must lie in [0, 1); got {blockage_ratio!r}.")
    lam = blockage_ratio
    denominator = 1.0 - 2.1050 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5 + 0.72603 * lam ** 6
    if denominator <= 0.0:
        raise ValueError(f"Wall-factor series is singular at blockage_ratio={blockage_ratio!r}.")
    return (1.0 - 0.75857 * lam ** 5) / denominator


def reference_resistances(
        blockage_ratio: float, particle_radius: float, peak_velocity: float = 1.0,
) -> dict[str, float]:
    r"""Reference forces of the two resistance problems, in the package's units ($\mu = 1$).

    Stokes flow is linear, so the force on a sphere at axial velocity $U$ in
    an ambient Poiseuille flow of peak velocity $u_c$ is
    $F(U) = F_0 + U F_1$, with

    $$ F_0 = 6\pi a u_c K_P(\lambda), \qquad F_1 = -6\pi a K_H(\lambda), \qquad
       U_\text{free} = -F_0/F_1 = u_c K_P/K_H. $$

    These are series-based references (see
    [`brenner_poiseuille_wall_factor`][magnetofluidics_pinn.physics.hydrodynamic_drag.brenner_poiseuille_wall_factor]
    and [`haberman_sayre_factor`][magnetofluidics_pinn.verification.benchmarks.haberman_sayre_factor]),
    adequate to a few percent at moderate $\lambda$; they are not exact
    [@richou2003correction; @haberman1958motion].

    Args:
    - `blockage_ratio`: $\lambda = a/R$.
    - `particle_radius`: Dimensionless sphere radius $a > 0$.
    - `peak_velocity`: Ambient centreline velocity $u_c > 0$.

    Returns:
    - A dict with `"wall_factor_poiseuille"`, `"wall_factor_towed"`,
      `"force_fixed"`, `"force_unit_translation"` and `"force_free_velocity"`.

    Raises:
    - `ValueError`: If `particle_radius` or `peak_velocity` is not finite and
      strictly positive, or from the wall-factor functions.
    """
    if not (math.isfinite(particle_radius) and particle_radius > 0.0):
        raise ValueError(f"particle_radius must be finite and strictly positive; got {particle_radius!r}.")
    if not (math.isfinite(peak_velocity) and peak_velocity > 0.0):
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    k_poiseuille = brenner_poiseuille_wall_factor(blockage_ratio)
    k_towed = haberman_sayre_factor(blockage_ratio)
    stokes_scale = 6.0 * math.pi * particle_radius
    return {
        "wall_factor_poiseuille": k_poiseuille,
        "wall_factor_towed": k_towed,
        "force_fixed": stokes_scale * peak_velocity * k_poiseuille,
        "force_unit_translation": -stokes_scale * k_towed,
        "force_free_velocity": peak_velocity * k_poiseuille / k_towed,
    }


def shell_force_profile(
        flow_network,
        particle_config: ParticleConfig,
        particle_state: ParticleState,
        shell_factors: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
        n_quadrature_points: int = 361,
) -> pd.DataFrame:
    r"""Evaluate the axial force on concentric spherical shells around the particle.

    Because $\nabla\cdot\boldsymbol\sigma = \mathbf{0}$ in the fluid, the
    force is independent of the shell radius for an exact solution. The
    spread over shells is an a-posteriori error indicator for the traction.

    Args:
    - `flow_network`: Callable mapping `(r, z)` to `(u_r, u_z, p)`.
    - `particle_config`: Particle whose `radius` $a$ defines the base shell.
    - `particle_state`: Supplies the axial centre.
    - `shell_factors`: Shell radii as multiples of $a$; each must be $\ge 1$
      and the shell must stay inside the channel (caller's responsibility).
    - `n_quadrature_points`: Polar-angle nodes per shell.

    Returns:
    - A `pandas.DataFrame` with columns `"shell_factor"`, `"shell_radius"`
      and `"force"`.

    Raises:
    - `ValueError`: If `shell_factors` is empty or contains a value below 1.
    """
    if not shell_factors:
        raise ValueError("shell_factors must contain at least one entry.")
    if any(factor < 1.0 for factor in shell_factors):
        raise ValueError(f"shell_factors must all be >= 1 (shells must lie in the fluid); got {shell_factors!r}.")
    records = [
        {
            "shell_factor": factor,
            "shell_radius": factor * particle_config.radius,
            "force": surface_traction_force(
                flow_network,
                dataclasses.replace(particle_config, radius=factor * particle_config.radius),
                particle_state,
                n_quadrature_points,
            ).item(),
        }
        for factor in shell_factors
    ]
    return pd.DataFrame.from_records(records)


def shell_force_spread(profile: pd.DataFrame) -> float:
    r"""Relative spread $(F_\max - F_\min)/\abs{\bar F}$ of a shell-force profile.

    Args:
    - `profile`: Output of
      [`shell_force_profile`][magnetofluidics_pinn.verification.benchmarks.shell_force_profile].

    Returns:
    - The relative spread; `inf` if the mean force is numerically zero.

    Raises:
    - `ValueError`: If `profile` has no `"force"` column or no rows.
    """
    if "force" not in profile or profile.empty:
        raise ValueError("profile must be a non-empty DataFrame with a 'force' column.")
    forces = profile["force"].to_numpy(dtype=float)
    mean_magnitude = abs(float(forces.mean()))
    if mean_magnitude < 1.0e-12:
        return math.inf
    return float(forces.max() - forces.min()) / mean_magnitude
