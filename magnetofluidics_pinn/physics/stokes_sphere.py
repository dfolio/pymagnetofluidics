r"""Closed-form Stokes-flow references for a sphere.

Provides the exact unbounded Stokes field of a translating sphere
[@happel1983low; @kim2005microhydrodynamics] and the classical tube-wall
resistance factor [@haberman1958motion]. These are *independent* references
used to test the traction integral and the tube resistance, in the
dimensionless convention $\mu^* = 1$ used across the package.
"""

from __future__ import annotations

import math

import torch

# Relative tolerance used to decide that a point lies strictly inside the sphere.
_INTERIOR_TOLERANCE = 1.0e-6


def _require_finite(value: float, name: str) -> None:
    """Raise `ValueError` if `value` is not a finite number.

    Args:
    - `value`: Number to check.
    - `name`: Name used in the error message.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")


def stokes_sphere_field(
    coordinates: torch.Tensor, center_z: float, radius: float, velocity: float
) -> torch.Tensor:
    r"""Evaluate the unbounded Stokes field of a sphere translating along the axis.

    With $\boldsymbol\zeta = \mathbf x - \mathbf x_p$, $\rho = \norm{\boldsymbol\zeta}$
    and $\mu = 1$:

    $$ \mathbf u = \frac{3a}{4}\left(\frac{\mathbf I}{\rho} + \frac{\boldsymbol\zeta\boldsymbol\zeta}{\rho^3}\right)\cdot\mathbf U
    + \frac{a^3}{4}\left(\frac{\mathbf I}{\rho^3} - \frac{3\boldsymbol\zeta\boldsymbol\zeta}{\rho^5}\right)\cdot\mathbf U,
    \qquad p = \frac{3a}{2}\,\frac{\mathbf U\cdot\boldsymbol\zeta}{\rho^3}. $$

    The force the fluid exerts on the sphere is $F_z = -6\pi a U$.

    Args:
    - `coordinates`: Tensor `(n_points, 2)` of `(r, z)` points, all outside or on the sphere.
    - `center_z`: Axial position of the sphere center.
    - `radius`: Sphere radius $a > 0$.
    - `velocity`: Axial translation velocity $U$.

    Returns:
    - Tensor `(n_points, 3)` with columns $(u_r, u_z, p)$.

    Raises:
    - `ValueError`: If `coordinates` is not `(n_points, 2)`, if a scalar is not
      finite, if `radius <= 0`, or if a point lies strictly inside the sphere.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (n_points, 2); got {tuple(coordinates.shape)}.")
    for name, value in (("center_z", center_z), ("radius", radius), ("velocity", velocity)):
        _require_finite(value, name)
    if radius <= 0.0:
        raise ValueError(f"radius must be strictly positive; got {radius!r}.")

    r = coordinates[:, 0:1]
    zeta = coordinates[:, 1:2] - center_z
    rho_sq = r.square() + zeta.square()
    if torch.any(rho_sq < radius**2 * (1.0 - _INTERIOR_TOLERANCE)):
        raise ValueError("coordinates contain points strictly inside the sphere.")
    
    rho = torch.sqrt(rho_sq)
    rho3 = rho * rho_sq
    rho5 = rho3 * rho_sq
    a = radius
    u_z = velocity * (
            0.75 * a * (1.0 / rho + zeta.square() / rho3)
            + 0.25 * a ** 3 * (1.0 / rho3 - 3.0 * zeta.square() / rho5)
    )
    u_r = velocity * (0.75 * a * r * zeta / rho3 - 0.75 * a ** 3 * r * zeta / rho5)
    pressure = 1.5 * a * velocity * zeta / rho3
    return torch.cat([u_r, u_z, pressure], dim=1)


def stokes_drag_reference(radius: float, velocity: float) -> float:
    r"""Return the unbounded Stokes force on a translating sphere, $F_z = -6\pi a U$.

    Args:
    - `radius`: Sphere radius $a > 0$.
    - `velocity`: Axial velocity $U$.

    Returns:
    - The axial force the fluid exerts on the sphere, in $\mu^* = 1$ units.

    Raises:
    - `ValueError`: If `radius <= 0` or an argument is not finite.
    """
    _require_finite(radius, "radius")
    _require_finite(velocity, "velocity")
    if radius <= 0.0:
        raise ValueError(f"radius must be strictly positive; got {radius!r}.")
    return -6.0 * math.pi * radius * velocity


def haberman_sayre_wall_factor(blockage_ratio: float) -> float:
    r"""Evaluate the Haberman–Sayre wall-correction factor for a sphere on the tube axis.

    $$ K_w(\lambda) = \frac{1 - 0.75857\lambda^5}
    {1 - 2.1050\lambda + 2.0865\lambda^3 - 1.7068\lambda^5 + 0.72603\lambda^6}, $$

    with $\lambda = a/R$ [@haberman1958motion; @happel1983low]. The drag on a
    sphere moving through still fluid is $6\pi\mu a U K_w$.

    Args:
    - `blockage_ratio`: $\lambda \in [0, 1)$.

    Returns:
    - The factor $K_w \ge 1$.

    Raises:
    - `ValueError`: If `blockage_ratio` is outside $[0, 1)$ or the series denominator is not positive.
    """
    if not (0.0 <= blockage_ratio < 1.0):
        raise ValueError(f"blockage_ratio must lie in [0, 1); got {blockage_ratio!r}.")
    lam = blockage_ratio
    denominator = 1.0 - 2.1050 * lam + 2.0865 * lam**3 - 1.7068 * lam**5 + 0.72603 * lam**6
    if denominator <= 0.0:
        raise ValueError(f"Wall-factor series is singular at blockage ratio {blockage_ratio!r}.")
    return (1.0 - 0.75857 * lam**5) / denominator


def tube_resistance_reference(
    particle_radius: float, channel_radius: float, peak_velocity: float
) -> tuple[float, float]:
    r"""Reference resistance pair $(F_A, F_B)$ of an on-axis sphere in a Poiseuille tube.

    By Stokes linearity the axial force is $F = F_A + U_p F_B$, with

    $$ F_A = 6\pi a K_w\left(1 - \tfrac{2}{3}\lambda^2\right)U_{\max},\qquad
    F_B = -6\pi a K_w, $$

    where the $(1-\tfrac23\lambda^2)$ factor is the Faxén curvature correction. This is a
    leading-order reference, adequate to a few percent for $\lambda \lesssim 0.5$.

    Args:
    - `particle_radius`: Sphere radius $a$.
    - `channel_radius`: Tube radius $R > a$.
    - `peak_velocity`: Centerline velocity $U_{\max}$ of the undisturbed flow.

    Returns:
    - The tuple `(ambient_force, translation_force)`.

    Raises:
    - `ValueError`: If the radii are not $0 < a < R$ or an argument is not finite.
    """
    for name, value in (("particle_radius", particle_radius), ("channel_radius", channel_radius),
                        ("peak_velocity", peak_velocity)):
        _require_finite(value, name)
    if not (0.0 < particle_radius < channel_radius):
        raise ValueError("Radii must satisfy 0 < particle_radius < channel_radius.")
    lam = particle_radius / channel_radius
    k_w = haberman_sayre_wall_factor(lam)
    base = 6.0 * math.pi * particle_radius * k_w
    return base * (1.0 - 2.0 * lam**2 / 3.0) * peak_velocity, -base

def stokes_reduced_stream_function(
    s: torch.Tensor, z: torch.Tensor, center_z: float, radius: float, velocity: torch.Tensor | float,
    rho_floor_fraction: float = 0.5,
) -> torch.Tensor:
    r"""Evaluate $F_S = \psi_S / s$ of the unbounded Stokes sphere, with $s = r^2$.

    $$ F_S = -\frac{U}{4}\left(\frac{3a}{\rho} - \frac{a^3}{\rho^3}\right),\qquad
    u_r = r\,\partial_z F_S,\quad u_z = -2\,(F_S + s\,\partial_s F_S), $$

    which reproduces [`stokes_sphere_field`][magnetofluidics_pinn.physics.stokes_sphere.stokes_sphere_field].

    Args:
    - `s`: Tensor `(n, 1)` with $s = r^2$.
    - `z`: Tensor `(n, 1)` with the axial coordinate.
    - `center_z`: Sphere center.
    - `radius`: Sphere radius $a > 0$.
    - `velocity`: Sphere velocity $U$ (scalar or broadcastable tensor).
    - `rho_floor_fraction`: Clamp $\rho \ge$ this fraction of $a$, to avoid the center singularity
      at points inside the sphere (these points are never used in the fluid).

    Returns:
    - Tensor `(n, 1)` with $F_S$.
    """
    rho_sq = torch.clamp(s + (z - center_z).square(), min=(rho_floor_fraction * radius) ** 2)
    rho = torch.sqrt(rho_sq)
    return -0.25 * velocity * (3.0 * radius / rho - radius**3 / (rho * rho_sq))


def stokes_sphere_pressure(
    s: torch.Tensor, z: torch.Tensor, center_z: float, radius: float, velocity: torch.Tensor | float,
    rho_floor_fraction: float = 0.5,
) -> torch.Tensor:
    r"""Evaluate the Stokes-sphere pressure $p = \tfrac32 a U \zeta/\rho^3$ ($\mu = 1$).

    Args:
    - `s`, `z`, `center_z`, `radius`, `velocity`, `rho_floor_fraction`: As in
      [`stokes_reduced_stream_function`][magnetofluidics_pinn.physics.stokes_sphere.stokes_reduced_stream_function].

    Returns:
    - Tensor `(n, 1)` with $p$.
    """
    zeta = z - center_z
    rho_sq = torch.clamp(s + zeta.square(), min=(rho_floor_fraction * radius) ** 2)
    return 1.5 * radius * velocity * zeta / (rho_sq * torch.sqrt(rho_sq))


def oseen_drag_correction(reynolds_radius: float) -> float:
    r"""Estimate the Oseen inertial correction to the Stokes drag of a sphere.

    $$ \frac{F}{F_\text{Stokes}} \approx 1 + \frac{3}{16}\,Re_d = 1 + \frac{3}{8}\,Re_a,
    \qquad Re_a = \frac{\rho\,\abs{U_\text{rel}}\,a}{\mu}, $$

    an unbounded-fluid estimate valid for $Re \lesssim 1$ [@happel1983low]. In a tube it is
    only an order-of-magnitude indicator of how far the creeping-flow model can be trusted.

    Args:
    - `reynolds_radius`: Radius-based particle Reynolds number $Re_a \ge 0$.

    Returns:
    - The factor $1 + \tfrac38 Re_a$.

    Raises:
    - `ValueError`: If `reynolds_radius` is negative or not finite.
    """
    if not (math.isfinite(reynolds_radius) and reynolds_radius >= 0.0):
        raise ValueError(f"reynolds_radius must be finite and non-negative; got {reynolds_radius!r}.")
    return 1.0 + 0.375 * reynolds_radius