r"""Analytic liftings of the sphere-in-tube stream function, written in $(s, z)$.

Every function of this module is a pure function of the reduced radial
coordinate $s = r^2$ and the axial coordinate $z$; nothing divides by $r$.
The stream function is $\psi = s\,F(s, z)$ with the convention

$$u_r = -\frac{1}{r}\frac{\partial \psi}{\partial z} = r\,g, \quad g = -\frac{\partial F}{\partial z}, \qquad
   u_z = \frac{1}{r}\frac{\partial \psi}{\partial r} = 2\left(F + s\,\frac{\partial F}{\partial s}\right),$$

so $u_r$ and $u_z$ are regular on the symmetry axis, and $u_z(0, z) = 2F(0, z)$ holds exactly.

**Analytic pieces.** For a sphere of radius $a$ centered at $z_p$ on the axis of a tube of radius $R$,
with ambient Poiseuille amplitude $U$ and sphere velocity $U_p$, let
$\zeta = z - z_p$, $\rho^2 = s + \zeta^2$, $W = U_p - U$ and $\kappa = U/R^2$:

- Poiseuille base: $F_P = U\left(\tfrac12 - \tfrac{s}{4R^2}\right)$, with pressure $p_P = -4U(z - L)/R^2$.
- Translating sphere (unbounded Stokes solution at the relative velocity $W$):
  $F_T = \tfrac{W}{4}\left(\tfrac{3a}{\rho} - \tfrac{a^3}{\rho^3}\right)$, with pressure $p_T = \tfrac32 a W\,\zeta/\rho^3$.
- Parabolic-curvature cancellation (Lamb modes $n = 2$ and $n = 4$), with $\xi = \zeta/\rho$:
  $F_\text{shear} = \rho^{-2}\left[C_1\rho + D_1\rho^{-1} + \tfrac32(5\xi^2 - 1)(C_3\rho^{-1} + D_3\rho^{-3})\right]$ and
  $C_1 = \tfrac{\kappa a^3}{2},\; D_1 = -\tfrac{3\kappa a^5}{10},\; C_3 = -\tfrac{7\kappa a^5}{60},\; D_3 = \tfrac{\kappa a^7}{12}$.
  Only the $C_n$ terms carry pressure: $p_\text{shear} = 2C_1\,\zeta/\rho^3 + 30\,C_3\,P_3(\xi)/\rho^4$
  with $P_3(\xi) = \tfrac12(5\xi^3 - 3\xi)$ (the $D_n$ terms are potential flows).

The sum $F_d = F_T + F_\text{shear}$ with $p_d = p_T + p_\text{shear}$ is an exact Stokes solution of the
unbounded problem. It sets the sphere velocity to $U_p$ on $\rho = a$ at every polar angle.

**Two ways to restore the wall condition** (`LiftingMode`):

- `"taper"`: multiply $F_d$ and $p_d$ by a compactly supported $C^\infty$ weight $\tau(\rho)$.
  The cross terms of the product rule are a residual the network must absorb.
- `"wall"`: keep $F_d$ whole and add a correction acting between $r_0$ and the wall,
  $\psi_c = -\chi(r)\left[\psi_d(R, z) + (r - R)\,\partial_r\psi_d(R, z)\right]$, with $\chi(R) = 1$, $\chi'(R) = 0$ and
  $\chi = 0$ for $r \le r_0$, $a < r_0 < R$. The wall velocity is then exactly zero. The residual of the lifting alone
  depends strongly on the blend width $R - r_0$ (measured for $a = 0.25R$, $L = 8R$, fixed sphere, mean of $R_r^2 + R_z^2$):
  $\approx 3.0\times10^3$ for $r_0 = a + 0.5(R - a)$, $4.0\times10^2$ for $0.25$, $1.6\times10^2$ for $0.1$ and $1.1\times10^2$ for
  $0.02$, against $8.7\times10^2$ for the taper with margin 8. Unlike the taper residual, which is local to the taper ball,
  the wall residual extends along the whole tube. The default `wall_start_fraction = 0.05` therefore starts the blend close to the sphere.
- `"unbounded"`: no wall condition (verification only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient

LiftingMode = Literal["taper", "wall", "unbounded"]

# Distances to the sphere centre are floored at this fraction of the radius, so that the analytic
# continuation inside the solid sphere stays finite (the field is never used there).
_RHO_FLOOR_FRACTION: float = 0.5
_VALID_MODES: tuple[str, ...] = ("taper", "wall", "unbounded")


@dataclass(frozen=True)
class LiftingGeometry:
    r"""Geometry and flow constants shared by every analytic lifting.

    All values are plain Python floats, so the formulas stay exact in `float64` and never trigger a
    device synchronization.

    Args:
    - `radius`: Tube radius $R > 0$.
    - `length`: Tube length $L > 0$.
    - `particle_z`: Sphere centre $z_p \in (0, L)$.
    - `particle_radius`: Sphere radius $a$, with $0 < a < R$.
    - `particle_velocity`: Sphere velocity $U_p$.
    - `flow_amplitude`: Ambient centreline velocity $U$ (zero for a sphere moving in still fluid).
    - `taper_support`: Support of the taper in $\rho^2$ units; the weight acts inside $\rho^2 < a^2 + $ support.
    - `wall_start_radius`: Radius $r_0$ where the wall correction starts, with $a < r_0 < R$.

    Raises:
    - `ValueError`: If a value is not finite or the ordering constraints are violated.
    """

    radius: float
    length: float
    particle_z: float
    particle_radius: float
    particle_velocity: float
    flow_amplitude: float
    taper_support: float
    wall_start_radius: float

    def __post_init__(self) -> None:
        values = (self.radius, self.length, self.particle_z, self.particle_radius,
                  self.particle_velocity, self.flow_amplitude, self.taper_support, self.wall_start_radius)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"LiftingGeometry values must be finite; got {values!r}.")
        if self.radius <= 0.0 or self.length <= 0.0:
            raise ValueError(f"radius and length must be strictly positive, got {self.radius!r} and {self.length!r}.")
        if not 0.0 < self.particle_radius < self.radius:
            raise ValueError(f"particle_radius must satisfy 0 < a < radius, got {self.particle_radius!r}.")
        if not 0.0 < self.particle_z < self.length:
            raise ValueError(f"particle_z must lie inside the channel, got {self.particle_z!r}.")
        if self.taper_support <= 0.0:
            raise ValueError(f"taper_support must be strictly positive, got {self.taper_support!r}.")
        if not self.particle_radius < self.wall_start_radius < self.radius:
            raise ValueError(f"wall_start_radius must satisfy a < r_0 < radius, got {self.wall_start_radius!r}.")

    @property
    def curvature(self) -> float:
        r"""Return the Poiseuille curvature $\kappa = U/R^2$."""
        return self.flow_amplitude / self.radius**2

    @property
    def relative_velocity(self) -> float:
        r"""Return the sphere velocity relative to the ambient axis velocity, $W = U_p - U$."""
        return self.particle_velocity - self.flow_amplitude

    @property
    def shear_coefficients(self) -> tuple[float, float, float, float]:
        r"""Return the Lamb coefficients $(C_1, D_1, C_3, D_3)$ cancelling the parabolic curvature on the sphere."""
        kappa, a = self.curvature, self.particle_radius
        return (0.5 * kappa * a**3, -0.3 * kappa * a**5, -7.0 * kappa * a**5 / 60.0, kappa * a**7 / 12.0)


def build_lifting_geometry(
    radius: float,
    length: float,
    particle_z: float,
    particle_radius: float,
    particle_velocity: float,
    flow_amplitude: float,
    taper_margin: float,
    wall_start_fraction: float = 0.05,
) -> LiftingGeometry:
    r"""Build a `LiftingGeometry`, deriving the taper support and the wall-correction start.

    The taper support is $\min(m\,a^2,\ 0.9\,c)$ in $\rho^2$ units, where $m$ is `taper_margin` and
    $c = \min(R^2 - a^2,\ z_p^2 - a^2,\ (L - z_p)^2 - a^2)$ is the clearance of the sphere. The wall correction starts at
    $r_0 = a + f\,(R - a)$ with $f$ = `wall_start_fraction`.

    Args:
    - `radius`, `length`, `particle_z`, `particle_radius`, `particle_velocity`, `flow_amplitude`: See `LiftingGeometry`.
    - `taper_margin`: Taper support as a multiple of $a^2$; strictly positive.
    - `wall_start_fraction`: Fraction $f \in (0, 1)$ of the gap $R - a$ left between the sphere and the start of the wall blend.

    Returns:
    - The populated `LiftingGeometry`.

    Raises:
    - `ValueError`: If the sphere does not fit strictly inside the channel, `taper_margin <= 0` or `wall_start_fraction` is outside $(0, 1)$.
    """
    if taper_margin <= 0.0:
        raise ValueError(f"taper_margin must be strictly positive; got {taper_margin!r}.")
    if not 0.0 < wall_start_fraction < 1.0:
        raise ValueError(f"wall_start_fraction must lie in (0, 1); got {wall_start_fraction!r}.")
    if particle_radius <= 0.0:
        raise ValueError(f"particle_radius must be strictly positive; got {particle_radius!r}.")
    if radius <= 0.0:
        raise ValueError(f"domain.radius must be strictly positive; got {radius!r}.")
    if particle_radius >= radius:
        raise ValueError(
            f"particle_radius ({particle_radius!r}) must be strictly smaller than domain.radius ({radius!r})."
        )
    clearance = min(
        radius**2 - particle_radius**2,
        particle_z**2 - particle_radius**2,
        (length - particle_z) ** 2 - particle_radius**2,
    )
    if clearance <= 0.0:
        raise ValueError("The sphere must lie strictly inside the channel, clear of both ends.")
    return LiftingGeometry(
        radius=float(radius), length=float(length), particle_z=float(particle_z),
        particle_radius=float(particle_radius), particle_velocity=float(particle_velocity),
        flow_amplitude=float(flow_amplitude),
        taper_support=float(min(taper_margin * particle_radius**2, 0.9 * clearance)),
        wall_start_radius=float(particle_radius + wall_start_fraction * (radius - particle_radius)),
    )


def smooth_unit_step(x: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the $C^\infty$ unit step $\varphi(x) = f(x)/(f(x) + f(1 - x))$ with $f(t) = e^{-1/t}$ for $t > 0$.

    $\varphi \equiv 0$ for $x \le 0$ and $\varphi \equiv 1$ for $x \ge 1$, and every derivative vanishes at both junctions.

    Args:
    - `x`: Tensor of any shape.

    Returns:
    - Tensor of the same shape with values in $[0, 1]$.
    """
    rising = torch.where(x > 0.0, torch.exp(-1.0 / x.clamp(min=1.0e-12)), torch.zeros_like(x))
    falling = torch.where(x < 1.0, torch.exp(-1.0 / (1.0 - x).clamp(min=1.0e-12)), torch.zeros_like(x))
    return rising / (rising + falling)


def squared_distance(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $\rho^2 = s + (z - z_p)^2$, floored inside the sphere.

    Args:
    - `s`: Tensor `(n, 1)` with $s = r^2$.
    - `z`: Tensor `(n, 1)` with the axial coordinate.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $\max(\rho^2, (f a)^2)$, where $f$ is the module-level floor fraction.
    """
    floor = (_RHO_FLOOR_FRACTION * geometry.particle_radius) ** 2
    return (s + (z - geometry.particle_z).square()).clamp(min=floor)


def poiseuille_reduced(s: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return the reduced Poiseuille stream function $F_P = U(\tfrac12 - s/(4R^2))$.

    Args:
    - `s`: Tensor `(n, 1)` with $s = r^2$.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $F_P$; $u_z = 2(F_P + sF_{P,s}) = U(1 - s/R^2)$.
    """
    return geometry.flow_amplitude * (0.5 - 0.25 * s / geometry.radius**2)


def poiseuille_pressure(z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return the Poiseuille pressure $p_P = -4U(z - L)/R^2$, anchored at $p_P(L) = 0$.

    Args:
    - `z`: Tensor `(n, 1)` with the axial coordinate.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $p_P$.
    """
    return -4.0 * geometry.flow_amplitude * (z - geometry.length) / geometry.radius**2


def translating_sphere_reduced(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $F_T = \tfrac{W}{4}\left(3a/\rho - a^3/\rho^3\right)$, the unbounded translating-sphere solution.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $F_T$.
    """
    a = geometry.particle_radius
    rho = torch.sqrt(squared_distance(s, z, geometry))
    return 0.25 * geometry.relative_velocity * (3.0 * a / rho - a**3 / rho**3)


def translating_sphere_pressure(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return the Stokes-sphere pressure $p_T = \tfrac32 a W\,\zeta/\rho^3$.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $p_T$.
    """
    rho = torch.sqrt(squared_distance(s, z, geometry))
    return 1.5 * geometry.particle_radius * geometry.relative_velocity * (z - geometry.particle_z) / rho**3


def quadratic_shear_reduced(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $F_\text{shear}$, the Lamb $n = 2$ and $n = 4$ modes cancelling the parabolic curvature on the sphere.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $F_\text{shear} = \psi_\text{shear}/s$.
    """
    c1, d1, c3, d3 = geometry.shear_coefficients
    rho2 = squared_distance(s, z, geometry)
    rho = torch.sqrt(rho2)
    xi_sq = (z - geometry.particle_z).square() / rho2
    mode_one = c1 * rho + d1 / rho
    mode_three = 1.5 * (5.0 * xi_sq - 1.0) * (c3 / rho + d3 / rho**3)
    return (mode_one + mode_three) / rho2


def quadratic_shear_pressure(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $p_\text{shear} = 2C_1\,\zeta/\rho^3 + 30\,C_3\,P_3(\xi)/\rho^4$.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $p_\text{shear}$ (the potential-flow companions $D_n$ carry no pressure).
    """
    c1, _, c3, _ = geometry.shear_coefficients
    rho = torch.sqrt(squared_distance(s, z, geometry))
    zeta = z - geometry.particle_z
    xi = zeta / rho
    legendre_3 = 0.5 * xi * (5.0 * xi.square() - 3.0)
    return 2.0 * c1 * zeta / rho**3 + 30.0 * c3 * legendre_3 / rho**4


def disturbance_reduced(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $F_d = F_T + F_\text{shear}$, the exact unbounded disturbance of the sphere.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $F_d$.
    """
    return translating_sphere_reduced(s, z, geometry) + quadratic_shear_reduced(s, z, geometry)


def disturbance_pressure(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $p_d = p_T + p_\text{shear}$, the pressure paired with `disturbance_reduced`.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with $p_d$.
    """
    return translating_sphere_pressure(s, z, geometry) + quadratic_shear_pressure(s, z, geometry)


def taper_weight(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return the compactly supported weight $\tau = 1 - \varphi\big((\rho^2 - a^2)/\text{support}\big)$.

    $\tau = 1$ inside and on the sphere with every derivative equal to zero there, and $\tau = 0$ for
    $\rho^2 \ge a^2 + $ support.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)` with values in $[0, 1]$.
    """
    argument = (squared_distance(s, z, geometry) - geometry.particle_radius**2) / geometry.taper_support
    return 1.0 - smooth_unit_step(argument)


def wall_correction_reduced(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry) -> torch.Tensor:
    r"""Return $F_c = \psi_c/s$, the correction supported near the wall.

    $$ \psi_c = -\chi(r)\left[\psi_d(R, z) + (r - R)\,\partial_r\psi_d(R, z)\right], \qquad
       \chi(r) = \varphi\!\left(\frac{r - r_0}{R - r_0}\right), $$

    with $\psi_d(R, z) = R^2 F_d(R^2, z)$ and $\partial_r\psi_d(R, z) = 2R\,(F_d + sF_{d,s})\big|_{s = R^2}$.
    The $s$-derivative is taken by automatic differentiation at the wall, which keeps the dependence on `z` in the graph.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.

    Returns:
    - Tensor `(n, 1)`; identically zero for $r \le r_0$.
    """
    radius = geometry.radius
    with torch.enable_grad():
        wall_s = torch.full_like(z, radius**2).requires_grad_(True)
        wall_f = disturbance_reduced(wall_s, z, geometry)
        (wall_df_ds,) = torch.autograd.grad(
            wall_f, wall_s, grad_outputs=torch.ones_like(wall_f), create_graph=True
        )
    psi_wall = radius**2 * wall_f
    dpsi_dr_wall = 2.0 * radius * (wall_f + radius**2 * wall_df_ds)
    # Clamping s at r_0^2 makes chi exactly zero (and flat) in the interior, so s is never zero in the division.
    s_safe = s.clamp(min=geometry.wall_start_radius**2)
    r = torch.sqrt(s_safe)
    chi = smooth_unit_step((r - geometry.wall_start_radius) / (radius - geometry.wall_start_radius))
    return -chi * (psi_wall + (r - radius) * dpsi_dr_wall) / s_safe


def reduced_lifting(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry, mode: LiftingMode) -> torch.Tensor:
    r"""Return the analytic reduced stream function $F_\text{lift}$ of the chosen lifting mode.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.
    - `mode`: `"taper"` ($F_P + \tau F_d$), `"wall"` ($F_P + F_d + F_c$) or `"unbounded"` ($F_P + F_d$).

    Returns:
    - Tensor `(n, 1)` with $F_\text{lift}$.

    Raises:
    - `ValueError`: If `mode` is unknown.
    """
    _check_mode(mode)
    base = poiseuille_reduced(s, geometry)
    disturbance = disturbance_reduced(s, z, geometry)
    if mode == "taper":
        return base + taper_weight(s, z, geometry) * disturbance
    if mode == "wall":
        return base + disturbance + wall_correction_reduced(s, z, geometry)
    return base + disturbance


def lifting_pressure(s: torch.Tensor, z: torch.Tensor, geometry: LiftingGeometry, mode: LiftingMode) -> torch.Tensor:
    r"""Return the analytic pressure paired with `reduced_lifting` (ungauged).

    The pressure follows the velocity: `"taper"` uses $p_P + \tau p_d$, the other modes use $p_P + p_d$.

    Args:
    - `s`, `z`: Tensors `(n, 1)`.
    - `geometry`: Geometry constants.
    - `mode`: See `reduced_lifting`.

    Returns:
    - Tensor `(n, 1)` with $p_\text{lift}$.

    Raises:
    - `ValueError`: If `mode` is unknown.
    """
    _check_mode(mode)
    disturbance = disturbance_pressure(s, z, geometry)
    if mode == "taper":
        disturbance = taper_weight(s, z, geometry) * disturbance
    return poiseuille_pressure(z, geometry) + disturbance


def fields_from_reduced_stream(reduced: torch.Tensor, sz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Differentiate a reduced stream function and return $(g, u_z)$ with $u_r = r\,g$.

    Args:
    - `reduced`: Tensor `(n, 1)`, $F$ evaluated from slices of `sz`.
    - `sz`: Tensor `(n, 2)` holding $(s, z)$ and requiring gradients; `reduced` must depend on it.

    Returns:
    - The pair `(g, u_z)` of tensors `(n, 1)`, with $g = -F_z$ and $u_z = 2(F + sF_s)$.
      The graph is kept (`create_graph=True`) so that the Stokes residual can differentiate further.
    """
    gradient = scalar_field_gradient(reduced, sz)
    s = sz[:, 0:1]
    return -gradient[:, 1:2], 2.0 * (reduced + s * gradient[:, 0:1])


def _check_mode(mode: str) -> None:
    """Raise `ValueError` if `mode` is not a supported lifting mode."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES!r}; got {mode!r}.")
