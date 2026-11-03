r"""Force on a fixed sphere from the translating-sphere solution, by the Lorentz reciprocal theorem.

Let $(\hat{\mathbf u}, \hat{\boldsymbol\sigma})$ be the Stokes flow of the sphere translating with unit axial velocity in
fluid at rest, in the same tube, and $\hat{\mathbf t} = \hat{\boldsymbol\sigma}\cdot\mathbf n$ its traction on the sphere surface
$S$ (normal $\mathbf n$ pointing into the fluid). Let $\mathbf u_\infty$ be the ambient flow, a Stokes solution that satisfies
the wall condition and has $\nabla\cdot\boldsymbol\sigma_\infty = \mathbf 0$ everywhere, including inside the sphere. For a tube long
enough for the disturbances to decay, the force on the sphere held fixed in $\mathbf u_\infty$ is

$$ F_A = -\oint_S \mathbf u_\infty \cdot \hat{\mathbf t}\,\mathrm dA
       = -2\pi a^2 \int_{-1}^{1} u_{\infty,z}\big(a\sqrt{1 - x^2}\big)\,\hat t_z(x)\,\mathrm dx, \qquad x = \cos\theta, $$

since $\mathbf u_\infty$ is purely axial. In an unbounded fluid $\hat t_z = -3/(2a)$ is uniform and the formula returns
$F_A = 6\pi a\,\langle u_{\infty,z}\rangle_S$, which is Faxen's first law [@happel1983low; @kim2005microhydrodynamics]. In the
tube the result is exact up to the accuracy of the translation field, and it needs no second training.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from magnetofluidics_pinn.physics.quadrature import sphere_quadrature
from magnetofluidics_pinn.verification.force_checks import FlowField, _resolve_placement, axial_traction

AmbientProfile = Callable[[torch.Tensor], torch.Tensor]


def poiseuille_axial_profile(peak_velocity: float, channel_radius: float) -> AmbientProfile:
    r"""Build the ambient profile $u_{\infty,z}(r) = U\,(1 - r^2/R^2)$.

    Args:
    - `peak_velocity`: Centreline velocity $U$, finite.
    - `channel_radius`: Tube radius $R$, finite and strictly positive.

    Returns:
    - A callable mapping a tensor of radial positions to the axial ambient velocity.

    Raises:
    - `ValueError`: If an argument is invalid.
    """
    if not math.isfinite(peak_velocity):
        raise ValueError(f"peak_velocity must be finite; got {peak_velocity!r}.")
    if not (math.isfinite(channel_radius) and channel_radius > 0.0):
        raise ValueError(f"channel_radius must be finite and strictly positive; got {channel_radius!r}.")
    return lambda r: peak_velocity * (1.0 - (r / channel_radius).square())


def reciprocal_resistance_pair(
    translation_network: FlowField,
    particle_z: float,
    particle_radius: float,
    ambient_profile: AmbientProfile,
    n_nodes: int = 32,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    r"""Return $(F_A, F_B)$ from the traction of the translating-sphere solution alone.

    $F_B = 2\pi a^2 \sum_i w_i\,\hat t_z(\theta_i)$ is the force per unit velocity, and
    $F_A = -2\pi a^2 \sum_i w_i\,u_{\infty,z}(a\sin\theta_i)\,\hat t_z(\theta_i)$ the force on the fixed sphere.

    Args:
    - `translation_network`: Flow field `(r, z) -> (u_r, u_z, p)` of the sphere translating at unit velocity in still fluid.
    - `particle_z`: Sphere centre.
    - `particle_radius`: Sphere radius, strictly positive. The traction must be taken on the particle surface itself:
      a larger control sphere would involve the unknown disturbance of the fixed-sphere problem.
    - `ambient_profile`: Callable $r \mapsto u_{\infty,z}(r)$, see `poiseuille_axial_profile`.
    - `n_nodes`: Gauss-Legendre nodes in $\cos\theta$, at least 2.
    - `device`, `dtype`: Placement overrides for plain callables.

    Returns:
    - The pair `(ambient_force, translation_force)` as Python floats.

    Raises:
    - `ValueError`: If `n_nodes < 2`, the radius is not positive, or the ambient profile is not finite.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least 2; got {n_nodes!r}.")
    resolved_device, resolved_dtype = _resolve_placement(translation_network, device, dtype)
    theta, weights = sphere_quadrature(n_nodes, "gauss_legendre", resolved_device, resolved_dtype)
    traction = axial_traction(translation_network, particle_z, particle_radius, theta)
    ambient = ambient_profile(particle_radius * torch.sin(theta))
    if not bool(torch.isfinite(ambient).all()):
        raise ValueError("ambient_profile returned non-finite values.")
    prefactor = 2.0 * math.pi * particle_radius**2
    translation_force = (prefactor * (weights * traction).sum()).item()
    ambient_force = (-prefactor * (weights * ambient * traction).sum()).item()
    return ambient_force, translation_force
