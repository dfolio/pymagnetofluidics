r"""Exterior-domain stream-function network for a sphere in a tube.

Replaces the immersed rigid core of `_particle_core_loss`. The velocity is derived
from a Stokes stream function, so continuity holds identically, and the axis, wall
and sphere conditions are imposed by construction (see the class docstring).
The idea follows the exact-imposition technique of [@sukumar2022exact] and
[@lu2021physics].
"""

from __future__ import annotations

import math

import torch
from torch import nn

from magnetofluidics_pinn.types import Domain
from magnetofluidics_pinn.physics.stokes_sphere import (  # NEW
    stokes_reduced_stream_function, stokes_sphere_pressure,
)


def bump_function(q: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the compactly supported $C^\infty$ bump $T(q) = \exp\big(1 - 1/(1-q^2)\big)$ for $\abs{q} < 1$.

    $T(0) = 1$, $T'(0) = 0$, and $T \equiv 0$ for $\abs{q} \ge 1$.

    Args:
    - `q`: Tensor of any shape.

    Returns:
    - Tensor of the same shape with values in $[0, 1]$.
    """
    q_sq = q.square()
    inside = q_sq < 1.0
    safe = torch.where(inside, 1.0 - q_sq, torch.ones_like(q_sq))
    return torch.where(inside, torch.exp(1.0 - 1.0 / safe), torch.zeros_like(q_sq))


class StreamFunctionSphereFlow(nn.Module):
    r"""Flow around an on-axis sphere in a tube, from $\psi = s\,F(s, z)$ with $s = r^2$.

    $$ u_r = r F_z,\qquad u_z = -2\,(F + s F_s),\qquad
    F = F_P + T\,(F_g - F_P) + w\,d^2\,N(s/R^2, z/L). $$

    Constraints that hold for *every* weight value:
    - Continuity: $\div\mathbf u = 0$ identically.
    - Axis: $u_r(0, z) = 0$ and $\partial_r u_z|_{r=0} = 0$ (even in $r$).
    - Wall: $\mathbf u(R, z) = \mathbf 0$ ($w$ has a double zero, $T \equiv 0$ there).
    - Sphere: $\mathbf u = (0, U_p)$ on $\rho = a$ ($d$ vanishes, $T = 1$, $\nabla T = 0$).
    - Far field: $T \equiv 0$ at both ends, so the base flow is the Poiseuille profile.

    The inlet profile, outlet pressure and the PDE itself remain soft. Pressure is
    $p = p_P + p_\text{net} - p_\text{net}(0, L)$.

    Args:
    - `raw_network`: MLP with 2 inputs $(s/R^2,\ z/L)$ and 2 outputs $(N, p_\text{net})$.
    - `domain`: Nondimensional channel geometry.
    - `particle_axial_position`: Sphere center $z_p$.
    - `particle_radius`: Sphere radius $a$, with $0 < a < R$.
    - `particle_axial_velocity`: Sphere velocity $U_p$.
    - `flow_amplitude`: Undisturbed centerline velocity; defaults to `domain.u_max`.
      Use `0.0` for a sphere translating in still fluid.
    - `taper_margin`: Fraction in $(0, 1)$ of the available clearance used as taper support.

    Raises:
    - `ValueError`: If the sphere does not fit strictly inside the channel with a
      positive taper support, or an argument is invalid.
    """

    def __init__(
        self,
        raw_network: nn.Module,
        domain: Domain,
        particle_axial_position: float,
        particle_radius: float,
        particle_axial_velocity: float = 0.0,
        flow_amplitude: float | None = None,
        taper_margin: float = 0.9,
    ) -> None:
        super().__init__()
        amplitude = domain.u_max if flow_amplitude is None else flow_amplitude
        scalars = (particle_axial_position, particle_radius, particle_axial_velocity, amplitude)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("All scalar arguments must be finite.")
        if not (0.0 < taper_margin < 1.0):
            raise ValueError(f"taper_margin must lie in (0, 1); got {taper_margin!r}.")
        if not (0.0 < particle_radius < domain.radius):
            raise ValueError("particle_radius must satisfy 0 < a < domain.radius.")
        clearance = min(
            domain.radius**2 - particle_radius**2,
            particle_axial_position**2 - particle_radius**2,
            (domain.length - particle_axial_position) ** 2 - particle_radius**2,
        )
        if clearance <= 0.0:
            raise ValueError("The sphere must lie strictly inside the channel, clear of both ends.")
        self.raw_network = raw_network
        for name, value in (
            ("radius", domain.radius), ("length", domain.length),
            ("particle_z", particle_axial_position), ("particle_a", particle_radius),
            ("particle_u", particle_axial_velocity), ("amplitude", amplitude),
            ("support", taper_margin * clearance),
        ):
            # Buffers travel with `state_dict`, so a checkpoint carries the full constraint.
            self.register_buffer(name, torch.tensor(float(value)))
    
    def _taper(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # NEW
        """Evaluate the compactly supported taper $T$ at $(s, z)$.

        Args:
        - `s`: Tensor `(n, 1)` with $r^2$.
        - `z`: Tensor `(n, 1)` with the axial coordinate.

        Returns:
        - Tensor `(n, 1)` with $T \\in [0, 1]$.
        """
        rho_sq = s + (z - self.particle_z).square()
        return bump_function((rho_sq - self.particle_a ** 2) / self.support)
    
    def _reduced_stream_function(self, sz: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
        """Evaluate $F(s, z)$.

        Args:
        - `sz`: Tensor `(n, 2)` with columns $(s, z)$.
        - `raw_out`: Raw network output `(n, 2)`.

        Returns:
        - Tensor `(n, 1)` with $F$.
        """
        s, z = sz[:, 0:1], sz[:, 1:2]
        radius_sq = self.radius**2
        f_poiseuille = -self.amplitude * (0.5 - s / (4.0 * radius_sq))
        relative_velocity = self.particle_u - self.amplitude  # CHANGED: Stokes lifting uses U_s = U_p - U_max
        f_stokes = stokes_reduced_stream_function(s, z, self.particle_z, self.particle_a, relative_velocity)
        f_shear = -self.amplitude * s / (4.0 * radius_sq)  # CHANGED: curvature part of the rigid-body mismatch
        rho_sq = s + (z - self.particle_z).square()
        wall = (1.0 - s / radius_sq).square()
        # CHANGED: bounded distance in [0, 1). It keeps the double zero on the sphere but removes the
        # O(1/a^2) amplification that produced the 1e9 initial loss.
        distance = (rho_sq - self.particle_a ** 2) / (rho_sq + self.particle_a ** 2)
        return (f_poiseuille + self._taper(s, z) * (f_stokes + f_shear)
            + wall * distance.square() * raw_out[:, 0:1])

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map `(r, z)` to `(u_r, u_z, p)`.

        Args:
        - `coordinates`: Tensor `(n, 2)`. If it requires gradients, the output stays
          differentiable with respect to it, as the PDE residual needs.

        Returns:
        - Tensor `(n, 3)`.

        Raises:
        - `ValueError`: If `coordinates` is not `(n, 2)`.
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (n, 2); got {tuple(coordinates.shape)}.")
        reference = self.radius
        coords = coordinates.to(device=reference.device, dtype=reference.dtype)
        grad_enabled = torch.is_grad_enabled()
        if not coords.requires_grad:
            coords = coords.detach().requires_grad_(True)
        with torch.enable_grad():
            r, z = coords[:, 0:1], coords[:, 1:2]
            s = r.square()
            sz = torch.cat([s, z], dim=1)
            # CHANGED: local axial input (z - z_p)/R instead of z/L, so the sphere spans O(1) of the input range.
            net_in = torch.cat([s / self.radius ** 2, (z - self.particle_z) / self.radius], dim=1)
            raw_out = self.raw_network(net_in)
            f_value = self._reduced_stream_function(sz, raw_out)
            (grad_f,) = torch.autograd.grad(
                f_value, sz, grad_outputs=torch.ones_like(f_value), create_graph=True
            )
            u_r = r * grad_f[:, 1:2]
            u_z = -2.0 * (f_value + s * grad_f[:, 0:1])
            # CHANGED: the gauge point p(0, L) now uses the same local input.
            gauge_in = torch.stack([
                torch.zeros_like(self.radius), (self.length - self.particle_z) / self.radius
            ]).unsqueeze(0)
            p_ref = self.raw_network(gauge_in)[:, 1:2]
            p_base = -4.0 * self.amplitude * (z - self.length) / self.radius**2
            # NEW: Stokes pressure lifting, tapered like the velocity lifting.
            p_stokes = self._taper(s, z) * stokes_sphere_pressure(
                s, z, self.particle_z, self.particle_a, self.particle_u - self.amplitude)
            pressure = p_base + p_stokes + raw_out[:, 1:2] - p_ref
            output = torch.cat([u_r, u_z, pressure], dim=1)
        return output if grad_enabled else output.detach()


def apply_stream_function_constraint(
    network: nn.Module,
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    particle_axial_velocity: float = 0.0,
    flow_amplitude: float | None = None,
    taper_margin: float = 0.9,
) -> nn.Module:
    """Wrap a 2-input, 2-output MLP as a [`StreamFunctionSphereFlow`][magnetofluidics_pinn.networks.stream_function.StreamFunctionSphereFlow].

    Args:
    - `network`: MLP with 2 inputs and 2 outputs (for example `build_mlp(2, 2, ...)`).
    - `domain`, `particle_axial_position`, `particle_radius`, `particle_axial_velocity`,
      `flow_amplitude`, `taper_margin`: See the class docstring.

    Returns:
    - The wrapped module, placed on the device and dtype of `network`.

    Raises:
    - `ValueError`: If `network` does not map `(n, 2)` to `(n, 2)`, or on invalid geometry.
    """
    parameter = next(network.parameters())
    with torch.no_grad():
        probe = network(torch.zeros(1, 2, device=parameter.device, dtype=parameter.dtype))
    if probe.shape != (1, 2):
        raise ValueError(f"network must map (n, 2) to (n, 2); got an output of shape {tuple(probe.shape)}.")
    wrapped = StreamFunctionSphereFlow(
        network, domain, particle_axial_position, particle_radius,
        particle_axial_velocity, flow_amplitude, taper_margin,
    )
    return wrapped.to(device=parameter.device, dtype=parameter.dtype)
