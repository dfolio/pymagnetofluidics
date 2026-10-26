r"""Exterior-domain stream-function network for a sphere in a tube.

Replaces the immersed rigid core of `_particle_core_loss`. The velocity is derived
from a Stokes stream function, so continuity holds identically, and the axis, wall
and sphere conditions are imposed by construction (see the class docstring).
The idea follows the exact-imposition technique of [@sukumar2022exact] and
[@lu2021physics].

The reduced stream function $F$ depends on $(s, z)$ both
explicitly (the analytic liftings) and through the learned correction $N$. The
previous `forward` fed the network with inputs built from $(r, z)$ but
differentiated $F$ with respect to a *separate* tensor `sz`, so the paths through
$N$ were silently dropped and $\div\mathbf u = 0$ was **not** satisfied. The
network inputs are now computed from `sz` itself, so a single autograd path carries
the complete derivative $\partial F / \partial (s, z)$.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from magnetofluidics_pinn.physics.stokes_sphere import (  # NEW
    stokes_reduced_stream_function, stokes_sphere_pressure,
)
from magnetofluidics_pinn.data_types import Domain


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

    $$u_r = r F_z,\qquad u_z = -2\,(F + s F_s),\qquad
    F = F_P + T\,(F_g - F_P) + w\,d^2\,N(s/R^2, z/L).$$

    Constraints that hold for *every* weight value:
    - Continuity: $\div\mathbf u = 0$ identically.
    - Axis: $u_r(0, z) = 0$ and $\partial_r u_z|_{r=0} = 0$ (even in $r$).
    - Wall: $\mathbf u(R, z) = \mathbf 0$ ($w$ has a double zero, $T \equiv 0$ there).
    - Sphere: $\mathbf u = (0, U_p)$ on $\rho = a$ ($d^2$ vanishes together with its gradient,
      $T = 1$, $\nabla T = 0$).
    - Far field: $T \equiv 0$ at both ends, so the base flow is the Poiseuille profile.
    - Flux: $Q(z) = -2\pi R^2 F(R^2, z) = \pi R^2 U_\text{amp}/2$ wherever $T = 0$.

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
    - `taper_margin`: Multiple of $a^2$ setting the taper support in $\rho^2$ units;
       the bump acts inside the ball of radius $a\sqrt{1 + \text{taper\_margin}}$ around the sphere, capped at $0.9$
       times the available wall-and-end clearance.

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
            taper_margin: float = 3,
    ) -> None:
        super().__init__()
        amplitude = domain.u_max if flow_amplitude is None else flow_amplitude
        scalars = (particle_axial_position, particle_radius, particle_axial_velocity, amplitude)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("All scalar arguments must be finite.")
        if not (math.isfinite(taper_margin) and taper_margin > 0.0):
            raise ValueError(f"taper_margin must be finite and strictly positive; got {taper_margin!r}.")
        if not (0.0 < particle_radius < domain.radius):
            raise ValueError("particle_radius must satisfy 0 < a < domain.radius.")
        clearance = min(
            domain.radius ** 2 - particle_radius ** 2,
            particle_axial_position ** 2 - particle_radius ** 2,
            (domain.length - particle_axial_position) ** 2 - particle_radius ** 2,
        )
        if clearance <= 0.0:
            raise ValueError("The sphere must lie strictly inside the channel, clear of both ends.")
        self.raw_network = raw_network
        for name, value in (
                ("radius", domain.radius), ("length", domain.length),
                ("particle_z", particle_axial_position), ("particle_a", particle_radius),
                ("particle_u", particle_axial_velocity), ("amplitude", amplitude),
                ("support", min(taper_margin * particle_radius ** 2, 0.9 * clearance)),
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
        radius_sq = self.radius ** 2
        f_poiseuille = -self.amplitude * (0.5 - s / (4.0 * radius_sq))
        relative_velocity = self.particle_u - self.amplitude  # CHANGED: Stokes lifting uses U_s = U_p - U_max
        f_stokes = stokes_reduced_stream_function(s, z, self.particle_z, self.particle_a, relative_velocity)
        f_shear = -self.amplitude * s / (4.0 * radius_sq)  # CHANGED: curvature part of the rigid-body mismatch
        rho_sq = s + (z - self.particle_z).square()
        wall = (1.0 - s / radius_sq).square()
        distance = (rho_sq - self.particle_a ** 2) / (rho_sq + self.particle_a ** 2)
        return (f_poiseuille + self._taper(s, z) * (f_stokes + f_shear)
                + wall * distance.square() * raw_out[:, 0:1])
    
    def _evaluate_reduced(self, sz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the network and $F$ from a single tensor `sz`.

        CHANGED: the network inputs are derived from `sz`, so differentiating $F$ with
        respect to `sz` includes every path through the learned correction.

        Args:
        - `sz`: Tensor `(n, 2)` with columns $(s, z)$ that is part of the autograd graph.

        Returns:
        - The tuple `(F, raw_out)` with shapes `(n, 1)` and `(n, 2)`.
        """
        net_in = torch.cat(
            [sz[:, 0:1] / self.radius ** 2, (sz[:, 1:2] - self.particle_z) / self.radius], dim=1
        )
        raw_out = self.raw_network(net_in)
        return self._reduced_stream_function(sz, raw_out), raw_out
    
    def _pressure(self, sz: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the pressure before the gauge shift, as a function of `sz`.

        Args:
        - `sz`: Tensor `(n, 2)` with columns $(s, z)$.
        - `raw_out`: Raw network output `(n, 2)` computed from `sz`.

        Returns:
        - Tensor `(n, 1)` with $p_P + T\,p_S + p_\text{net}$.
        """
        s, z = sz[:, 0:1], sz[:, 1:2]
        p_base = -4.0 * self.amplitude * (z - self.length) / self.radius ** 2
        p_stokes = self._taper(s, z) * stokes_sphere_pressure(
            s, z, self.particle_z, self.particle_a, self.particle_u - self.amplitude)
        return p_base + p_stokes + raw_out[:, 1:2]
    
    def _gauge_pressure(self) -> torch.Tensor:
        r"""Return $p_\text{net}(0, L)$, the network pressure at the gauge point.

        Returns:
        - Tensor `(1, 1)`.
        """
        gauge_in = torch.stack([
            torch.zeros_like(self.radius), (self.length - self.particle_z) / self.radius
        ]).unsqueeze(0)
        return self.raw_network(gauge_in)[:, 1:2]
    
    def stokes_residual_sz(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the unweighted axisymmetric Stokes residual without any $1/r$ term.

        With $u_r = r\,g$, $g = F_z$, and $u_z = h = -2(F + sF_s)$, every singular term of the
        cylindrical operator cancels analytically. Differentiating with respect to $(s, z)$ gives

        $$f_r = r\,\big(-2p_s + 4s\,g_{ss} + 8g_s + g_{zz}\big), \qquad
           f_z = -p_z + 4s\,h_{ss} + 4h_s + h_{zz}, \qquad
           \div\mathbf u = 2g + 2s\,g_s + h_z \equiv 0.$$

        These are exactly the columns of `stokes_residual(..., residual_form="standard")`
        (checked symbolically), but they are finite on the axis, need no switching threshold, and
        carry no $r$ or $r^2$ weight. They are therefore the physical force densities, the quantity
        whose volume integral is the force error.

        Args:
        - `coordinates`: Tensor `(n, 2)` with `(r, z)`, any $r \ge 0$. It need not require gradients.

        Returns:
        - Tensor `(n, 3)` with columns `(f_r, f_z, divergence)`, differentiable with respect to
          the network parameters.

        Raises:
        - `ValueError`: If `coordinates` is not `(n, 2)` or contains a negative radius.
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (n, 2); got {tuple(coordinates.shape)}.")
        reference = self.radius
        coords = coordinates.detach().to(device=reference.device, dtype=reference.dtype)
        if torch.any(coords[:, 0] < 0.0):
            raise ValueError("coordinates contain negative radial values.")
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            r = coords[:, 0:1]
            sz = torch.cat([r.square(), coords[:, 1:2]], dim=1).requires_grad_(True)  # leaf in (s, z)
            s = sz[:, 0:1]
            
            def d(value: torch.Tensor) -> torch.Tensor:
                """Gradient of a scalar field with respect to `sz`, keeping the graph."""
                return torch.autograd.grad(
                    value, sz, grad_outputs=torch.ones_like(value), create_graph=True)[0]
            
            f_value, raw_out = self._evaluate_reduced(sz)
            grad_f = d(f_value)
            g = grad_f[:, 1:2]
            h = -2.0 * (f_value + s * grad_f[:, 0:1])
            grad_g, grad_h = d(g), d(h)
            g_ss, g_zz = d(grad_g[:, 0:1])[:, 0:1], d(grad_g[:, 1:2])[:, 1:2]
            h_ss, h_zz = d(grad_h[:, 0:1])[:, 0:1], d(grad_h[:, 1:2])[:, 1:2]
            grad_p = d(self._pressure(sz, raw_out) - self._gauge_pressure())
            f_r = r * (-2.0 * grad_p[:, 0:1] + 4.0 * s * g_ss + 8.0 * grad_g[:, 0:1] + g_zz)
            f_z = -grad_p[:, 1:2] + 4.0 * s * h_ss + 4.0 * grad_h[:, 0:1] + h_zz
            divergence = 2.0 * g + 2.0 * s * grad_g[:, 0:1] + grad_h[:, 1:2]
            residual = torch.cat([f_r, f_z, divergence], dim=1)
        return residual if grad_enabled else residual.detach()
    
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
            f_value, raw_out = self._evaluate_reduced(sz)
            (grad_f,) = torch.autograd.grad(
                f_value, sz, grad_outputs=torch.ones_like(f_value), create_graph=True
            )
            u_r = r * grad_f[:, 1:2]
            u_z = -2.0 * (f_value + s * grad_f[:, 0:1])
            pressure = self._pressure(sz, raw_out) - self._gauge_pressure()
            output = torch.cat([u_r, u_z, pressure], dim=1)
        return output if grad_enabled else output.detach()


def apply_stream_function_constraint(
        network: nn.Module,
        domain: Domain,
        particle_axial_position: float,
        particle_radius: float,
        particle_axial_velocity: float = 0.0,
        flow_amplitude: float | None = None,
        taper_margin: float = 3.0,
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
