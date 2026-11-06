r"""Exterior-domain stream-function network for a sphere in a tube (formulation in $(s, z)$, $s = r^2$).

**What changed with respect to the previous implementation** (marked `# CHANGED` / `# NEW` below):

1. *No division by $r$.* The network works with $\psi = s\,F(s, z)$ and returns
   $u_r = r\,g$, $u_z = 2(F + sF_s)$ with $g = -F_z$. The old $u_z = \psi_r/r$ evaluated to zero at $r = 0$ exactly and
   amplified the float32 round-off of its second derivatives like $1/r^2$.
2. *Analytic pressure.* The pressure is $p = p_P + p_\text{lift} + p_\text{net} - [\cdot](0, L)$. The previous class
   documented $p_P$ but returned only the learned part, so the optimizer had to rebuild the Poiseuille pressure drop
   (and the Stokeslet pressure) next to a large taper residual.
3. *Two wall liftings.* `lifting_mode="taper"` keeps the compactly supported weight, `lifting_mode="wall"` keeps the
   exact unbounded disturbance and corrects the wall only (see `networks.lifting`).
4. *Optional end envelope* for the learned correction, which makes the inlet and outlet velocity exactly Poiseuille.
5. *Geometry as Python floats.* No device buffers, hence no `.item()` synchronization in the forward pass and no
   float32 rounding of the constants when the network is cast to `float64`.

**Structure.** With $F = F_\text{lift} + w\,d^2\,e(z)\,N$, where $w = (1 - s/R^2)^2$, $d = (\rho^2 - a^2)/(\rho^2 + a^2)$ and
$e$ is the optional envelope, the following hold for every weight value: continuity, the axis conditions
$u_r(0, z) = 0$ and $\partial_r u_z|_{r=0} = 0$, the wall condition $\mathbf u(R, z) = \mathbf 0$, and the rigid-body
condition $\mathbf u = (0, U_p)$ on $\rho = a$. The inlet profile, the outlet pressure and the Stokes momentum balance
remain soft terms [@sukumar2022exact; @lu2021physics; @richterpowell2022neural].
"""

from __future__ import annotations

import torch
from torch import nn

from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.networks.lifting import (
    LiftingGeometry,
    LiftingMode,
    build_lifting_geometry,
    fields_from_reduced_stream,
    lifting_pressure,
    reduced_lifting,
    smooth_unit_step,
    squared_distance,
)
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual_sz

_smooth_unit_step = smooth_unit_step  # kept for backward compatibility with code importing the private name
_NETWORK_MODES: tuple[str, ...] = ("taper", "wall")


def bump_function(q: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the compactly supported $C^\infty$ bump $T(q) = \exp\big(1 - 1/(1-q^2)\big)$ for $\abs{q} < 1$ (unchanged).

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
    r"""Stream-function flow network around an on-axis spherical particle in a tube.

    Args:
    - `raw_network`: MLP with 2 inputs $(s/R^2,\ (z - z_p)/R)$ and 2 outputs $(N, p_\text{net})$.
    - `domain`: Nondimensional channel geometry.
    - `particle_axial_position`: Sphere centre $z_p$.
    - `particle_radius`: Sphere radius $a$, with $0 < a < R$.
    - `particle_axial_velocity`: Sphere velocity $U_p$.
    - `flow_amplitude`: Undisturbed centreline velocity; defaults to `domain.u_max`. Use `0.0` for a sphere moving in still fluid.
    - `taper_margin`: Taper support as a multiple of $a^2$ (`"taper"` mode); the taper ball has radius $a\sqrt{1 + m}$, capped
      at $0.9$ times the available clearance. Defaults to `1.5`.
    - `particle_velocity`: Backward-compatible keyword alias for `particle_axial_velocity`.
    - `lifting_mode`: `"taper"` (default) or `"wall"`; see `networks.lifting`.
    - `wall_start_fraction`: Blend start of the `"wall"` mode, $r_0 = a + f(R - a)$ (ignored by the `"taper"` mode).
    - `end_envelope`: If `True`, multiplies the learned correction by $[4z(L - z)/L^2]^2$, so the inlet and outlet velocity are
      exactly the ambient Poiseuille profile. Only valid with the `"taper"` mode, whose analytic disturbance vanishes at the ends.

    Raises:
    - `ValueError`: If the sphere does not fit strictly inside the channel, if `taper_margin <= 0`, if `lifting_mode` is
      unknown, or if `end_envelope` is combined with the `"wall"` mode.
    """

    def __init__(
        self,
        raw_network: nn.Module,
        domain: Domain,
        particle_axial_position: float,
        particle_radius: float,
        particle_axial_velocity: float = 0.0,
        flow_amplitude: float | None = None,
        taper_margin: float | None = None,
        *,
        particle_velocity: float | None = None,
        lifting_mode: LiftingMode = "taper",  # NEW
        end_envelope: bool = False,  # NEW
        wall_start_fraction: float = 0.05,  # NEW
    ) -> None:
        super().__init__()
        if lifting_mode not in _NETWORK_MODES:
            raise ValueError(f"lifting_mode must be one of {_NETWORK_MODES!r}; got {lifting_mode!r}.")
        if end_envelope and lifting_mode != "taper":
            raise ValueError("end_envelope requires lifting_mode='taper' (the wall mode leaves a disturbance tail at the ends).")
        margin = 1.5 if taper_margin is None else float(taper_margin)
        velocity = float(particle_velocity if particle_velocity is not None else particle_axial_velocity)
        amplitude = float(domain.u_max if flow_amplitude is None else flow_amplitude)
        self.raw_network = raw_network
        # CHANGED: constants live in an immutable Python-float record instead of float32 device buffers.
        self._geometry: LiftingGeometry = build_lifting_geometry(
            domain.radius, domain.length, particle_axial_position, particle_radius, velocity, amplitude, margin,
            wall_start_fraction,
        )
        self._taper_margin = margin
        self._lifting_mode: LiftingMode = lifting_mode
        self._end_envelope = bool(end_envelope)

    # ------------------------------------------------------------------ read-only descriptors
    @property
    def geometry(self) -> LiftingGeometry:
        """Return the immutable geometry record."""
        return self._geometry

    @property
    def lifting_mode(self) -> LiftingMode:
        """Return the wall-lifting mode, `"taper"` or `"wall"`."""
        return self._lifting_mode

    @property
    def end_envelope(self) -> bool:
        """Return whether the learned correction carries the inlet/outlet envelope."""
        return self._end_envelope

    @property
    def radius(self) -> float:
        """Return the tube radius $R$ as a Python float."""
        return self._geometry.radius

    @property
    def length(self) -> float:
        """Return the tube length $L$ as a Python float."""
        return self._geometry.length

    @property
    def particle_z(self) -> float:
        """Return the sphere centre $z_p$ as a Python float."""
        return self._geometry.particle_z

    @property
    def particle_radius(self) -> float:
        """Return the sphere radius $a$ as a Python float."""
        return self._geometry.particle_radius

    @property
    def particle_u(self) -> float:
        """Return the sphere velocity $U_p$ as a Python float."""
        return self._geometry.particle_velocity

    @property
    def flow_amplitude(self) -> float:
        """Return the ambient centreline velocity as a Python float."""
        return self._geometry.flow_amplitude

    @property
    def support(self) -> float:
        r"""Return the taper support in $\rho^2$ units."""
        return self._geometry.taper_support

    @property
    def particle_axial_position(self) -> float:
        """Return the sphere centre $z_p$ as a Python float."""
        return self._geometry.particle_z

    @property
    def particle_axial_velocity(self) -> float:
        """Return the sphere velocity $U_p$ as a Python float."""
        return self._geometry.particle_velocity

    @property
    def particle_velocity(self) -> float:
        """Alias for `particle_axial_velocity`."""
        return self._geometry.particle_velocity

    @property
    def u_p(self) -> float:
        """Shorthand alias for the sphere velocity $U_p$."""
        return self._geometry.particle_velocity

    @property
    def a(self) -> float:
        """Shorthand alias for the sphere radius $a$."""
        return self._geometry.particle_radius

    @property
    def u_peak(self) -> float:
        """Shorthand alias for the ambient amplitude $U$."""
        return self._geometry.flow_amplitude

    @property
    def wall_start_fraction(self) -> float:
        r"""Return the blend start fraction $f$ with $r_0 = a + f(R - a)$."""
        g = self._geometry
        return (g.wall_start_radius - g.particle_radius) / (g.radius - g.particle_radius)

    @property
    def taper_margin(self) -> float:
        r"""Return the taper margin $m$ (support $= \min(m a^2,\ 0.9\,\text{clearance})$)."""
        return self._taper_margin

    # ------------------------------------------------------------------ building blocks
    def _raw_inputs(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""Map $(s, z)$ to the network inputs $\left(s/R^2,\ (z - z_p)/R\right)$.

        Args:
        - `s`, `z`: Tensors `(n, 1)`.

        Returns:
        - Tensor `(n, 2)`.
        """
        geometry = self._geometry
        return torch.cat([s / geometry.radius**2, (z - geometry.particle_z) / geometry.radius], dim=1)

    def _structural_factor(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""Return $w\,d^2\,e(z)$, the factor that gives the learned correction its structural zeros.

        $w = (1 - s/R^2)^2$ is a double zero at the wall, $d^2$ a double zero on the sphere surface and $e$ the optional envelope.
        The axis needs no factor because $\psi = sF$.

        Args:
        - `s`, `z`: Tensors `(n, 1)`.

        Returns:
        - Tensor `(n, 1)`.
        """
        geometry = self._geometry
        rho2 = squared_distance(s, z, geometry)
        a2 = geometry.particle_radius**2
        wall_factor = (1.0 - s / geometry.radius**2).square()
        distance_factor = ((rho2 - a2) / (rho2 + a2)).square()
        factor = wall_factor * distance_factor
        if self._end_envelope:  # NEW
            factor = factor * (4.0 * z * (geometry.length - z) / geometry.length**2).square()
        return factor

    def _total_pressure(self, s: torch.Tensor, z: torch.Tensor, raw_pressure: torch.Tensor) -> torch.Tensor:
        """Return the ungauged pressure `p_lift + p_net`.

        Args:
        - `s`, `z`: Tensors `(n, 1)`.
        - `raw_pressure`: Learned pressure head `(n, 1)`.

        Returns:
        - Tensor `(n, 1)`.
        """
        return lifting_pressure(s, z, self._geometry, self._lifting_mode) + raw_pressure

    def _reference_pressure(self, like: torch.Tensor) -> torch.Tensor:
        """Return the ungauged pressure at the outlet axis point $(s, z) = (0, L)$, shape `(1, 1)`.

        Args:
        - `like`: Tensor whose dtype and device the reference point inherits.

        Returns:
        - Tensor `(1, 1)`.
        """
        s_ref = torch.zeros(1, 1, dtype=like.dtype, device=like.device)
        z_ref = torch.full((1, 1), self._geometry.length, dtype=like.dtype, device=like.device)
        raw = self.raw_network(self._raw_inputs(s_ref, z_ref))
        return self._total_pressure(s_ref, z_ref, raw[:, 1:2])  # CHANGED: no `.item()`, no device synchronization

    # ------------------------------------------------------------------ fields
    def reduced_stream(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""Return the full reduced stream function $F = F_\text{lift} + w\,d^2\,e\,N$ (velocity part only).

        Args:
        - `s`, `z`: Tensors `(n, 1)`.

        Returns:
        - Tensor `(n, 1)`.
        """
        correction = self.raw_network(self._raw_inputs(s, z))[:, 0:1]
        return reduced_lifting(s, z, self._geometry, self._lifting_mode) + self._structural_factor(s, z) * correction

    def fields_sz(self, sz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Return $(g, u_z, p)$ as differentiable functions of $(s, z)$, with $u_r = r\,g$.

        Args:
        - `sz`: Tensor `(n, 2)` holding $(s, z)$ and requiring gradients (a leaf, or an intermediate of the graph).

        Returns:
        - The tuple `(g, u_z, p)` of tensors `(n, 1)`, graph attached. The pressure is gauged so that $p(0, L) = 0$.
        """
        s, z = sz[:, 0:1], sz[:, 1:2]
        raw = self.raw_network(self._raw_inputs(s, z))
        reduced = (reduced_lifting(s, z, self._geometry, self._lifting_mode)
                   + self._structural_factor(s, z) * raw[:, 0:1])
        g, u_z = fields_from_reduced_stream(reduced, sz)
        pressure = self._total_pressure(s, z, raw[:, 1:2]) - self._reference_pressure(s)
        return g, u_z, pressure

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Return $(u_r, u_z, p)$ at $(r, z)$ coordinates.

        Args:
        - `coordinates`: Tensor `(n, 2)` with $(r, z)$, $r \ge 0$. If it requires gradients, the graph is kept, so that
          further derivatives with respect to the coordinates are available.

        Returns:
        - Tensor `(n, 3)` with columns $(u_r, u_z, p)$. The output is detached if the caller runs under `torch.no_grad()`.
        """
        dtype, device = resolve_module_dtype(self.raw_network), resolve_module_device(self.raw_network)
        coords = coordinates.to(device=device, dtype=dtype)
        keep_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            radial = coords[:, 0:1]
            if coords.requires_grad:
                sz = torch.cat([radial.square(), coords[:, 1:2]], dim=1)
            else:
                sz = torch.cat([radial.detach().square(), coords[:, 1:2].detach()], dim=1).requires_grad_(True)
            g, u_z, pressure = self.fields_sz(sz)
            output = torch.cat([radial * g, u_z, pressure], dim=1)  # CHANGED: u_r = r g, u_z regular on the axis
        return output if keep_graph else output.detach()

    def stokes_residual_sz(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the native Stokes residual at $(r, z)$ coordinates.

        Delegates to `physics.fluid_residuals.stokes_residual_sz`. This is the standard force density written in $(s, z)$,
        so it matches `stokes_residual(..., residual_form="standard")` away from the axis and stays finite on it.

        Args:
        - `coordinates`: Tensor `(n, 2)` with $(r, z)$; gradients with respect to them are not needed.

        Returns:
        - Tensor `(n, 3)` with columns $(R_r, R_z, \nabla\cdot\mathbf u)$, graph attached to the network parameters.
        """
        dtype, device = resolve_module_dtype(self.raw_network), resolve_module_device(self.raw_network)
        coords = coordinates.detach().to(device=device, dtype=dtype)
        with torch.enable_grad():
            sz = torch.cat([coords[:, 0:1].square(), coords[:, 1:2]], dim=1).requires_grad_(True)
            return stokes_residual_sz(self.fields_sz, sz)


def apply_stream_function_constraint(
    network: nn.Module,
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    particle_axial_velocity: float = 0.0,
    flow_amplitude: float | None = None,
    taper_margin: float = 3.0,
    *,
    lifting_mode: LiftingMode = "taper",  # NEW
    end_envelope: bool = False,  # NEW
    wall_start_fraction: float = 0.05,  # NEW
) -> nn.Module:
    """Wrap a 2-input, 2-output MLP as a `StreamFunctionSphereFlow`.

    Args:
    - `network`: MLP with 2 inputs and 2 outputs (for example `build_mlp(2, 2, ...)`).
    - `domain`, `particle_axial_position`, `particle_radius`, `particle_axial_velocity`, `flow_amplitude`, `taper_margin`,
      `lifting_mode`, `end_envelope`, `wall_start_fraction`: See the class docstring.

    Returns:
    - The wrapped module, placed on the device and dtype of `network`.

    Raises:
    - `ValueError`: If `network` does not map `(n, 2)` to `(n, 2)`, or on invalid geometry or options.
    """
    parameter = next(network.parameters())
    with torch.no_grad():
        probe = network(torch.zeros(1, 2, device=parameter.device, dtype=parameter.dtype))
    if probe.shape != (1, 2):
        raise ValueError(f"network must map (n, 2) to (n, 2); got an output of shape {tuple(probe.shape)}.")
    wrapped = StreamFunctionSphereFlow(
        network, domain, particle_axial_position, particle_radius, particle_axial_velocity, flow_amplitude,
        taper_margin, lifting_mode=lifting_mode, end_envelope=end_envelope, wall_start_fraction=wall_start_fraction,
    )
    return wrapped.to(device=parameter.device, dtype=parameter.dtype)
