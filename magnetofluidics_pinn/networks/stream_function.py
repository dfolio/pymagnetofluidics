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

import torch
from torch import nn

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.device_utils import resolve_module_device, \
    resolve_module_dtype
from magnetofluidics_pinn.physics.fluid_residuals import (
    axisymmetric_vector_laplacian,
    compute_flow_derivatives,
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


def _smooth_unit_step(s: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the $C^\infty$ unit step $\varphi(s)$ with flat junctions.

    $$ \varphi(s) = \frac{f(s)}{f(s) + f(1-s)}, \qquad f(t) = \begin{cases}
    e^{-1/t} & t > 0 \\ 0 & t \le 0 \end{cases} $$

    Satisfies $\varphi \equiv 0$ for $s \le 0$ and $\varphi \equiv 1$ for
    $s \ge 1$, with every derivative vanishing at both junctions, so any
    composite $T = 1 - \varphi$ is $C^\infty$ across its plateau edges.

    Args:
    - `s`: Tensor of any shape.

    Returns:
    - Tensor of the same shape with values in $[0, 1]$.
    """
    # NEW (L9.1): replaces the quintic smoothstep, which was only C^2 and
    # interpreted its argument in rho units (see `_taper`).
    positive = s > 0.0
    below_one = s < 1.0
    f = torch.where(positive, torch.exp(-1.0 / s.clamp(min=1.0e-12)),
                    torch.zeros_like(s))
    g = torch.where(below_one, torch.exp(-1.0 / (1.0 - s).clamp(min=1.0e-12)),
                    torch.zeros_like(s))
    return f / (f + g)


class StreamFunctionSphereFlow(nn.Module):
    r"""Stream-function flow network around an on-axis spherical particle.

    Reconstructs velocity from an axisymmetric Stokes stream function $\psi(r, z)$:
    $$u_r(r, z) = -\frac{1}{r} \frac{\partial \psi}{\partial z}, \qquad
    u_z(r, z) = \frac{1}{r} \frac{\partial \psi}{\partial r},$$
    guaranteeing incompressibility $\nabla \cdot \mathbf{u} \equiv 0$ identically
    everywhere by construction [@happel1983low].

    Physical constraints satisfied structurally:

    - **Continuity**: $\nabla \cdot \mathbf{u} \equiv 0$ identically across all weights.
    - **Axis Regularity**: $u_r(0, z) = 0$ and
      $\left.\frac{\partial u_z}{\partial r}\right|_{r=0} = 0$.
    - **Channel Wall No-Slip**: $\mathbf{u}(R, z) = \mathbf{0}$.
    - **Sphere No-Slip**: $\mathbf{u}|_S = (0, U_p)$ on $\rho = a$.
    - **Far-Field Poiseuille Recovery**: Outside the taper ball ($\rho \ge a + \Delta$),
      the flow matches the undisturbed Hagen-Poiseuille profile identically.

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
       the bump acts inside the ball of radius $a\sqrt{1 + \text{taper\_margin}}$ around
       the sphere, capped at $0.9$ times the available wall-and-end clearance.
    - `particle_velocity`: Backward-compatible keyword alias for `particle_axial_velocity`.

    Raises:
    - `ValueError`: If the sphere does not fit strictly inside the channel with a
      positive taper support, or an argument is invalid.
    """

    radius: torch.Tensor
    length: torch.Tensor
    particle_z: torch.Tensor
    particle_radius: torch.Tensor
    particle_u: torch.Tensor
    flow_amplitude: torch.Tensor
    support: torch.Tensor

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
        # NEW: Backward-compatible keyword alias
    ) -> None:
        super().__init__()
        if particle_radius <= 0.0:
            raise ValueError(
                f"particle_radius must be strictly positive; got {particle_radius!r}.")
        if domain.radius <= 0.0:
            raise ValueError(
                f"domain.radius must be strictly positive; got {domain.radius!r}.")
        if particle_radius >= domain.radius:
            raise ValueError(
                f"particle_radius ({particle_radius!r}) must be strictly smaller than "
                f"domain.radius ({domain.radius!r})."
            )
        clearance = min(
            domain.radius ** 2 - particle_radius ** 2,
            particle_axial_position ** 2 - particle_radius ** 2,
            (domain.length - particle_axial_position) ** 2 - particle_radius ** 2,
        )
        if clearance <= 0.0:
            raise ValueError(
                "The sphere must lie strictly inside the channel, clear of both ends.")
        self.raw_network = raw_network

        # Compute initial scalar parameters in local scope
        r_val = float(domain.radius)
        l_val = float(domain.length)
        z_p_val = float(particle_axial_position)
        a_val = float(particle_radius)

        # CHANGED: Harmonize translational velocity alias
        u_p_val = float(
            particle_velocity if particle_velocity is not None else particle_axial_velocity)
        u_amp_val = float(domain.u_max if flow_amplitude is None else flow_amplitude)

        # CHANGED (L9.1): the margin is a dimensionless multiple of a^2 in
        # rho^2 units; the previous min(1.5*a, R - a) mixed a rho-unit length
        # into a rho^2-unit law.
        default_margin = 1.5

        self._taper_margin = float(
            default_margin if taper_margin is None else taper_margin)
        if self._taper_margin <= 0.0:
            raise ValueError(
                f"taper_margin must be strictly positive; got {self._taper_margin!r}.")

        support_val = float(min(self._taper_margin * a_val ** 2, 0.9 * clearance))

        # CHANGED: Register buffers directly without prior assignment to self.__dict__.
        # This completely resolves `KeyError: "attribute 'radius' already exists"`.
        # 0-D tensor buffers automatically migrate to CUDA when .to(device) is called.
        buffer_specs = (
            ("radius", r_val),
            ("length", l_val),
            ("particle_z", z_p_val),
            ("particle_radius", a_val),
            ("particle_u", u_p_val),
            ("flow_amplitude", u_amp_val),
            ("support", support_val),
        )
        for name, value in buffer_specs:
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float32))

    @property
    def particle_axial_position(self) -> float:
        """Return the sphere center coordinate $z_p$ as a Python float."""
        # CHANGED: Type-safe conversion via .item()
        return float(self.particle_z.item())

    @property
    def particle_axial_velocity(self) -> float:
        """Return the particle translational velocity $U_p$ as a Python float."""
        return float(self.particle_u.item())

    @property
    def particle_velocity(self) -> float:
        """Alias for `particle_axial_velocity` returning a Python float."""
        return self.particle_axial_velocity

    @property
    def u_p(self) -> float:
        """Shorthand alias for particle translational velocity $U_p$."""
        return self.particle_axial_velocity

    @property
    def a(self) -> float:
        """Shorthand alias for particle radius $a$ as a Python float."""
        return float(self.particle_radius.item())

    @property
    def u_peak(self) -> float:
        """Shorthand alias for flow amplitude $U_{\mathrm{peak}}$ as a Python float."""
        return float(self.flow_amplitude.item())

    @property
    def taper_margin(self) -> float:
        r"""Return the physical margin $\Delta$ defining the taper ball radius
        $R_{\mathrm{taper}} = a + \Delta$.

        Returns:
        * Positive float representing the taper margin in domain units.
        """
        # NEW: Public taper margin property
        return self._taper_margin

    def _taper(self, rho: torch.Tensor) -> torch.Tensor:
        r"""Compute the $C^2$-smooth compact-support radial taper function $\tau(\rho)$.

        Evaluates a quintic smoothstep over the normalized distance $\xi = (\rho - a) / \Delta$:
        $$\tau(\rho) = \begin{cases}
            1.0 & \rho \le a, \\
            1 - \left(10\xi^3 - 15\xi^4 + 6\xi^5\right) & a < \rho < a + \Delta, \\
            0.0 & \rho \ge a + \Delta,
        \end{cases}$$
        satisfying $\tau(a) = 1$, $\tau(a + \Delta) = 0$, and $\tau'(a) = \tau''(a) = \tau'(a + \Delta) = \tau''(a + \Delta) = 0$.

        Args:
        - `rho`: Distance-to-center tensor of shape `(n_points, 1)`.

        Returns:
        - Taper factor tensor of the same shape with values in $[0, 1]`.
        """
        # CHANGED (L9.1): consume the support buffer in rho^2 units, matching
        # the taper ball radius a*sqrt(1 + margin); the previous quintic used
        # the margin as a rho-unit length, giving a ball of radius a + m*a
        # that overlapped the wall and voided far-field recovery.
        s = (rho.square() - self.particle_radius.square()) / self.support
        return 1.0 - _smooth_unit_step(s)

    def _raw_inputs(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Map physical coordinates to the network's dimensionless inputs.

        Args:
        - `coordinates`: Spatial coordinates tensor of shape `(n_points, 2)`
          representing $(r, z)$.

        Returns:
        - Tensor of shape `(n_points, 2)` holding
          $\left( r^{2}/R^{2},\ (z - z_p)/R \right)$.
        """
        # NEW (L9.1): particle-relative input contract; the previous
        # (r^2/R^2, z) fed raw channel lengths to the network.
        r_norm_sq = coordinates[:, 0:1].square() / self.radius.square()
        axial_offset = (coordinates[:, 1:2] - self.particle_z) / self.radius
        return torch.cat([r_norm_sq, axial_offset], dim=1)

    def _evaluate_reduced(self, coordinates: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate the raw network to obtain the reduced correction and raw pressure.

        Args:
        - `coordinates`: Spatial coordinates tensor of shape `(n_points, 2)`
          representing $(r, z)$.

        Returns:
        - Tuple `(raw_psi, raw_p)` of shape `(n_points, 1)`.
        """
        # CHANGED (L9.1): route through `_raw_inputs`.
        raw_output = self.raw_network(self._raw_inputs(coordinates))
        return raw_output[:, 0:1], raw_output[:, 1:2]

    def _gauge_pressure(self, raw_p: torch.Tensor,
                        coordinates: torch.Tensor) -> torch.Tensor:
        r"""Anchor the pressure relative to the outlet centerline $p(0, L) = 0$.

        Args:
        - `raw_p`: Ungauged pressure tensor of shape `(n_points, 1)`.
        - `coordinates`: Context coordinate tensor for device and dtype.

        Returns:
        - Gauged pressure tensor of shape `(n_points, 1)`.
        """
        # CHANGED (L9.1): normalize the reference input through the same
        # `_raw_inputs` mapping, otherwise the gauge reads the wrong inputs.
        ref_coord = torch.tensor([[0.0, self.length.item()]], device=coordinates.device,
                                 dtype=coordinates.dtype)
        p_ref = self.raw_network(self._raw_inputs(ref_coord))[:, 1:2]
        return raw_p - p_ref

    def _pressure(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Compute the fully gauged scalar pressure field $p(r, z)$.

        Args:
        - `coordinates`: Tensor of shape `(n_points, 2)` containing $(r, z)$.

        Returns:
        - Gauged pressure tensor of shape `(n_points, 1)`.
        """
        # NEW: Modular pressure evaluator
        _, raw_p = self._evaluate_reduced(coordinates)
        return self._gauge_pressure(raw_p, coordinates)

    def _base_stream_function(self, r: torch.Tensor) -> torch.Tensor:
        r"""Compute the exact Hagen-Poiseuille background stream function $\psi_0(r)$.

        Evaluates:
        $$\psi_0(r) = \int_0^r u_z(r') \, r' \, dr' =
        U_{\mathrm{peak}} \left( \frac{r^2}{2} - \frac{r^4}{4 R^2} \right).$$

        Args:
        * `r`: Radial coordinate tensor of shape `(n_points, 1)`.

        Returns:
        * Background stream function tensor of shape `(n_points, 1)`.
        """
        r_sq = r.square()
        return self.flow_amplitude * (
            0.5 * r_sq - 0.25 * r_sq.square() / (self.radius ** 2))

    def _translating_sphere_stream(self, r: torch.Tensor, z_offset: torch.Tensor,
                                   rho: torch.Tensor) -> torch.Tensor:
        r"""Exact unbounded translating-sphere stream function $\psi_T$.

        $$ \psi_T = \frac{W r^{2}}{4}\left(\frac{3a}{\rho} - \frac{a^{3}}{\rho^{3}}\right),
        \qquad W = U_p - u_{\max}, $$

        the classical Stokes solution whose surface velocity is $W\hat z$ at
        every polar angle and which decays to zero at infinity
        [@happel1983low].

        Args:
        - `r`: Radial coordinates of shape `(n_points, 1)`.
        - `z_offset`: Axial offsets $z - z_p$ of shape `(n_points, 1)`.
        - `rho`: Distances to the sphere center of shape `(n_points, 1)`.

        Returns:
        - Stream-function contribution of shape `(n_points, 1)`.
        """
        # NEW (L9.1): analytic lifting evaluated at the relative velocity.
        relative_velocity = self.particle_u - self.flow_amplitude
        return 0.25 * relative_velocity * r.square() * (
            3.0 * self.particle_radius / rho
            - self.particle_radius.pow(3) / rho.pow(3))

    def _quadratic_shear_stream(self, r: torch.Tensor, z_offset: torch.Tensor,
                                rho: torch.Tensor) -> torch.Tensor:
        r"""Exact fixed-sphere disturbance $\psi_{\text{shear}}$ in the quadratic ambient.

        The local Poiseuille curvature $-\kappa r^{2}\hat z$,
        $\kappa = u_{\max}/R^{2}$, is cancelled on the sphere by the Lamb
        general solution restricted to its $n = 1$ and $n = 3$ modes
        [@happel1983low]:

        $$ \psi_{\text{shear}} = \left(C_1 \rho + \frac{D_1}{\rho}\right)\frac{r^{2}}{\rho^{2}}
        + \frac{3}{2}\,\frac{r^{2}}{\rho^{2}}\left(5\xi^{2} - 1\right)
        \left(\frac{C_3}{\rho} + \frac{D_3}{\rho^{3}}\right), $$

        with $\xi = (z - z_p)/\rho$ and coefficients fixed by $f(a) = f'(a) = 0$
        in each mode: $C_1 = \kappa a^{3}/2$, $D_1 = -3\kappa a^{5}/10$,
        $C_3 = -7\kappa a^{5}/60$, $D_3 = \kappa a^{7}/12$. Its surface velocity
        is $+\kappa r^{2}\hat z$, exactly cancelling the background curvature.

        Args:
        - `r`, `z_offset`, `rho`: See `_translating_sphere_stream`.

        Returns:
        - Stream-function contribution of shape `(n_points, 1)`.
        """
        # NEW (L9.1): parabolic-curvature cancellation lifting.
        curvature = self.flow_amplitude / self.radius.square()
        a = self.particle_radius
        rho_sq = rho.square()
        mode_one = (0.5 * curvature * a.pow(3) * rho
                    - 0.3 * curvature * a.pow(5) / rho) * r.square() / rho_sq
        xi_sq = z_offset.square() / rho_sq
        mode_three = 1.5 * (r.square() / rho_sq) * (5.0 * xi_sq - 1.0) * (
            -7.0 * curvature * a.pow(5) / (60.0 * rho)
            + curvature * a.pow(7) / (12.0 * rho.pow(3)))
        return mode_one + mode_three

    def _neural_correction(self, r: torch.Tensor, z: torch.Tensor,
                           rho: torch.Tensor) -> torch.Tensor:
        r"""Learned correction carrying the triple structural zero.

        $$ r^{2}\,\left(1 - \frac{r^{2}}{R^{2}}\right)^{2}
        \left(\frac{\rho^{2} - a^{2}}{\rho^{2} + a^{2}}\right)^{2} N, $$

        so the correction and its first derivatives vanish on the axis, the
        wall (double zero), and the sphere surface (double zero), for every
        network weight value.

        Args:
        - `r`, `z`: Cylindrical coordinates of shape `(n_points, 1)`.
        - `rho`: Distances to the sphere center of shape `(n_points, 1)`.

        Returns:
        - Correction stream function of shape `(n_points, 1)`.
        """
        # NEW (L9.1): replaces the single-zero tanh envelope, whose nonzero
        # axial gradient at rho = a leaked an O(1) untrained radial velocity
        # onto the surface (assertion error 4.889e+02).
        raw_psi, _ = self._evaluate_reduced(torch.cat([r, z], dim=1))
        axis_factor = r.square()  # CHANGED: psi(0, z) = 0 is now structural.
        wall_factor = (1.0 - r.square() / self.radius.square()).square()
        d = (rho.square() - self.particle_radius.square()) \
            / (rho.square() + self.particle_radius.square())
        return axis_factor * wall_factor * d.square() * raw_psi

    def _stream_function(self, coordinates: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        r"""Evaluate the total stream function $\psi(r, z)$ and physical pressure $p(r, z)$.

        Composes the base profile with the tapered reduced perturbation:
        $$\psi(r, z) = \psi_0(r) + \tau(\rho) \cdot \psi_{\mathrm{reduced}}(r, z).$$

        Args:
        - `coordinates`: Tensor of shape `(n_points, 2)` representing $(r, z)$.

        Returns:
        - Tuple `(psi, p)` of scalar tensors of shape `(n_points, 1)`.
        """
        r = coordinates[:, 0:1]
        z = coordinates[:, 1:2]
        z_offset = z - self.particle_z
        rho = torch.sqrt(r.square() + z_offset.square() + 1.0e-14)

        # CHANGED (L9.1): psi = psi_P + T*(psi_T + psi_shear) + r^2 w d^2 N;
        # the base profile is no longer subtracted inside the taper.
        tau = self._taper(rho)
        psi_analytic = self._translating_sphere_stream(r, z_offset, rho) \
                       + self._quadratic_shear_stream(r, z_offset, rho)
        total_psi = self._base_stream_function(r) \
                    + tau * psi_analytic + self._neural_correction(r, z, rho)
        gauged_p = self._pressure(coordinates)
        return total_psi, gauged_p

    def stokes_residual_sz(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the native Stokes momentum and continuity balance in $(s, z) \equiv (r, z)$.

        Evaluates:
        $$\mathbf{R}_{\mathrm{momentum}} = -\nabla p + \nabla^2 \mathbf{u} = \mathbf{0}, \qquad
        R_{\mathrm{continuity}} = \nabla \cdot \mathbf{u} = 0.$$

        Args:
        - `coordinates`: Tensor of shape `(n_points, 2)` containing interior $(r, z)$
          coordinates requiring gradients.

        Returns:
        - Tensor of shape `(n_points, 3)` with momentum residuals in columns 0 and 1,
          and continuity (identically zero by construction) in column 2.
        """
        param_device = next(self.raw_network.parameters()).device
        param_dtype = next(self.raw_network.parameters()).dtype
        coords = coordinates.to(device=param_device, dtype=param_dtype)
        coords_grad = coords if coords.requires_grad else coords.clone().requires_grad_(
            True)
        r = coords_grad[:, 0:1]

        # Compute flow derivatives with regularized vector Laplacian
        derivs = compute_flow_derivatives(self.forward, coords_grad,
                                          compute_second_order=True)
        lap_r, lap_z = axisymmetric_vector_laplacian(
            derivs=derivs,
            radius=r,
            residual_form="standard",
        )

        momentum_r = -derivs.dp_dr + lap_r
        momentum_z = -derivs.dp_dz + lap_z
        continuity = torch.zeros_like(momentum_r)

        return torch.cat([momentum_r, momentum_z, continuity], dim=1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct the divergence-free $(u_r, u_z, p)$ flow field from the stream function.

        Args:
        * `coordinates`: Tensor of shape `(n_points, 2)` containing $(r, z)$.

        Returns:
        * Tensor of shape `(n_points, 3)` containing `(u_r, u_z, p)`.
        """
        param_device = resolve_module_device(self.raw_network)
        param_dtype = resolve_module_dtype(self.raw_network)
        coords = coordinates.to(device=param_device, dtype=param_dtype)
        coords_grad = coords if coords.requires_grad else coords.clone().requires_grad_(
            True)

        r = coords_grad[:, 0:1]
        # Regularize divisor to avoid zero-division on the symmetry axis
        safe_r = torch.where(r > 1.0e-7, r, torch.full_like(r, 1.0e-7))

        with torch.enable_grad():
            psi, p = self._stream_function(coords_grad)
            grad_psi = scalar_field_gradient(psi, coords_grad)

            # u_r = -(1/r) * dpsi/dz
            dpsi_dz = grad_psi[:, 1:2]
            u_r = -dpsi_dz / safe_r
            u_r = torch.where(r > 1.0e-7, u_r, torch.zeros_like(u_r))

            # u_z = (1/r) * dpsi/dr
            dpsi_dr = grad_psi[:, 0:1]
            u_z = dpsi_dr / safe_r

        return torch.cat([u_r, u_z, p], dim=1)


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
        probe = network(
            torch.zeros(1, 2, device=parameter.device, dtype=parameter.dtype))
    if probe.shape != (1, 2):
        raise ValueError(
            f"network must map (n, 2) to (n, 2); got an output of shape {tuple(probe.shape)}.")
    wrapped = StreamFunctionSphereFlow(
        network, domain, particle_axial_position, particle_radius,
        particle_axial_velocity, flow_amplitude, taper_margin,
    )
    return wrapped.to(device=parameter.device, dtype=parameter.dtype)
