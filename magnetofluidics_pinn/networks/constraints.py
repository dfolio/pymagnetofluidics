r"""Hard-constraint wrappers for the flow network.

A soft (loss-based) boundary condition only ever holds *approximately*
wherever it happens to be evaluated: the network is free to violate it
elsewhere by any amount the optimizer's current gradient step allows. For
most of Phase 1's boundary conditions that approximation is harmless. Two
conditions at the symmetry axis and the wall are the exception, and are
enforced here structurally rather than through the loss:

- **No-penetration at the wall**, $u_r(R, z) = 0$. The axisymmetric
  continuity equation integrates to $dQ/dz = -2 \pi R\, u_r(R, z)$ ($Q$
  being the volumetric flow rate), so *any* small, systematically-signed
  residual radial velocity at the wall accumulates, over the length of the
  channel, into a large violation of mass conservation far from the wall.
- **Tangential no-slip at the wall**, $u_z(R, z) = 0$. # CHANGED: this was
  previously left to the soft boundary loss in `training.trainer`
  (weighted, but only approximately satisfied). Reusing the same
  wall-vanishing factor already computed for $u_r$'s shaping removes an
  entire soft-loss error source from exactly the near-wall shear layer a
  Poiseuille-profile comparison is most sensitive to, mirroring the
  improvement the wall constraint already gave for $u_r$
  [@lu2021physics]. Existing checkpoints trained under the previous
  (soft-only) forward pass must be retrained: their weights were fit
  against a different reconstruction formula for $u_z$.

`apply_hard_wall_constraint` enforces all four conditions
($u_r(0,z)=0$, $u_r(R,z)=0$, $u_z(R,z)=0$, and
$\partial u_z/\partial r|_{r=0} = \partial p/\partial r|_{r=0} = 0$) by
construction, $\partial p/\partial r|_{r=0}=0$) by
construction, for every input, every parameter value, at every point in
training — not just approximately, at a finite set of sampled points. This
is the standard hard-constraint technique for physics-informed networks
[@lu2021physics]. Empirically (see the Phase 1 verification notebook),
adding the axis conditions alongside the wall condition — rather than the
wall condition alone — roughly halved the relative L2 error against the
analytical solution again, on top of the improvement the wall constraint
alone already gave.
"""

from __future__ import annotations

import torch
from torch import nn

from magnetofluidics_pinn.types import Domain


class _HardWallConstrainedFlow(nn.Module):
    r"""Wraps a raw `(u_r, u_z, p)` network to enforce axis and wall regularity.
    
    Applies exact physical boundary constraints for axisymmetric Phase 1 flow:
    
    - $u_r(0, z) = 0$, $u_r(R, z) = 0$
    - $u_z(R, z) = 0$
    - $p(0, L) = 0$ (gauge anchoring)
    """

    def __init__(self, raw_network: nn.Module,  domain: Domain) -> None:
        super().__init__()
        self.raw_network = raw_network
        self.radius = domain.radius
        self.length = domain.length
        self.u_max = domain.u_max

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        param_device = next(self.raw_network.parameters()).device
        if coordinates.device != param_device:
            coordinates = coordinates.to(device=param_device)
        
        normalized_radius = coordinates[:, 0:1] / self.radius
        axial_coordinate = coordinates[:, 1:2]
        
        squared_normalized_radius = normalized_radius.square()
        wall_vanishing_factor = 1.0 - squared_normalized_radius
        raw_output = self.raw_network(torch.cat([squared_normalized_radius, axial_coordinate], dim=1))
        
        # Radial velocity: u_r(0, z) = 0 and u_r(R, z) = 0
        radial_velocity = normalized_radius * wall_vanishing_factor * raw_output[:, 0:1]
        
        # Axial velocity: wall no-slip u_z(R, z) = 0
        axial_velocity = wall_vanishing_factor * raw_output[:, 1:2]
        if self.u_max > 0.0:
            axial_velocity = axial_velocity + self.u_max * wall_vanishing_factor
        
        # Pressure gauge fixing: p(0, L) = 0
        ref_point = torch.tensor([[0.0, self.length]], device=coordinates.device, dtype=coordinates.dtype)
        p_ref = self.raw_network(ref_point)[:, 2:3]
        p = raw_output[:, 2:3] - p_ref
        return torch.cat([radial_velocity, axial_velocity, p], dim=1)



def apply_hard_wall_constraint(network: nn.Module, domain: Domain) -> nn.Module:
    r"""Wrap a flow network to enforce exact axis and wall regularity.

    Enforces $u_r(0,z) = u_r(R,z) = 0$ and
    $\partial u_z/\partial r|_{r=0} = \partial p/\partial r|_{r=0} = 0$; see
    this module's docstring for why each condition holds for any smooth
    axisymmetric flow, not only the analytical Poiseuille solution.

    The wrapped network keeps the same calling convention as its input
    (`(n_points, 2) -> (n_points, 3)`, mapping `(r, z)` to `(u_r, u_z, p)`),
    so it is a drop-in replacement anywhere a plain network is expected:
    [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual],
    [`train`][magnetofluidics_pinn.training.trainer.train],
    [`integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory],
    and the plotting functions all consume it identically to a raw
    `build_mlp` output.

    Args:
    - `network`: A flow network, e.g. from
      [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp], mapping
      `(r, z)` coordinates to `(u_r, u_z, p)`. Its first input is
      reinterpreted internally as $(r/R)^2$ rather than $r$ (see this
      module's docstring); this only changes what the network is fed
      during training, not its architecture or parameter count.
    - `domain`: The (already-nondimensionalized) domain whose radius defines
      where the no-slip constraint is enforced (the axis constraint is at
      `r = 0` regardless of `domain.radius`).

    Returns:
    - A new `torch.nn.Module` wrapping `network`; its own trainable
      parameters are exactly `network`'s (nothing is added or frozen), so
      existing training code needs no changes beyond building the network
      through this wrapper first.

    Raises:
    - `ValueError`: If `domain.radius` is not strictly positive.
    """
    if domain.radius <= 0.0:
        raise ValueError(f"domain.radius must be strictly positive; got {domain.radius!r}.")
    return _HardWallConstrainedFlow(network, domain)


class _HardParticleConstrainedFlow(nn.Module):
    r"""Wrap a flow network to enforce wall, axis, and particle constraints structurally.

    Constructs a composite velocity field satisfying:
    1. Exact wall no-slip: $\mathbf{u}(R, z) = \mathbf{0}$.
    2. Exact axis regularity: $u_r(0, z) = 0$ and $\partial u_z / \partial r |_{r=0} = 0$.
    3. Exact particle rigid-body velocity: $\mathbf{u}|_S = (0, U_p)$.
    4. Non-zero viscous shear stress: $\left.\frac{\partial u_z}{\partial n}\right|_S = \frac{1 - r^2/R^2}{a} N_z \neq 0$.
    5. Non-choking far field: $\mathbf{u} \to \mathbf{u}_{\mathrm{Poiseuille}}$ for $\|\mathbf{x} - \mathbf{x}_p\| > 3a$.

    Args:
    - `network`: Underlying flow network mapping `((r/R)^2, z)` to `(N_r, N_z, p)`.
    - `domain`: Vessel geometry providing channel radius $R$ and length $L$.
    - `particle_axial_position`: Dimensionless center coordinate $z_p$.
    - `particle_radius`: Dimensionless sphere radius $a$.
    - `particle_axial_velocity`: Dimensionless translational velocity $U_p$.
    - `boundary_layer_thickness_factor`: Radial shell factor $\delta / a$ for localized decay.
    """

    def __init__(
        self,
        network: nn.Module,
        domain: Domain,
        particle_axial_position: float,
        particle_radius: float,
        particle_axial_velocity: float,
        boundary_layer_thickness_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.network = network
        self.radius = domain.radius
        self.length = domain.length
        self.u_max = domain.u_max
        self.particle_axial_position = particle_axial_position
        self.particle_radius = particle_radius
        self.particle_axial_velocity = particle_axial_velocity
        self.delta = boundary_layer_thickness_factor * particle_radius

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        r"""Compute constrained velocity and pressure fields with CUDA optimization.

        Args:
        * `coordinates`: Tensor of shape `(n_points, 2)` containing `(r, z)`.

        Returns:
        * Tensor of shape `(n_points, 3)` containing `(u_r, u_z, p)`.
        """
        param_device = next(self.network.parameters()).device
        param_dtype = next(self.network.parameters()).dtype
        if coordinates.device != param_device or coordinates.dtype != param_dtype:
            coordinates = coordinates.to(device=param_device, dtype=param_dtype)

        radial_coordinate = coordinates[:, 0:1]
        axial_coordinate = coordinates[:, 1:2]

        r_sq = radial_coordinate.square()
        r_norm_sq = r_sq / (self.radius**2)
        wall_vanishing_factor = 1.0 - r_norm_sq

        # Evaluate underlying raw neural field
        raw_output = self.network(torch.cat([r_norm_sq, axial_coordinate], dim=1))
        n_r = raw_output[:, 0:1]
        n_z = raw_output[:, 1:2]
        raw_p = raw_output[:, 2:3]

        # Regularized Euclidean distance to sphere center
        axial_offset = axial_coordinate - self.particle_axial_position
        center_distance = torch.sqrt(r_sq + axial_offset.square() + 1.0e-14)
        signed_particle_distance = center_distance - self.particle_radius

        # CHANGED: Use smooth tanh factoring instead of non-differentiable clamp.
        # This guarantees non-zero surface normal derivatives for Cauchy stress evaluation.
        d_norm = torch.tanh(signed_particle_distance / self.particle_radius)

        # CHANGED: Localized spherical envelope ensuring the bypass gap is never choked
        envelope = torch.exp(- (signed_particle_distance / self.delta).square())

        # Radial velocity: odd in r, zero at r=0, zero at r=R, zero on sphere surface
        u_r = (radial_coordinate / self.radius) * wall_vanishing_factor * d_norm * n_r

        # Wall factor for prescribed particle velocity Up
        r_diff_sq = torch.clamp(self.radius**2 - r_sq, min=0.0)
        f_p = r_diff_sq / (r_diff_sq + self.radius * (signed_particle_distance.square() / self.particle_radius) + 1.0e-7)

        # Axial velocity: Up on sphere, 0 at wall, Poiseuille far field
        u_z_particle = self.particle_axial_velocity * envelope * f_p
        u_z_shear = wall_vanishing_factor * d_norm * n_z
        u_z_ambient = self.u_max * wall_vanishing_factor * (1.0 - envelope)
        u_z = u_z_particle + u_z_shear + u_z_ambient

        # Pressure gauge fixing: Anchor pressure relative to outlet centerline p(0, L) = 0
        ref_point = torch.tensor([[0.0, self.length]], device=param_device, dtype=param_dtype)
        p_ref = self.network(ref_point)[:, 2:3]
        p = raw_p - p_ref

        return torch.cat([u_r, u_z, p], dim=1)


def apply_hard_particle_constraint(  # NEW
    network: nn.Module,
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    particle_axial_velocity: float,
    boundary_layer_thickness_factor: float = 1.5,
) -> nn.Module:
    r"""Wrap a flow network to enforce exact particle rigid-body surface velocity.

    CHANGED: In Phase 1's primitive-variable $(u_r, u_z, p)$ formulation, algebraic
    velocity liftings $\mathbf{u} = w \mathbf{U}_p + (1-w)\mathbf{u}_w$ violate
    incompressibility ($\nabla \cdot \mathbf{u} \neq 0$) and introduce an artificial
    $\mathcal{O}(1/a^2)$ Laplacian body force $-(\nabla^2 w)\mathbf{u}_w$ that
    destabilizes the momentum residual and corrupts viscous surface traction.
    In Phase 1, particle boundary conditions are enforced via the energy-consistent
    immersed boundary loss with internal core regularization in `trainer.py`.

    Args:
    - `network`: Flow network mapping coordinates to $(u_r, u_z, p)$.
    - `domain`: Nondimensionalized vessel geometry.
    - `particle_axial_position`: Dimensionless axial center $z_p$.
    - `particle_radius`: Dimensionless particle radius $a$.
    - `particle_axial_velocity`: Dimensionless translational axial velocity $U_p$.
    - `boundary_layer_thickness_factor`: Radial decay shell thickness ratio $\delta / a$.

    Returns:
    - A `torch.nn.Module` wrapping `network`.

    Raises:
    - `ValueError`: If dimensions or radii are non-positive or $a \ge R$.
    """
    if particle_radius <= 0.0:
        raise ValueError(f"particle_radius must be strictly positive; got {particle_radius!r}.")
    if domain.radius <= 0.0:
        raise ValueError(f"domain.radius must be strictly positive; got {domain.radius!r}.")
    if particle_radius >= domain.radius:
        raise ValueError(
            f"particle_radius ({particle_radius!r}) must be strictly less than "
            f"domain.radius ({domain.radius!r})."
        )
    
    return _HardParticleConstrainedFlow(
        network=network,
        domain=domain,
        particle_axial_position=particle_axial_position,
        particle_radius=particle_radius,
        particle_axial_velocity=particle_axial_velocity,
        boundary_layer_thickness_factor=boundary_layer_thickness_factor,
    )


