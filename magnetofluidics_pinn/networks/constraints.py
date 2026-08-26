"""Hard-constraint wrappers for the flow network.

A soft (loss-based) boundary condition only ever holds *approximately*
wherever it happens to be evaluated: the network is free to violate it
elsewhere by any amount the optimizer's current gradient step allows. For
most of Phase 1's boundary conditions that approximation is harmless. Two
conditions on the radial velocity `u_r` are the exception, and both are
enforced here structurally rather than through the loss:

- **No-slip at the wall**, $u_r(R, z) = 0$. The axisymmetric continuity
  equation integrates to $\frac{dQ}{dz} = 2 \pi R u_r(R, z)$ (Q being the volumetric
  flow rate), so *any* small, systematically-signed residual radial
  velocity at the wall accumulates, over the length of the channel, into a
  large violation of mass conservation far from the wall.
- **Regularity on the symmetry axis**, $u_r(0, z) = 0$. This is not a
  Poiseuille-specific condition: it holds for *any* smooth axisymmetric
  flow. The radial direction is a coordinate singularity at `r = 0` (which
  physical direction counts as "radial" depends on the azimuthal angle a
  point is approached from), so a nonzero `u_r` there would make the
  underlying 3D Cartesian velocity field discontinuous at the axis. See
  e.g. Batchelor (1967), "An Introduction to Fluid Dynamics", section 2.2,
  for the general regularity conditions at a cylindrical coordinate pole.

`apply_hard_wall_constraint` wraps a raw flow network so its radial
velocity output is *exactly* zero at both `r = 0` and `r = domain.radius`,
for every input, every parameter value, at every point in training — not
just approximately zero at a finite set of sampled points. This is the
standard hard-constraint technique for physics-informed networks [@lu2021physics].
Empirically (see the Phase 1 verification notebook),
adding the axis condition alongside the wall condition — rather than the
wall condition alone — roughly halved the relative L2 error against the
analytical solution again, on top of the improvement the wall constraint
alone already gave.
"""

from __future__ import annotations

import torch
from torch import nn

from magnetofluidics_pinn.types import Domain


class _HardWallConstrainedFlow(nn.Module):
    """Wraps a raw `(u_r, u_z, p)` network to enforce `u_r(0, z) = u_r(R, z) = 0`.

    Not part of the public API directly; constructed by
    [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint].
    """
    
    def __init__(self, raw_network: nn.Module, radius: float) -> None:
        super().__init__()
        self.raw_network = raw_network
        self.radius = radius
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        raw_output = self.raw_network(coordinates)
        normalized_radius = coordinates[:, 0:1] / self.radius
        # (r/R) * (1 - (r/R)^2) vanishes at r = 0 (axis regularity, linear
        # in r as required for a smooth axisymmetric field) and at r = R
        # (no-slip), for every z and every parameter value; u_z and p pass
        # through unchanged, still shaped entirely by the usual soft losses
        # (inlet, outlet, PDE residual).
        radial_shaping_factor = normalized_radius * (1.0 - normalized_radius ** 2)
        radial_velocity = radial_shaping_factor * raw_output[:, 0:1]
        return torch.cat([radial_velocity, raw_output[:, 1:2], raw_output[:, 2:3]], dim=1)


def apply_hard_wall_constraint(network: nn.Module, domain: Domain) -> nn.Module:
    """Wrap a flow network so `u_r` vanishes exactly at the axis and the wall.

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
      `(r, z)` coordinates to `(u_r, u_z, p)`.
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
    return _HardWallConstrainedFlow(network, domain.radius)
