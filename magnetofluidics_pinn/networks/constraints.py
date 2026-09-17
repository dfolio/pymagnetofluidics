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
    """Wraps a raw `(u_r, u_z, p)` network to enforce axis and wall regularity.

    Not part of the public API directly; constructed by
    [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint].
    """

    def __init__(self, raw_network: nn.Module, radius: float) -> None:
        super().__init__()
        self.raw_network = raw_network
        self.radius = radius
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        normalized_radius = coordinates[:, 0:1] / self.radius
        axial_coordinate = coordinates[:, 1:2]
        
        squared_normalized_radius = normalized_radius ** 2
        raw_output = self.raw_network(torch.cat([squared_normalized_radius, axial_coordinate], dim=1))
        
        radial_shaping_factor = normalized_radius * (1.0 - squared_normalized_radius)
        radial_velocity = radial_shaping_factor * raw_output[:, 0:1]
        
        # CHANGED: u_z(R, z) = 0 (tangential no-slip) was previously left
        # to the soft wall-boundary loss in `training.trainer`. Reusing
        # the (1 - (r/R)^2) wall-vanishing factor already computed above -
        # even in r, so the axis regularity inherited from the (r/R)^2
        # input reparametrization is preserved, and exactly zero at r=R -
        # hard-enforces it structurally instead, for every input and every
        # parameter value. Pressure is left untouched: it has no Dirichlet
        # condition at the wall, only the axis regularity already implied
        # by evenness in r.
        wall_vanishing_factor = 1.0 - squared_normalized_radius
        axial_velocity = wall_vanishing_factor * raw_output[:, 1:2]
        
        return torch.cat([radial_velocity, axial_velocity, raw_output[:, 2:3]], dim=1)


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
    return _HardWallConstrainedFlow(network, domain.radius)

