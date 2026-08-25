"""Hard-constraint wrappers for the flow network.

A soft (loss-based) boundary condition only ever holds *approximately*
wherever it happens to be evaluated: the network is free to violate it
elsewhere by any amount the optimizer's current gradient step allows. For
most of Phase 1's boundary conditions that approximation is harmless. The
no-slip wall condition is the exception: because the axisymmetric
continuity equation integrates to `dQ/dz = 2 pi R u_r(R, z)` (Q being the
volumetric flow rate), *any* small, systematically-signed residual radial
velocity at the wall accumulates, over the length of the channel, into a
large violation of mass conservation far from the wall — which is exactly
the failure mode a manufactured-solution audit traced the Phase 1
convergence gap back to.

`apply_hard_wall_constraint` removes that failure mode structurally: it
wraps a raw flow network so its radial-velocity output is *exactly* zero
at `r = domain.radius` for every input, every parameter value, at every
point in training — not just approximately zero at a finite set of sampled
wall points. This is the standard hard-constraint technique for
physics-informed networks (see e.g. Lu et al., 2021, "Physics-informed
neural networks with hard constraints for inverse design").
"""

from __future__ import annotations

import torch
from torch import nn

from magnetofluidics_pinn.types import Domain


class _HardWallConstrainedFlow(nn.Module):
    """Wraps a raw `(u_r, u_z, p)` network to enforce `u_r(R, z) = 0` exactly.

    Not part of the public API directly; constructed by
    [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint].
    """

    def __init__(self, raw_network: nn.Module, radius: float) -> None:
        super().__init__()
        self.raw_network = raw_network
        self.radius = radius

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        raw_output = self.raw_network(coordinates)
        radius_coordinate = coordinates[:, 0:1]
        # (1 - (r/R)^2) vanishes at r = R for every z and every parameter
        # value, so u_r is exactly zero at the wall by construction; u_z
        # and p pass through unchanged, still shaped entirely by the usual
        # soft losses (inlet, outlet, PDE residual).
        wall_distance_factor = 1.0 - (radius_coordinate / self.radius) ** 2
        radial_velocity = wall_distance_factor * raw_output[:, 0:1]
        return torch.cat([radial_velocity, raw_output[:, 1:2], raw_output[:, 2:3]], dim=1)


def apply_hard_wall_constraint(network: nn.Module, domain: Domain) -> nn.Module:
    """Wrap a flow network so the no-slip wall condition holds exactly.

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
      where the constraint is enforced.

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
