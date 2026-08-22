"""Physics residuals of the viscous flow equations.

Each residual function takes the network (mapping coordinates to velocity
and pressure) and a set of coordinates, and returns the pointwise PDE
residual through automatic differentiation. A residual of zero everywhere
means the network's output satisfies the governing equations exactly.
"""

from __future__ import annotations

from typing import Callable

import torch

from magnetofluidics_pinn.config import FluidConfig


def stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    fluid_config: FluidConfig,
) -> torch.Tensor:
    """Evaluate the incompressible Stokes-flow residual.

    Enforces continuity, `div(u) = 0`, and the steady momentum balance,
    `-grad(p) + laplacian(u) = 0`, in dimensionless form, at each coordinate.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, n_dims)`
      to a tensor of shape `(n_points, n_dims + 1)`, the last column being
      pressure and the preceding columns being velocity components.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients, sampled inside the domain.
    - `fluid_config`: Fluid configuration; must have `regime == "stokes"`.

    Returns:
    - Tensor of shape `(n_points, n_dims + 1)` with the continuity residual
      in the last column and the momentum residuals in the preceding
      columns.

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"stokes"`.
    """
    if fluid_config.regime != "stokes":
        raise ValueError(
            f"Expected a Stokes fluid configuration, got regime={fluid_config.regime!r}."
        )
    raise NotImplementedError("Implementation scheduled for Step 2.")


def navier_stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    fluid_config: FluidConfig,
) -> torch.Tensor:
    """Evaluate the incompressible, unsteady Navier-Stokes residual.

    Enforces continuity and the momentum balance including the convective and
    unsteady terms, in dimensionless form, at each coordinate.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, n_dims)`
      (including time) to a tensor of shape `(n_points, n_dims + 1)`, the
      last column being pressure and the preceding columns being velocity
      components.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients, including a time coordinate.
    - `fluid_config`: Fluid configuration; must have
      `regime == "navier_stokes"`.

    Returns:
    - Tensor of shape `(n_points, n_dims + 1)` with the continuity residual
      in the last column and the momentum residuals in the preceding
      columns.

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"navier_stokes"`.
    """
    if fluid_config.regime != "navier_stokes":
        raise ValueError(
            "Expected a Navier-Stokes fluid configuration, "
            f"got regime={fluid_config.regime!r}."
        )
    raise NotImplementedError("Implementation scheduled for a later roadmap phase.")
