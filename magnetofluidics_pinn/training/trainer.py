"""Training loop.

Runs the optimization procedure given a network, a domain, and a training
configuration, returning a new, trained network instance rather than
mutating the one passed in.
"""

from __future__ import annotations

import copy
from functools import partial

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import (
    inlet_velocity_condition,
    no_slip_condition,
    outlet_pressure_condition,
)
from magnetofluidics_pinn.config import FieldConfig, FluidConfig, TrainingConfig
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.collocation import (
    boundary_face_sizes,
    sample_collocation_points,
)
from magnetofluidics_pinn.training.losses import compose_loss
from magnetofluidics_pinn.types import Domain

# Boundary-condition terms are weighted more heavily than the interior PDE
# residual: there are typically far fewer boundary points than interior
# ones, so an unweighted sum lets the optimizer trade a small amount of
# boundary error for a marginally lower bulk residual (Raissi, Perdikaris, &
# Karniadakis, 2019).
_BOUNDARY_LOSS_WEIGHT = 10.0

# The dimensionless inlet centerline velocity. `FluidConfig.reference_velocity`
# is, by construction (see `scaling.compute_scales`), the physical velocity
# that maps to a dimensionless value of 1, so this is not an arbitrary
# constant but the natural consequence of the package's own normalization.
_DIMENSIONLESS_PEAK_INLET_VELOCITY = 1.0

# Incompressible (Stokes) pressure is only defined up to an additive
# constant; fixing the outlet reference pressure at zero is the standard
# gauge choice.
_DIMENSIONLESS_OUTLET_REFERENCE_PRESSURE = 0.0


def _pde_loss(
    network: nn.Module, coordinates: torch.Tensor, fluid_config: FluidConfig
) -> torch.Tensor:
    """Mean-squared Stokes residual over a batch of interior coordinates."""
    residual = stokes_residual(network, coordinates, fluid_config)
    return torch.mean(residual**2)


def _velocity_boundary_loss(
    network: nn.Module, coordinates: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """Mean-squared error between predicted and target boundary velocity."""
    predicted_velocity = network(coordinates)[:, :2]
    return torch.mean((predicted_velocity - target_velocity) ** 2)


def _pressure_boundary_loss(
    network: nn.Module, coordinates: torch.Tensor, target_pressure: torch.Tensor
) -> torch.Tensor:
    """Mean-squared error between predicted and target boundary pressure."""
    predicted_pressure = network(coordinates)[:, 2:3]
    return torch.mean((predicted_pressure - target_pressure) ** 2)


def train(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    field_config: FieldConfig,
    training_config: TrainingConfig,
) -> nn.Module:
    """Train a network to satisfy the flow PDE and boundary conditions.

    Args:
    - `network`: Network instance to train, e.g., built by
      [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp].
    - `domain`: Vessel geometry the network is trained on. Must already be
      nondimensionalized (see
      [`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]).
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `field_config`: Prescribed magnetic field configuration (used only if
      the training loop also tracks particle trajectories during training).
      Phase 1's Stokes flow residual has no dependency on the magnetic
      field, so this argument is currently unused; it is kept for signature
      stability across later, coupled phases.
    - `training_config`: Training hyperparameters (epochs, learning rate,
      sampling sizes, device, random seed).

    Returns:
    - A trained `torch.nn.Module` instance. The `network` argument passed in
      is never modified; a deep copy is trained and returned instead.

    Raises:
    - `ValueError`: If `training_config.n_epochs` is not strictly positive.
    """
    if training_config.n_epochs <= 0:
        raise ValueError("training_config.n_epochs must be strictly positive.")
    del field_config

    resolved_device = torch.device(
        training_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Never mutate the network handed in by the caller: train a private
    # copy and return it, per this module's stated contract.
    trained_network = copy.deepcopy(network).to(resolved_device)
    optimizer = torch.optim.Adam(trained_network.parameters(), lr=training_config.learning_rate)

    n_wall, n_inlet, n_outlet = boundary_face_sizes(training_config.n_boundary_points)

    for epoch in range(training_config.n_epochs):
        # Re-sampling every epoch, with a deterministically varying seed,
        # exposes the network to a fresh set of collocation points instead
        # of overfitting to a single fixed grid, while remaining fully
        # reproducible for a given base seed.
        collocation = sample_collocation_points(
            domain=domain,
            n_interior=training_config.n_interior_points,
            n_boundary=training_config.n_boundary_points,
            random_seed=training_config.random_seed + epoch,
        )
        # `.clone()` guarantees a tensor distinct from the one owned by
        # `collocation`, so setting `requires_grad_` here never mutates a
        # value that a pure sampling function returned.
        interior = collocation.interior.to(resolved_device).clone().requires_grad_(True)
        boundary = collocation.boundary.to(resolved_device)

        wall_points = boundary[:n_wall]
        inlet_points = boundary[n_wall : n_wall + n_inlet]
        outlet_points = boundary[n_wall + n_inlet : n_wall + n_inlet + n_outlet]

        wall_target = no_slip_condition(domain, wall_points)
        inlet_target = inlet_velocity_condition(
            domain, inlet_points, peak_velocity=_DIMENSIONLESS_PEAK_INLET_VELOCITY
        )
        outlet_target = outlet_pressure_condition(
            domain, outlet_points, reference_pressure=_DIMENSIONLESS_OUTLET_REFERENCE_PRESSURE
        )

        total_loss = compose_loss(
            terms={
                "pde": partial(_pde_loss, trained_network, interior, fluid_config),
                "wall": partial(_velocity_boundary_loss, trained_network, wall_points, wall_target),
                "inlet": partial(
                    _velocity_boundary_loss, trained_network, inlet_points, inlet_target
                ),
                "outlet": partial(
                    _pressure_boundary_loss, trained_network, outlet_points, outlet_target
                ),
            },
            weights={
                "pde": 1.0,
                "wall": _BOUNDARY_LOSS_WEIGHT,
                "inlet": _BOUNDARY_LOSS_WEIGHT,
                "outlet": _BOUNDARY_LOSS_WEIGHT,
            },
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return trained_network
