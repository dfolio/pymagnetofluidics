"""Training loop.

Runs the optimization procedure given a network, a domain, and a training
configuration, returning a new, trained network instance rather than
mutating the one passed in.

The default recipe is Adam (global exploration) followed by L-BFGS (local
refinement) — see `TrainingConfig`'s docstring for the diagnosis that
motivated adding the L-BFGS phase: `stokes_residual` and the
boundary-condition targets are exact (verified against the analytical
Poiseuille solution via the method of manufactured solutions), but Adam
alone plateaus with the PDE residual still orders of magnitude too large to
match the analytical profile — boundary terms converging much faster than
the interior residual is a documented PINN training characteristic
[@krishnapriyan2021characterizing; @wang2021understanding], not evidence
that the residual or boundary-condition formulas themselves are wrong.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import (
    inlet_velocity_condition,
    no_slip_condition,
    outlet_pressure_condition,
)
from magnetofluidics_pinn.config import FieldConfig, FluidConfig, TrainingConfig
from magnetofluidics_pinn.device_utils import resolve_device
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


# Add:
def _axis_regularity_loss(
        network: nn.Module,
        axis_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Compute a loss term that penalizes nonzero radial velocity and
    nonzero axial gradient of axial velocity at the axis of symmetry (r=0).
    """
    coordinates = axis_coordinates.clone().requires_grad_(True)
    output = network(coordinates)
    
    radial_velocity = output[:, 0:1]
    axial_velocity = output[:, 1:2]
    (axial_gradient,) = torch.autograd.grad(
        outputs=axial_velocity,
        inputs=coordinates,
        grad_outputs=torch.ones_like(axial_velocity),
        create_graph=True,
    )
    return torch.mean(radial_velocity ** 2) + torch.mean(axial_gradient[:, 0:1] ** 2)


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


@dataclass(frozen=True)
class _TrainingBatch:
    """A resolved, device-placed set of interior and boundary sub-batches.

    Args:
    - `interior`: Interior collocation points, requiring gradients.
    - `wall_points`, `wall_target`: Wall coordinates and their target
      (zero) velocity.
    - `inlet_points`, `inlet_target`: Inlet coordinates and their target
      (parabolic) velocity.
    - `outlet_points`, `outlet_target`: Outlet coordinates and their target
      (reference) pressure.
    """

    interior: torch.Tensor
    wall_points: torch.Tensor
    wall_target: torch.Tensor
    inlet_points: torch.Tensor
    inlet_target: torch.Tensor
    outlet_points: torch.Tensor
    outlet_target: torch.Tensor


def _build_training_batch(
    domain: Domain, n_interior: int, n_boundary: int, random_seed: int, device: torch.device
) -> _TrainingBatch:
    """Sample and label one interior/boundary batch, ready for loss evaluation.

    Centralizes what was previously duplicated between the Adam loop and
    the L-BFGS refinement phase: sampling, splitting `boundary` into its
    three faces (via
    [`boundary_face_sizes`][magnetofluidics_pinn.sampling.collocation.boundary_face_sizes]),
    and evaluating each face's target condition.
    """
    collocation = sample_collocation_points(
        domain=domain, n_interior=n_interior, n_boundary=n_boundary,
        random_seed=random_seed, device=device,
    )
    # `.clone()` guarantees a tensor distinct from the one `collocation`
    # owns, so setting `requires_grad_` here never mutates a value that a
    # pure sampling function returned.
    interior = collocation.interior.clone().requires_grad_(True)

    n_wall, n_inlet, n_outlet = boundary_face_sizes(n_boundary)
    wall_points = collocation.boundary[:n_wall]
    inlet_points = collocation.boundary[n_wall : n_wall + n_inlet]
    outlet_points = collocation.boundary[n_wall + n_inlet : n_wall + n_inlet + n_outlet]

    return _TrainingBatch(
        interior=interior,
        wall_points=wall_points,
        wall_target=no_slip_condition(domain, wall_points),
        inlet_points=inlet_points,
        inlet_target=inlet_velocity_condition(
            domain, inlet_points, peak_velocity=_DIMENSIONLESS_PEAK_INLET_VELOCITY
        ),
        outlet_points=outlet_points,
        outlet_target=outlet_pressure_condition(
            domain, outlet_points, reference_pressure=_DIMENSIONLESS_OUTLET_REFERENCE_PRESSURE
        ),
    )


def _composed_loss(network: nn.Module, batch: _TrainingBatch, fluid_config: FluidConfig) -> torch.Tensor:
    """Assemble the same weighted PDE + boundary loss used by both training phases."""
    return compose_loss(
        terms={
            "pde": partial(_pde_loss, network, batch.interior, fluid_config),
            "wall": partial(_velocity_boundary_loss, network, batch.wall_points, batch.wall_target),
            "inlet": partial(_velocity_boundary_loss, network, batch.inlet_points, batch.inlet_target),
            "outlet": partial(_pressure_boundary_loss, network, batch.outlet_points, batch.outlet_target),
        },
        weights={
            "pde": 1.0,
            "wall": _BOUNDARY_LOSS_WEIGHT,
            "inlet": _BOUNDARY_LOSS_WEIGHT,
            "outlet": _BOUNDARY_LOSS_WEIGHT,
        },
    )


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
      [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp], optionally
      wrapped with
      [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint]
      (recommended: see that function's docstring for why the no-slip wall
      condition specifically benefits from a hard constraint).
    - `domain`: Vessel geometry the network is trained on. Must already be
      nondimensionalized (see
      [`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]).
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `field_config`: Prescribed magnetic field configuration (used only if
      the training loop also tracks particle trajectories during training).
      Phase 1's Stokes flow residual has no dependency on the magnetic
      field, so this argument is currently unused; it is kept for signature
      stability across later, coupled phases.
    - `training_config`:Training hyperparameters — see
      [`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig] for the
      Adam-phase and L-BFGS-refinement-phase settings. Note that
      `use_lbfgs_refinement` defaults to `True` and runs regardless of
      `n_epochs`: a small `n_epochs` (e.g., for a quick interactive check)
      does not by itself produce a fast call — also pass
      `use_lbfgs_refinement=False` for that.

    Returns:
    - A trained `torch.nn.Module` instance. The `network` argument passed in
      is never modified; a deep copy is trained and returned instead.

    Raises:
    - `ValueError`: If `training_config.n_epochs` is not strictly positive,
      or if any enabled L-BFGS setting (`lbfgs_n_interior_points`,
      `lbfgs_n_boundary_points`, `lbfgs_rounds`, `lbfgs_iterations_per_round`)
      is not strictly positive.
    """
    if training_config.n_epochs <= 0:
        raise ValueError("training_config.n_epochs must be strictly positive.")
    if training_config.use_lbfgs_refinement:
        if training_config.lbfgs_n_interior_points <= 0 or training_config.lbfgs_n_boundary_points <= 0:
            raise ValueError(
                "training_config.lbfgs_n_interior_points and lbfgs_n_boundary_points "
                "must be strictly positive when use_lbfgs_refinement is True."
            )
        if training_config.lbfgs_rounds <= 0 or training_config.lbfgs_iterations_per_round <= 0:
            raise ValueError(
                "training_config.lbfgs_rounds and lbfgs_iterations_per_round "
                "must be strictly positive when use_lbfgs_refinement is True."
            )
    del field_config

    resolved_device = resolve_device(training_config.device)

    # Never mutate the network handed in by the caller: train a private
    # copy and return it, per this module's stated contract.
    trained_network = copy.deepcopy(network).to(resolved_device)

    _run_adam_phase(trained_network, domain, fluid_config, training_config, resolved_device)
    if training_config.use_lbfgs_refinement:
        _run_lbfgs_phase(trained_network, domain, fluid_config, training_config, resolved_device)

    return trained_network


def _run_adam_phase(
    trained_network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    resolved_device: torch.device,
) -> None:
    """First-order, stochastic-collocation phase: fast, global exploration.

    Mutates `trained_network`'s parameters in place via `optimizer.step()`
    (an unavoidable, intentional side effect of any training loop); the
    surrounding `train()` is what protects the caller's original network
    from that mutation, by operating on a copy.
    """
    optimizer = torch.optim.Adam(trained_network.parameters(), lr=training_config.learning_rate)

    for epoch in range(training_config.n_epochs):
        # Re-sampling every epoch, with a deterministically-varying seed,
        # exposes the network to a fresh set of collocation points instead
        # of overfitting to a single fixed grid, while remaining fully
        # reproducible for a given base seed.
        batch = _build_training_batch(
            domain, training_config.n_interior_points, training_config.n_boundary_points,
            training_config.random_seed + epoch, resolved_device,
        )
        total_loss = _composed_loss(trained_network, batch, fluid_config)

        optimizer.zero_grad()
        total_loss.backward()
        if training_config.gradient_clip_norm is not None:
            # Nested second-order autograd (needed for the PDE residual)
            # occasionally produces a large gradient spike; clipping caps
            # its effect on a single step without changing what loss is
            # being optimized.
            torch.nn.utils.clip_grad_norm_(
                trained_network.parameters(), training_config.gradient_clip_norm
            )
        optimizer.step()


def _run_lbfgs_phase(
    trained_network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    resolved_device: torch.device,
) -> None:
    """Second-order, fixed-batch refinement phase, run after Adam.

    L-BFGS is a full-batch quasi-Newton method: its curvature estimate is
    only valid across calls that see the *same* loss surface, so — unlike
    the Adam phase — the collocation batch is sampled once and held fixed
    for every internal iteration of every round.
    """
    batch = _build_training_batch(
        domain, training_config.lbfgs_n_interior_points, training_config.lbfgs_n_boundary_points,
        training_config.random_seed - 1, resolved_device,
    )
    optimizer = torch.optim.LBFGS(
        trained_network.parameters(),
        lr=1.0,
        max_iter=training_config.lbfgs_iterations_per_round,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _composed_loss(trained_network, batch, fluid_config)
        loss.backward()
        return loss

    for _ in range(training_config.lbfgs_rounds):
        optimizer.step(closure)
