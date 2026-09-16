"""Training loop.

Runs the optimization procedure given a network, a domain, and a training
configuration, returning a new, trained network instance rather than
mutating the one passed in, alongside a record of how each loss term
evolved during training.

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
import time
from dataclasses import dataclass

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import (
    inlet_velocity_condition,
    no_slip_condition,
    outlet_pressure_condition,
)
from magnetofluidics_pinn.config import MagneticFieldConfig, FluidConfig, TrainingConfig
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
# boundary error for a marginally lower bulk residual [@raissi2019physics].
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
    return torch.mean(residual ** 2)


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

    Centralizes what would otherwise be duplicated between the Adam loop
    and the L-BFGS refinement phase: sampling, splitting `boundary` into
    its three faces (via
    [`boundary_face_sizes`][magnetofluidics_pinn.sampling.collocation.boundary_face_sizes]),
    and evaluating each face's target condition.

    Args:
    - `domain`: Vessel geometry to sample from; must already be
      nondimensionalized.
    - `n_interior`: Number of interior collocation points.
    - `n_boundary`: Number of boundary collocation points.
    - `random_seed`: Seed passed straight through to
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points].
    - `device`: Device every tensor in the returned batch is placed on.

    Returns:
    - A [`_TrainingBatch`][magnetofluidics_pinn.training.trainer._TrainingBatch]
      with the interior points (`requires_grad_(True)`) and every boundary
      face paired with its target condition.
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
    inlet_points = collocation.boundary[n_wall: n_wall + n_inlet]
    outlet_points = collocation.boundary[n_wall + n_inlet: n_wall + n_inlet + n_outlet]
    
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


@dataclass(frozen=True)
class _LossComponents:
    """The four independent loss terms and their combined weighted total.

    Every field is still a scalar tensor attached to the autograd graph
    (nothing here has been `.item()`-ed yet), so `total` is safe to call
    `.backward()` on directly.

    Args:
    - `pde`: Mean-squared Stokes residual over the batch's interior points.
    - `wall`: Mean-squared no-slip error over the batch's wall points.
    - `inlet`: Mean-squared inlet-velocity error over the batch's inlet
      points.
    - `outlet`: Mean-squared outlet-pressure error over the batch's outlet
      points.
    - `total`: The weighted sum of the four terms above, via
      [`compose_loss`][magnetofluidics_pinn.training.losses.compose_loss].
    """
    
    pde: torch.Tensor
    wall: torch.Tensor
    inlet: torch.Tensor
    outlet: torch.Tensor
    total: torch.Tensor


def _evaluate_loss_components(
        network: nn.Module, batch: _TrainingBatch, fluid_config: FluidConfig
) -> _LossComponents:
    """Compute each loss term once and combine them into the weighted total.

    Computing each term exactly once here - rather than letting
    [`compose_loss`][magnetofluidics_pinn.training.losses.compose_loss] call
    a fresh callable per term - is what lets both training phases log or
    record each component's value without a second, wasted forward pass.

    Args:
    - `network`: Network to evaluate.
    - `batch`: Interior and boundary points, as built by
      [`_build_training_batch`][magnetofluidics_pinn.training.trainer._build_training_batch].
    - `fluid_config`: Fluid configuration selecting the flow regime.

    Returns:
    - A [`_LossComponents`][magnetofluidics_pinn.training.trainer._LossComponents]
      instance holding all four individual terms and their weighted total.
    """
    pde_value = _pde_loss(network, batch.interior, fluid_config)
    wall_value = _velocity_boundary_loss(network, batch.wall_points, batch.wall_target)
    inlet_value = _velocity_boundary_loss(network, batch.inlet_points, batch.inlet_target)
    outlet_value = _pressure_boundary_loss(network, batch.outlet_points, batch.outlet_target)
    total_loss = compose_loss(
        terms={
            "pde"   : lambda: pde_value,
            "wall"  : lambda: wall_value,
            "inlet" : lambda: inlet_value,
            "outlet": lambda: outlet_value,
        },
        weights={
            "pde"   : 1.0,
            "wall"  : _BOUNDARY_LOSS_WEIGHT,
            "inlet" : _BOUNDARY_LOSS_WEIGHT,
            "outlet": _BOUNDARY_LOSS_WEIGHT,
        },
    )
    return _LossComponents(pde=pde_value, wall=wall_value, inlet=inlet_value, outlet=outlet_value, total=total_loss)


@dataclass(frozen=True)
class LossHistory:
    """Loss values recorded at each step of one training phase.

    Every field is a plain tuple (not a list), so a returned
    [`TrainingHistory`][magnetofluidics_pinn.training.trainer.TrainingHistory]
    cannot be mutated by a caller after the fact - consistent with every
    other data-carrying type in this package being immutable.

    Args:
    - `step`: Step index (0-based) for each recorded entry - an Adam epoch
      index for the Adam phase, or an L-BFGS closure-call index for the
      L-BFGS phase (many per round, from that optimizer's internal line
      search) - one entry per value in the fields below.
    - `pde`, `wall`, `inlet`, `outlet`, `total`: The corresponding loss
      component at each `step`, in the same units
      [`_LossComponents`][magnetofluidics_pinn.training.trainer._LossComponents]
      reports them in.
    """
    
    step: tuple[int, ...]
    pde: tuple[float, ...]
    wall: tuple[float, ...]
    inlet: tuple[float, ...]
    outlet: tuple[float, ...]
    total: tuple[float, ...]


@dataclass(frozen=True)
class TrainingHistory:
    """Loss trajectories recorded during one `train()` call, for later plotting.

    Args:
    - `adam`: Per-epoch loss history for the Adam phase.
    - `lbfgs`: Per-closure-call loss history for the L-BFGS refinement
      phase; an empty
      [`LossHistory`][magnetofluidics_pinn.training.trainer.LossHistory]
      (every field an empty tuple) if
      `training_config.use_lbfgs_refinement` was `False`.
    """
    
    adam: LossHistory
    lbfgs: LossHistory
    
    def print_summary(self) -> None:
        """Prints a formatted summary table of initial, intermediate, and final loss values."""
        has_lbfgs = len(self.lbfgs.step) > 0
        fields = ("pde", "wall", "inlet", "outlet", "total")
        
        print("\n" + "=" * 78)
        print(
            f"{'Loss Component':<20} | {'Initial (Adam)':<14} | {'Post-Adam':<14} | {'Final Loss':<14} | {'Factor':<8}")
        print("-" * 78)
        
        for f in fields:
            initial = getattr(self.adam, f)[0]
            post_adam = getattr(self.adam, f)[-1]
            final = getattr(self.lbfgs, f)[-1] if has_lbfgs else post_adam
            reduction = initial / max(final, 1e-15)
            print(
                f"{f.upper():<20} | {initial:14.4e} | {post_adam:14.4e} | {final:14.4e} | {reduction:7.1f}x"
            )
        print("=" * 78 + "\n")


def _format_progress_line(
        prefix: str, total: float, pde: float, wall: float, inlet: float, outlet: float
) -> str:
    """Format one verbose progress line shared by both training phases.

    Args:
    - `prefix`: Leading label for the line, e.g. `"Epoch    12"` or
      `"L-BFGS round   2/  4"`.
    - `total`, `pde`, `wall`, `inlet`, `outlet`: The loss values to report,
      already reduced to plain Python floats (e.g. via `.item()`).

    Returns:
    - A single formatted line, e.g. `"Epoch    12 | Total Loss: 22.64972 |
      PDE Loss: 1.56034 | Wall Loss: 0.00312 | Inlet Loss: 0.00981 | Outlet
      Loss: 0.00004"`.
    """
    return (
        f"{prefix} | Total Loss: {total:.2e} | PDE Loss: {pde:.2e} | "
        f"Wall Loss: {wall:.2e} | Inlet Loss: {inlet:.2e} | Outlet Loss: {outlet:.2e}"
    )


def train(
        network: nn.Module,
        domain: Domain,
        fluid_config: FluidConfig,
        field_config: MagneticFieldConfig,
        training_config: TrainingConfig,
        verbose: bool = False,
        log_every: int = 100,
) -> tuple[nn.Module, TrainingHistory]:
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
    - `training_config`: Training hyperparameters — see
      [`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig] for the
      Adam-phase and L-BFGS-refinement-phase settings. Note that
      `use_lbfgs_refinement` defaults to `True` and runs regardless of
      `n_epochs`: a small `n_epochs` (e.g., for a quick interactive check)
      does not by itself produce a fast call — also pass
      `use_lbfgs_refinement=False` for that.
    - `verbose`: If `True`, print one progress line every `log_every`
      epochs during the Adam phase (always including the last), and one
      line per round during the L-BFGS phase, plus a one-line summary at
      the end. If `False` (the default), `train` prints nothing.
    - `log_every`: Epoch interval between printed Adam-phase progress
      lines when `verbose=True`; ignored otherwise.

    Returns:
    - A tuple `(trained_network, history)`. `trained_network` is a trained
      `torch.nn.Module` instance; the `network` argument passed in is never
      modified, a deep copy is trained and returned instead. `history` is a
      [`TrainingHistory`][magnetofluidics_pinn.training.trainer.TrainingHistory]
      recording how each loss term evolved, e.g. for plotting
      `history.adam.step` against `history.adam.total`.

    Raises:
    - `ValueError`: If `training_config.n_epochs` is not strictly positive,
      if `log_every` is not strictly positive, or if any enabled L-BFGS
      setting (`lbfgs_n_interior_points`, `lbfgs_n_boundary_points`,
      `lbfgs_rounds`, `lbfgs_iterations_per_round`) is not strictly
      positive.
    """
    if training_config.n_epochs <= 0:
        raise ValueError("training_config.n_epochs must be strictly positive.")
    if log_every <= 0:
        raise ValueError(f"log_every must be strictly positive; got {log_every!r}.")
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
    
    adam_start_time = time.perf_counter()
    adam_history = _run_adam_phase(
        trained_network, domain, fluid_config, training_config, resolved_device, verbose, log_every
    )
    if verbose:
        print(
            f"Adam phase finished in {time.perf_counter() - adam_start_time:.1f}s |"
            f"Final Total Loss: {adam_history.total[-1]:.2e}")
    
    lbfgs_start_time = time.perf_counter()
    lbfgs_history = LossHistory(step=(), pde=(), wall=(), inlet=(), outlet=(), total=())
    if training_config.use_lbfgs_refinement:
        lbfgs_history = _run_lbfgs_phase(
            trained_network, domain, fluid_config, training_config, resolved_device, verbose
        )
    
    if verbose:
        print(
            f"L-BFGS phase finished in {time.perf_counter() - lbfgs_start_time:.1f}s |"
            f"Final Total Loss: {lbfgs_history.total[-1]:.2e}")
        elapsed_seconds = time.perf_counter() - adam_start_time
        final_total = (lbfgs_history.total or adam_history.total)[-1]
        print(f"Training finished in {elapsed_seconds:.1f}s |"
              f"Final Total Loss: {final_total:.2e}")
    
    return trained_network, TrainingHistory(adam=adam_history, lbfgs=lbfgs_history)


def _run_adam_phase(
        trained_network: nn.Module,
        domain: Domain,
        fluid_config: FluidConfig,
        training_config: TrainingConfig,
        resolved_device: torch.device,
        verbose: bool,
        log_every: int,
) -> LossHistory:
    """First-order, stochastic-collocation phase: fast, global exploration.

    Mutates `trained_network`'s parameters in place via `optimizer.step()`
    (an unavoidable, intentional side effect of any training loop); the
    surrounding `train()` is what protects the caller's original network
    from that mutation, by operating on a copy.

    Args:
    - `trained_network`: The (already-copied) network to update in place.
    - `domain`, `fluid_config`, `training_config`: As passed to `train()`.
    - `resolved_device`: Device every collocation batch is placed on.
    - `verbose`: Whether to print per-epoch progress.
    - `log_every`: Epoch interval between printed progress lines.

    Returns:
    - A [`LossHistory`][magnetofluidics_pinn.training.trainer.LossHistory]
      with one entry per epoch.
    """
    optimizer = torch.optim.Adam(trained_network.parameters(), lr=training_config.learning_rate)
    steps: list[int] = []
    pde_values: list[float] = []
    wall_values: list[float] = []
    inlet_values: list[float] = []
    outlet_values: list[float] = []
    total_values: list[float] = []
    
    for epoch in range(training_config.n_epochs):
        # Re-sampling every epoch, with a deterministically-varying seed,
        # exposes the network to a fresh set of collocation points instead
        # of overfitting to a single fixed grid, while remaining fully
        # reproducible for a given base seed.
        batch = _build_training_batch(
            domain, training_config.n_interior_points, training_config.n_boundary_points,
            training_config.random_seed + epoch, resolved_device,
        )
        components = _evaluate_loss_components(trained_network, batch, fluid_config)
        
        optimizer.zero_grad()
        components.total.backward()
        if training_config.gradient_clip_norm is not None:
            # Nested second-order autograd (needed for the PDE residual)
            # occasionally produces a large gradient spike [@wang2021understanding];
            # clipping caps its effect on a single step without changing
            # what loss is being optimized.
            torch.nn.utils.clip_grad_norm_(
                trained_network.parameters(), training_config.gradient_clip_norm
            )
        optimizer.step()
        
        steps.append(epoch)
        pde_values.append(components.pde.item())
        wall_values.append(components.wall.item())
        inlet_values.append(components.inlet.item())
        outlet_values.append(components.outlet.item())
        total_values.append(components.total.item())
        
        if verbose and (epoch % log_every == 0 or epoch == training_config.n_epochs - 1):
            print(_format_progress_line(
                f"Epoch {epoch:5d}", total_values[-1], pde_values[-1], wall_values[-1],
                inlet_values[-1], outlet_values[-1],
            ))
    
    return LossHistory(
        step=tuple(steps), pde=tuple(pde_values), wall=tuple(wall_values),
        inlet=tuple(inlet_values), outlet=tuple(outlet_values), total=tuple(total_values),
    )


def _run_lbfgs_phase(
        trained_network: nn.Module,
        domain: Domain,
        fluid_config: FluidConfig,
        training_config: TrainingConfig,
        resolved_device: torch.device,
        verbose: bool,
) -> LossHistory:
    """Second-order, fixed-batch refinement phase, run after Adam.

    L-BFGS is a full-batch quasi-Newton method: its curvature estimate is
    only valid across calls that see the *same* loss surface, so — unlike
    the Adam phase — the collocation batch is sampled once and held fixed
    for every internal iteration of every round.

    Args:
    - `trained_network`: The (already Adam-trained) network to refine in
      place.
    - `domain`, `fluid_config`, `training_config`: As passed to `train()`.
    - `resolved_device`: Device the fixed collocation batch is placed on.
    - `verbose`: Whether to print one progress line per round.

    Returns:
    - A [`LossHistory`][magnetofluidics_pinn.training.trainer.LossHistory]
      with one entry per `closure` call - typically several per round,
      from `torch.optim.LBFGS`'s internal line search - which is a finer
      granularity than the once-per-round summary printed when `verbose`.
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
    
    steps: list[int] = []
    pde_values: list[float] = []
    wall_values: list[float] = []
    inlet_values: list[float] = []
    outlet_values: list[float] = []
    total_values: list[float] = []
    
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        components = _evaluate_loss_components(trained_network, batch, fluid_config)
        components.total.backward()
        
        steps.append(len(steps))
        pde_values.append(components.pde.item())
        wall_values.append(components.wall.item())
        inlet_values.append(components.inlet.item())
        outlet_values.append(components.outlet.item())
        total_values.append(components.total.item())
        return components.total
    
    round_width = len(str(training_config.lbfgs_rounds))
    for round_index in range(training_config.lbfgs_rounds):
        optimizer.step(closure)
        if verbose:
            prefix = f"L-BFGS round {round_index + 1:{round_width}d}/{training_config.lbfgs_rounds}"
            print(_format_progress_line(
                prefix, total_values[-1], pde_values[-1], wall_values[-1],
                inlet_values[-1], outlet_values[-1],
            ))
    
    return LossHistory(
        step=tuple(steps), pde=tuple(pde_values), wall=tuple(wall_values),
        inlet=tuple(inlet_values), outlet=tuple(outlet_values), total=tuple(total_values),
    )
