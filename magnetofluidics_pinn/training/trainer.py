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

**Decomposed, independently-weighted interior loss.** The interior
residual returned by `stokes_residual` bundles three physically distinct
equations (r-momentum, z-momentum, continuity) into one tensor. Each term
is reduced and weighted independently (`TrainingConfig.momentum_loss_weight`,
`.continuity_loss_weight`), alongside two physics-fidelity terms: a soft
penalty discouraging a negative predicted axial velocity
(`.positivity_loss_weight`) and an explicit global flow-rate conservation
constraint (`.conservation_loss_weight`, see `physics.conservation`)
[@hu2025pecann; @sigalingging2026massconserving].

**CHANGED (in v0.1.6)— single `train()` entry point for both the particle-free and
two-way-coupled problems.** This module previously exposed two, largely
parallel, public functions: `train()` for the ordinary (particle-free)
Stokes/Navier-Stokes problem, and `train_around_particle()` for the
two-way-coupled problem solved around an embedded
[`SphericalParticle`][magnetofluidics_pinn.types.SphericalParticle]. The
two entry points shared essentially everything — batch construction, loss
weighting, the Adam/L-BFGS phase structure, history bookkeeping — and
differed only in which collocation sampler was called and whether one
extra (particle-surface) loss term was included, which is exactly the
condition an optional argument is meant to express rather than a second
function. `train()` now accepts an optional
[`CouplingConfig`][magnetofluidics_pinn.config.CouplingConfig]: omitted (the
default), it reproduces every previous `train()` call's behavior exactly;
supplied, it reproduces every previous `train_around_particle()` call's
behavior exactly, without a second, independently-maintained copy of the
training loop. `verbose` and `log_every` move onto
[`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig] for the same
reason: both were previously identical, separately-repeated keyword
arguments on both entry points, one config field instead of two now fully
determines both what is trained and how it is reported.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import (
    inlet_velocity_condition,
    no_slip_condition,
    outlet_pressure_condition,
    rigid_body_velocity_condition,
)
from magnetofluidics_pinn.config import (DEFAULT_BOUNDARY_LOSS_WEIGHT, FluidConfig, MagneticFieldConfig, ParticleConfig,
                                         TrainingConfig,
                                         TwoWayCouplingConfig)  # NEW — promoted from this module's own _BOUNDARY_LOSS_WEIGHT; see config.py.; NEW
from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.physics.conservation import axial_flow_rate, poiseuille_reference_flow_rate
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.collocation import (
    boundary_face_sizes,
    sample_collocation_points,
    sample_collocation_points_with_particle,
)
from magnetofluidics_pinn.training.losses import compose_loss
from magnetofluidics_pinn.types import Domain, ParticleState

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

# NEW: ordered, single source of truth for every scalar loss component
# tracked during training (excluding "total", handled separately since it
# is the weighted combination of these, not an independent term). Reused
# by `_record_loss_components`, `LossHistory`'s field set,
# `TrainingHistory.print_summary`, and `visualization.plotting.
# plot_training_history`, so a future new term only needs to be added here.
LOSS_COMPONENT_NAMES: tuple[str, ...] = (
    "momentum_r", "momentum_z", "continuity", "wall", "inlet", "outlet", "positivity", "conservation",
)


def _component_names(coupling_config: TwoWayCouplingConfig | None) -> tuple[str, ...]:
    """Full ordered set of scalar loss-component names for one `train()` call.

    Single point of truth for the one difference the optional
    two-way-coupling term makes to every loss-tracking container in this
    module (`_LossComponents`, `LossHistory`, and every accumulator built
    from them): `LOSS_COMPONENT_NAMES` plus `"particle"`, appended last,
    when `coupling_config` is not `None`; `LOSS_COMPONENT_NAMES` unchanged
    otherwise. Every other function in this module derives its component
    set from this one instead of re-deriving the same conditional
    independently.

    Args:
    - `coupling_config`: The `TwoWayCouplingConfig` passed to `train()` for this
      call, or `None` for the uncoupled problem.

    Returns:
    - `LOSS_COMPONENT_NAMES`, with `"particle"` appended when
      `coupling_config is not None`.
    """
    return LOSS_COMPONENT_NAMES + (("particle",) if coupling_config is not None else ())


def _stokes_residual_losses(
        network: nn.Module, coordinates: torch.Tensor, fluid_config: FluidConfig,
        residual_form: Literal["standard", "r_weighted"] = "standard",  # NEW
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-squared Stokes residual, decomposed by governing equation.

    Args:
    - `network`: Flow network mapping `(r, z)` coordinates to
      `(u_r, u_z, p)`.
    - `coordinates`: Interior collocation points, requiring gradients.
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `residual_form`: NEW. `"standard"` (default) evaluates the equations
      as classically written, and requires `r > 0` everywhere (unchanged
      from every prior release). `"r_weighted"` evaluates the same
      equations pre-multiplied by $r$ or $r^2$ to remove the axis
      singularity analytically, and additionally accepts `r = 0`. See this
      module's docstring for the full derivation and the trade-off between
      the two.
      
    Returns:
    - A tuple `(momentum_r_loss, momentum_z_loss, continuity_loss)`, each a
      scalar tensor: the mean-squared residual of its own governing
      equation, from the three columns
      [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
      returns.
    """
    residual = stokes_residual(network, coordinates, fluid_config, residual_form=residual_form)
    momentum_r_loss = torch.mean(residual[:, 0:1].square())
    momentum_z_loss = torch.mean(residual[:, 1:2].square())
    continuity_loss = torch.mean(residual[:, 2:3].square())
    return momentum_r_loss, momentum_z_loss, continuity_loss


def _positivity_loss(network: nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    r"""Soft, one-sided penalty discouraging a negative predicted axial velocity.

    For Phase 1's straight channel, driven only by a positive inlet
    profile and a reference outlet pressure (no swirl, no adverse pressure
    reversal), $u_z \geq 0$ everywhere is a physical requirement; nothing
    in `networks.apply_hard_wall_constraint` structurally guarantees it.
    Kept as a *soft* penalty rather than a hard reparametrization, since
    Phase 3's bifurcated/Navier-Stokes regime will legitimately produce
    local recirculation ($u_z < 0$), which a hard sign constraint would
    have to be undone to support.

    Args:
    - `network`: Flow network mapping `(r, z)` coordinates to
      `(u_r, u_z, p)`.
    - `coordinates`: Interior collocation points (no gradient requirement:
      this term needs the network's output only, not its derivatives).

    Returns:
    - Scalar tensor: $\mathbb{E}\left[\mathrm{ReLU}(-u_z)^2\right]$ over
      `coordinates`; zero wherever every predicted $u_z$ is non-negative.
    """
    predicted_axial_velocity = network(coordinates)[:, 1:2]
    return torch.mean(torch.relu(-predicted_axial_velocity).square())


def _conservation_loss(
        network: nn.Module,
        domain: Domain,
        peak_velocity: float,
        n_stations: int,
        n_quadrature_points: int,
        device: torch.device,
) -> torch.Tensor:
    r"""Mean-squared deviation of the network's flow rate from its analytic reference.

    Complements the local, pointwise continuity term from
    `_stokes_residual_losses` with an explicit, global constraint on the
    macroscopic quantity mass conservation is actually about: penalizes
    $Q(z)$'s deviation from the known Hagen-Poiseuille reference flow rate
    at a set of axial stations spanning the full channel (see
    `physics.conservation`).

    Args:
    - `network`: Flow network mapping `(r, z)` coordinates to
      `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `peak_velocity`: Dimensionless inlet centerline velocity, matching
      the value `inlet_velocity_condition` was given (see
      `_DIMENSIONLESS_PEAK_INLET_VELOCITY`), so the analytic reference flow
      rate is consistent with the boundary condition actually imposed.
    - `n_stations`: Number of evenly-spaced axial stations spanning
      `[0, domain.length]` at which $Q(z)$ is evaluated.
    - `n_quadrature_points`: Radial trapezoidal-quadrature resolution per
      station; forwarded to `axial_flow_rate`.
    - `device`: Device the axial-station grid is created on; matches the
      already-resolved device the rest of the training batch lives on.

    Returns:
    - Scalar tensor: mean-squared deviation of the predicted $Q(z)$ from
      $Q_\mathrm{ref}$, over the requested stations.
    """
    axial_positions = torch.linspace(0.0, domain.length, n_stations, device=device)
    predicted_flow_rate = axial_flow_rate(network, domain.radius, axial_positions, n_quadrature_points)
    reference_flow_rate = poiseuille_reference_flow_rate(domain.radius, peak_velocity)
    return torch.mean((predicted_flow_rate - reference_flow_rate).square())


def _velocity_boundary_loss(
        network: nn.Module, coordinates: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """Mean-squared error between predicted and target boundary velocity."""
    predicted_velocity = network(coordinates)[:, :2]
    return torch.mean((predicted_velocity - target_velocity).square())


def _pressure_boundary_loss(
        network: nn.Module, coordinates: torch.Tensor, target_pressure: torch.Tensor
) -> torch.Tensor:
    """Mean-squared error between predicted and target boundary pressure."""
    predicted_pressure = network(coordinates)[:, 2:3]
    return torch.mean((predicted_pressure - target_pressure).square())


def _particle_conservation_loss(
        network: nn.Module,
        domain: Domain,
        particle_state: ParticleState,
        particle_config: ParticleConfig,
        peak_velocity: float,
        n_stations: int,
        n_quadrature_points: int,
        device: torch.device,
) -> torch.Tensor:
    r"""Mass-conservation loss, restricted to full-bore cross-sections away from particle.
 
    `_conservation_loss` spaces its stations uniformly across the *entire*
    `[0, domain.length]` span; some of those would fall inside the
    particle's axial extent `[axial_position - radius, axial_position +
    radius]`, where part of the cross-section is solid rather than fluid,
    and
    [`physics.conservation.axial_flow_rate`][magnetofluidics_pinn.physics.conservation.axial_flow_rate]'s
    full-bore quadrature does not know to exclude that solid core. Away
    from the particle, however, the check is not just still valid but
    *stronger* than in the unobstructed case: an incompressible fluid
    cannot pass through the rigid sphere, so the same volumetric flow rate
    the inlet condition prescribes must reappear, unchanged, at every
    upstream *and* downstream cross-section.
 
    Args:
    - `network`: Flow network mapping `(r, z)` coordinates to
      `(u_r, u_z, p)`.
    - `domain`, `particle`: As elsewhere in this module.
    - `peak_velocity`: Dimensionless inlet centerline velocity.
    - `n_stations`: Number of stations, split evenly between the upstream
      segment `[0, axial_position - radius]` and the downstream segment
      `[axial_position + radius, domain.length]`.
    - `n_quadrature_points`: Radial quadrature resolution per station,
      forwarded to `axial_flow_rate`.
    - `device`: Device the axial-station grids are created on.
 
    Returns:
    - Scalar tensor: mean-squared deviation of the predicted $Q(z)$ from
      $Q_\mathrm{ref}$, pooled over the upstream and downstream stations.
    """
    n_upstream = max(n_stations // 2, 1)
    n_downstream = max(n_stations - n_upstream, 1)
    upstream_positions = torch.linspace(
        0.0, particle_state.axial_position - particle_config.radius, n_upstream + 2, device=device
    )[1:-1]
    downstream_positions = torch.linspace(
        particle_state.axial_position + particle_config.radius, domain.length, n_downstream + 2, device=device
    )[1:-1]
    axial_positions = torch.cat([upstream_positions, downstream_positions])
    
    predicted_flow_rate = axial_flow_rate(network, domain.radius, axial_positions, n_quadrature_points)
    reference_flow_rate = poiseuille_reference_flow_rate(domain.radius, peak_velocity)
    return torch.mean((predicted_flow_rate - reference_flow_rate).square())


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
    - `particle_points`, `particle_target`: CHANGED — was a separate
      `_ObjectTrainingBatch` subclass-like dataclass; folded in here as
      an optional pair instead, mirroring how
      [`CollocationPoints.particle_surface`][magnetofluidics_pinn.types.CollocationPoints]
      itself is optional. `None` for a batch built without an
      `ObjectConfig`; both populated together otherwise.
    """
    
    interior: torch.Tensor
    wall_points: torch.Tensor
    wall_target: torch.Tensor
    inlet_points: torch.Tensor
    inlet_target: torch.Tensor
    outlet_points: torch.Tensor
    outlet_target: torch.Tensor
    particle_points: torch.Tensor | None = None  # NEW
    particle_target: torch.Tensor | None = None  # NEW


def _build_training_batch(
        domain: Domain,
        coupling_config: TwoWayCouplingConfig | None,  # NEW
        n_interior: int,
        n_boundary: int,
        random_seed: int,
        axis_clearance_fraction: float | None,
        device: torch.device,
) -> _TrainingBatch:
    """Sample and label one interior/boundary batch, ready for loss evaluation.

    Args:
    - `domain`: Vessel geometry to sample from; must already be
      nondimensionalized.
    - `coupling_config`: `None` for the ordinary (particle-free) problem;
      otherwise the two-way-coupling parameters (embedded particle, its
      translational velocity, and its collocation/loss settings) for the
      current call.
    - `n_interior`: Number of interior collocation points.
    - `n_boundary`: Number of boundary collocation points.
    - `random_seed`: Seed passed straight through to
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points].
    - `axis_clearance_fraction`: NEW. Passed straight through to
      [`sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points];
      `None` reproduces every prior release's sampling exactly. Sourced from
      `TrainingConfig.axis_clearance_fraction`, whose own docstring covers
      the numerical caveat around pairing it with `residual_form`.
    - `device`: Device every tensor in the returned batch is placed on.

    Returns:
    - A [`_TrainingBatch`][magnetofluidics_pinn.training.trainer._TrainingBatch]
      with the interior points (`requires_grad_(True)`) and every boundary
      face paired with its target condition; `particle_points` and
      `particle_target` populated only when `particle_config` is not
      `None`.
    """
    
    if coupling_config is None:
        collocation = sample_collocation_points(
            domain=domain, n_interior=n_interior, n_boundary=n_boundary,
            random_seed=random_seed, axis_clearance_fraction=axis_clearance_fraction,
            device=device,
        )
        particle_points, particle_target = None, None
    else:
        collocation = sample_collocation_points_with_particle(
            domain=domain,
            particle_state=coupling_config.particle_state.to(device=device),
            particle_config=coupling_config.particle_config, n_interior=n_interior, n_boundary=n_boundary,
            n_surface_points=coupling_config.n_surface_points, random_seed=random_seed,
            axis_clearance_fraction=axis_clearance_fraction, device=device,
        )
        particle_points = collocation.particle_surface
        particle_target = rigid_body_velocity_condition(
            particle_points, (0.0, coupling_config.axial_particle_velocity)
        )
    
    # `.clone()` guarantees a tensor distinct from the one `collocation`
    # owns, so setting `requires_grad_` here never mutates a value that a
    # pure sampling function returned.
    # Freeze the point set to maintain a deterministic loss landscape for Hessian approximation
    interior = collocation.interior.detach().clone().requires_grad_(True)
    
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
        particle_points=particle_points,
        particle_target=particle_target,
    )


@dataclass(frozen=True)
class _LossComponents:
    """The independent loss terms and their combined weighted total.

    Every field is still a scalar tensor attached to the autograd graph
    (nothing here has been `.item()`-ed yet), so `total` is safe to call
    `.backward()` on directly.

    Args:
    - `momentum_r`, `momentum_z`: Mean-squared r- and z-momentum residual
      over the batch's interior points.
    - `continuity`: Mean-squared continuity (mass-conservation) residual
      over the batch's interior points.
    - `wall`: Mean-squared no-slip error over the batch's wall points.
    - `inlet`: Mean-squared inlet-velocity error over the batch's inlet
      points.
    - `outlet`: Mean-squared outlet-pressure error over the batch's outlet
      points.
    - `positivity`: Soft penalty for negative predicted axial velocity.
    - `conservation`: Mean-squared deviation of the global flow rate $Q(z)$
      from its analytic reference.
    - `total`: The weighted sum of the eight terms above, via
      [`compose_loss`][magnetofluidics_pinn.training.losses.compose_loss].
    - `particle`: Mean-squared error between the network's predicted velocity on the
      particle's surface and its prescribed rigid-body velocity; `None`
      unless this call to `train()` was given an `ParticleConfig`.
    """
    
    momentum_r: torch.Tensor
    momentum_z: torch.Tensor
    continuity: torch.Tensor
    wall: torch.Tensor
    inlet: torch.Tensor
    outlet: torch.Tensor
    positivity: torch.Tensor
    conservation: torch.Tensor
    total: torch.Tensor
    particle: torch.Tensor | None


def _evaluate_loss_components(
        network: nn.Module,
        batch: _TrainingBatch,
        domain: Domain,
        fluid_config: FluidConfig,
        training_config: TrainingConfig,
        coupling_config: TwoWayCouplingConfig | None,  # NEW
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
    - `domain`: Already-nondimensionalized vessel geometry; needed by the
      global conservation term (`_conservation_loss`), which evaluates the
      flow rate at stations spanning `domain.length`.
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `training_config`: Supplies every per-term weight
      (`momentum_loss_weight`, `continuity_loss_weight`,
      `positivity_loss_weight`, `conservation_loss_weight`) and the
      conservation term's quadrature resolution.
    - `particle_config`: `None` for the ordinary (particle-free) problem;
      otherwise supplies the embedded sphere (for
      [`_particle_conservation_loss`][magnetofluidics_pinn.training.trainer._particle_conservation_loss]) and `particle_loss_weight`.

    Returns:
    - A [`_LossComponents`][magnetofluidics_pinn.training.trainer._LossComponents]
      instance holding every individual term and their weighted total,
      with `particle` populated only when `particle_config` is not `None`.
    """
    momentum_r_value, momentum_z_value, continuity_value = _stokes_residual_losses(
        network, batch.interior, fluid_config, residual_form=training_config.residual_form
    )
    wall_value = _velocity_boundary_loss(network, batch.wall_points, batch.wall_target)
    inlet_value = _velocity_boundary_loss(network, batch.inlet_points, batch.inlet_target)
    outlet_value = _pressure_boundary_loss(network, batch.outlet_points, batch.outlet_target)
    positivity_value = _positivity_loss(network, batch.interior)
    
    terms: dict[str, Callable[[], torch.Tensor]] = {
        "momentum_r": lambda: momentum_r_value,
        "momentum_z": lambda: momentum_z_value,
        "continuity": lambda: continuity_value,
        "wall"      : lambda: wall_value,
        "inlet"     : lambda: inlet_value,
        "outlet"    : lambda: outlet_value,
        "positivity": lambda: positivity_value,
    }
    weights: dict[str, float] = {
        "momentum_r": training_config.momentum_loss_weight,
        "momentum_z": training_config.momentum_loss_weight,
        "continuity": training_config.continuity_loss_weight,
        "wall"      : DEFAULT_BOUNDARY_LOSS_WEIGHT,
        "inlet"     : DEFAULT_BOUNDARY_LOSS_WEIGHT,
        "outlet"    : DEFAULT_BOUNDARY_LOSS_WEIGHT,
        "positivity": training_config.positivity_loss_weight,
    }
    
    if coupling_config is None:
        conservation_value = _conservation_loss(
            network, domain, _DIMENSIONLESS_PEAK_INLET_VELOCITY,
            training_config.n_conservation_stations, training_config.n_conservation_quadrature_points,
            batch.interior.device,
        )
        particle_value = None
    else:
        conservation_value = _particle_conservation_loss(
            network, domain,
            particle_state=coupling_config.particle_state.to(device=batch.interior.device),
            particle_config=coupling_config.particle_config,
            peak_velocity=_DIMENSIONLESS_PEAK_INLET_VELOCITY,
            n_stations=training_config.n_conservation_stations,
            n_quadrature_points=training_config.n_conservation_quadrature_points,
            device=batch.interior.device,
        )
        particle_value = _velocity_boundary_loss(network, batch.particle_points, batch.particle_target)
    
    terms["conservation"] = lambda: conservation_value
    weights["conservation"] = training_config.conservation_loss_weight
    if particle_value is not None:
        terms["particle"] = lambda: particle_value
        weights["particle"] = coupling_config.particle_loss_weight
    
    total_loss = compose_loss(terms=terms, weights=weights)
    return _LossComponents(
        momentum_r=momentum_r_value, momentum_z=momentum_z_value, continuity=continuity_value,
        wall=wall_value, inlet=inlet_value, outlet=outlet_value,
        positivity=positivity_value, conservation=conservation_value, total=total_loss,
        particle=particle_value,
    )


def _record_loss_components(components: _LossComponents) -> dict[str, float]:
    """Reduce every scalar tensor in `components` to a plain Python float.

    Single point of truth turning a `_LossComponents` (still attached to
    the autograd graph) into the plain-float dict `_run_adam_phase` and
    `_run_lbfgs_phase` accumulate into a `LossHistory`, and that
    `_format_progress_line` renders for verbose logging.

    Args:
    - `components`: Loss components for the current step, as returned by
      `_evaluate_loss_components`.

    Returns:
    - A dict mapping `"total"` and every name in
      [`_component_names`][magnetofluidics_pinn.training.trainer._component_names]
      (derived from whether `components.particle` is `None`) to its
      corresponding scalar value.
    """
    names = LOSS_COMPONENT_NAMES + (("particle",) if components.particle is not None else ())
    return {"total": components.total.item()} | {
        name: getattr(components, name).item() for name in names
    }


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
    - `momentum_r`, `momentum_z`, `continuity`, `wall`, `inlet`, `outlet`,
      `positivity`, `conservation`, `total`: The corresponding loss
      component at each `step`, in the same units
      [`_LossComponents`][magnetofluidics_pinn.training.trainer._LossComponents]
      reports them in.
    - `particle`: CHANGED — was a separate `particleLossHistory`
      subclass-like dataclass; folded in here as an optional field. `None`
      for a `train()` call made without an `ParticleConfig`; a tuple the
      same length as every other field otherwise.
    """
    
    step: tuple[int, ...]
    momentum_r: tuple[float, ...]
    momentum_z: tuple[float, ...]
    continuity: tuple[float, ...]
    wall: tuple[float, ...]
    inlet: tuple[float, ...]
    outlet: tuple[float, ...]
    positivity: tuple[float, ...]
    conservation: tuple[float, ...]
    total: tuple[float, ...]
    particle: tuple[float, ...] | None = None  # NEW


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
        """Prints a formatted summary table of initial, intermediate, and final loss values.

        CHANGED: the field list now includes `"particle"` whenever
        `self.adam.particle` is populated, so the same summary table
        format serves both the particle-free and two-way-coupled cases
        without a second, near-identical `print_summary` implementation
        (the former `particleTrainingHistory.print_summary`).
        """
        has_lbfgs = len(self.lbfgs.step) > 0
        fields = LOSS_COMPONENT_NAMES + (("particle",) if self.adam.particle is not None else ()) + ("total",)
        
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


def _format_progress_line(prefix: str, values: dict[str, float]) -> str:
    """Format one verbose progress line shared by both training phases.

    Args:
    - `prefix`: Leading label for the line, e.g. `"Epoch    12"` or
      `"L-BFGS round   2/  4"`.
    - `values`: Mapping from loss-term name to its already-reduced
      (`.item()`-ed) scalar value, in the order they should be printed.

    Returns:
    - A single formatted line, e.g. `"Epoch    12 | Total: 2.26e+01 |
      Momentum R: 1.56e+00 | Momentum Z: 3.21e-01 | Continuity: 4.02e-02 |
      ..."`.
    """
    terms = " | ".join(f"{name.replace('_', ' ').title()}: {value:.2e}" for name, value in values.items())
    return f"{prefix} | {terms}"


def train(
        network: nn.Module,
        domain: Domain,
        fluid_config: FluidConfig,
        training_config: TrainingConfig,
        field_config: MagneticFieldConfig | None = None,
        coupling_config: TwoWayCouplingConfig | None = None,  # NEW
) -> tuple[nn.Module, TrainingHistory]:
    """Train a network to satisfy the flow PDE and boundary conditions.

    CHANGED — this function now covers what two separate entry points
    used to (`train()` and `train_around_particle()`; see this module's
    docstring for the rationale). Called without `particle_config`, it
    reproduces the former `train()`'s behavior exactly: the ordinary
    (particle-free) Stokes/Navier-Stokes problem. Called with an
    [`ParticleConfig`][magnetofluidics_pinn.config.ParticleConfig], it
    reproduces the former `train_around_particle()`'s behavior: the
    two-way-coupled problem solved around a fixed, rigidly-translating
    [`Sphericalparticle`][magnetofluidics_pinn.types.Sphericalparticle] —
    instead of a passive tracer advected by an *undisturbed* flow field
    (`trajectory.integrate_trajectory` with a Faxén correction), this
    solves for the flow the particle itself perturbs, holding the particle
    fixed at `particle_config.particle.axial_position` and translating at
    `particle_config.particle_velocity` — a quasi-steady snapshot of the
    two-way problem, valid because Stokes flow has no memory: at every
    instant the flow field depends only on the particle's *current*
    position and velocity, not its history [@wang2026twoway].

    "Steady state" for that quasi-steady snapshot does not mean waiting
    out a physical transient — the governing equations have no time
    derivative to relax; `stokes_residual` describes an instantaneously
    steady balance regardless of how the training loss is converging. It
    means the training loss has converged: use `TrainingHistory`-style
    diagnostics (residual and boundary-loss magnitudes) *and*
    [`physics.conservation.axial_flow_rate`][magnetofluidics_pinn.physics.conservation.axial_flow_rate]
    evaluated upstream and downstream of the particle (matching the
    reference flow rate there is a strictly *stronger* correctness check
    than local residual smallness, since it also confirms the disturbed
    flow correctly reconnects past the sphere — see
    [`_particle_conservation_loss`][magnetofluidics_pinn.training.trainer._particle_conservation_loss]).
    
    Args:
    - `network`: Network instance to train, e.g., built by
      [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp], optionally
      wrapped with
      [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint]
      (recommended: see that function's docstring). That wrapper's axis
      regularity condition remains valid with an on-axis particle present
      too — see `Sphericalparticle`'s docstring — so it needs no
      particle-specific variant.
    - `domain`: Vessel geometry the network is trained on. Must already be
      nondimensionalized (see
      [`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]).
      With `particle_config` set, `particle_config.particle` must fit
      strictly inside it (checked by
      [`sample_collocation_points_with_particle`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points_with_particle]).
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `training_config`: Training hyperparameters — see
      [`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig] for the
      Adam-phase and L-BFGS-refinement-phase settings, and for the
      `verbose`/`log_every` reporting controls this function used to take
      as its own keyword arguments (now consulted from here instead — see
      this module's docstring). Note that `use_lbfgs_refinement` defaults
      to `True` and runs regardless of `n_epochs`: a small `n_epochs`
      (e.g., for a quick interactive check) does not by itself produce a
      fast call — also set `use_lbfgs_refinement=False` for that.
    - `field_config`: Prescribed magnetic field configuration. Currently
      unused: neither the Stokes/Navier-Stokes interior residual nor the
      particle's rigid-body surface condition depends on the magnetic
      field, so this argument has no effect on today's training run. It
      is kept as an explicit, optional argument (`None` by default) for
      forward compatibility with a magnetic-force-coupled loss term in a
      later phase, rather than dropped now and re-added later with a
      signature change.
    - `particle_config`: NEW. `None` (the default) trains the ordinary,
      particle-free problem. An
      [`particleConfig`][magnetofluidics_pinn.config.particleConfig]
      switches to the two-way-coupled problem described above, using its
      `particle`, `particle_velocity`, `n_particle_surface_points`, and
      `particle_loss_weight`.

    Returns:
    - A tuple `(trained_network, history)`. `trained_network` is a trained
      `torch.nn.Module` instance; the `network` argument passed in is never
      modified, a deep copy is trained and returned instead. `history` is a
      [`TrainingHistory`][magnetofluidics_pinn.training.trainer.TrainingHistory]
      recording how each loss term evolved, e.g. for plotting
      `history.adam.step` against `history.adam.total`; `history.adam.particle`
      (and `history.lbfgs.particle`) is populated only when `particle_config`
      was supplied.

    Raises:
    - `ValueError`: If `training_config.n_epochs` is not strictly positive,
      or if any enabled L-BFGS setting (`lbfgs_n_interior_points`,
      `lbfgs_n_boundary_points`, `lbfgs_rounds`, `lbfgs_iterations_per_round`)
      is not strictly positive. (`log_every` is validated eagerly by
      `TrainingConfig.__post_init__` and `n_particle_surface_points` /
      `particle_loss_weight` / `particle_velocity` by
      `particleConfig.__post_init__`, so neither is re-checked here.)
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
    # `field_config` is accepted (see the docstring above) but not yet
    # consumed by any loss term; deleting it here — rather than merely
    # ignoring it — makes that explicit and prevents an unused-parameter
    # warning from masking a genuine future oversight if a magnetic term
    # is added without wiring this argument through to it.
    del field_config
    
    resolved_device = resolve_device(training_config.device)
    
    # Never mutate the network handed in by the caller: train a private
    # copy and return it, per this module's stated contract.
    trained_network = copy.deepcopy(network).to(resolved_device)
    
    adam_start_time = time.perf_counter()
    adam_history = _run_adam_phase(
        trained_network, domain, fluid_config, training_config, coupling_config, resolved_device
    )
    if training_config.verbose:
        print(
            f"Adam phase finished in {time.perf_counter() - adam_start_time:.1f}s |"
            f"Final Total Loss: {adam_history.total[-1]:.2e}")
    
    lbfgs_start_time = time.perf_counter()
    empty_history_fields = ("step",) + _component_names(coupling_config) + ("total",)
    lbfgs_history = LossHistory(**{name: () for name in empty_history_fields})
    if training_config.use_lbfgs_refinement:
        lbfgs_history = _run_lbfgs_phase(
            trained_network, domain, fluid_config, training_config, coupling_config, resolved_device
        )
    
    if training_config.verbose:
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
        coupling_config: TwoWayCouplingConfig | None,  # NEW — replaces the former separate _run_particle_adam_phase.
        resolved_device: torch.device,
) -> LossHistory:
    """First-order, stochastic-collocation phase: fast, global exploration.

    Mutates `trained_network`'s parameters in place via `optimizer.step()`
    (an unavoidable, intentional side effect of any training loop); the
    surrounding `train()` is what protects the caller's original network
    from that mutation, by operating on a copy.

    CHANGED. `verbose` and `log_every` are now read from `training_config`
    (see this module's docstring) rather than taken as separate
    parameters, and `particle_config` is threaded through to
    `_build_training_batch`/`_evaluate_loss_components` — this single
    implementation replaces both the former `_run_adam_phase` and
    `_run_particle_adam_phase`.

    Args:
    - `trained_network`: The (already-copied) network to update in place.
    - `domain`, `fluid_config`, `training_config`: As passed to `train()`.
    - `coupling_config`: As passed to `train()`.
    - `resolved_device`: Device every collocation batch is placed on.

    Returns:
    - A [`LossHistory`][magnetofluidics_pinn.training.trainer.LossHistory]
      with one entry per epoch.
    """
    optimizer = torch.optim.Adam(trained_network.parameters(), lr=training_config.learning_rate)
    component_names = _component_names(coupling_config)
    accumulator: dict[str, list[float]] = {
        name: [] for name in ("step",) + component_names + ("total",)
    }
    
    for epoch in range(training_config.n_epochs):
        # Re-sampling every epoch, with a deterministically-varying seed,
        # exposes the network to a fresh set of collocation points instead
        # of overfitting to a single fixed grid, while remaining fully
        # reproducible for a given base seed.
        batch = _build_training_batch(
            domain, coupling_config, training_config.n_interior_points, training_config.n_boundary_points,
            training_config.random_seed + epoch,
            axis_clearance_fraction=training_config.axis_clearance_fraction,
            device=resolved_device,
        )
        components = _evaluate_loss_components(
            trained_network, batch, domain, fluid_config, training_config, coupling_config
        )
        
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
        
        current_values = _record_loss_components(components)
        accumulator["step"].append(epoch)
        for name, value in current_values.items():
            accumulator[name].append(value)
        
        if training_config.verbose and (
                epoch % training_config.log_every == 0 or epoch == training_config.n_epochs - 1
        ):
            print(_format_progress_line(f"Epoch {epoch:5d}", current_values))
    
    return LossHistory(**{name: tuple(values) for name, values in accumulator.items()})


def _run_lbfgs_phase(
        trained_network: nn.Module,
        domain: Domain,
        fluid_config: FluidConfig,
        training_config: TrainingConfig,
        coupling_config: TwoWayCouplingConfig | None,  # NEW
        resolved_device: torch.device,
) -> LossHistory:
    """Second-order, fixed-batch refinement phase, run after Adam.

    L-BFGS is a full-batch quasi-Newton method: its curvature estimate is
    only valid across calls that see the *same* loss surface, so — unlike
    the Adam phase — the collocation batch is sampled once and held fixed
    for every internal iteration of every round.

    CHANGED. `verbose` is now read from `training_config`, and
    `coupling_config` is threaded through — this single implementation
    replaces both the former `_run_lbfgs_phase` and
    `_run_particle_lbfgs_phase`.

    Args:
    - `trained_network`: The (already Adam-trained) network to refine in
      place.
    - `domain`, `fluid_config`, `training_config`: As passed to `train()`.
    - `particle_config`: As passed to `train()`.
    - `resolved_device`: Device the fixed collocation batch is placed on.

    Returns:
    - A [`LossHistory`][magnetofluidics_pinn.training.trainer.LossHistory]
      with one entry per `closure` call - typically several per round,
      from `torch.optim.LBFGS`'s internal line search - which is a finer
      granularity than the once-per-round summary printed when
      `training_config.verbose` is `True`.
    """
    batch = _build_training_batch(
        domain, coupling_config, training_config.lbfgs_n_interior_points, training_config.lbfgs_n_boundary_points,
        training_config.random_seed - 1,
        axis_clearance_fraction=training_config.axis_clearance_fraction,
        device=resolved_device,
    )
    optimizer = torch.optim.LBFGS(
        trained_network.parameters(),
        lr=1.0,
        max_iter=training_config.lbfgs_iterations_per_round,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    
    component_names = _component_names(coupling_config)
    accumulator: dict[str, list[float]] = {
        name: [] for name in ("step",) + component_names + ("total",)
    }
    
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        components = _evaluate_loss_components(
            trained_network, batch, domain, fluid_config, training_config, coupling_config
        )
        components.total.backward()
        
        current_values = _record_loss_components(components)
        accumulator["step"].append(len(accumulator["step"]))
        for name, value in current_values.items():
            accumulator[name].append(value)
        return components.total
    
    round_width = len(str(training_config.lbfgs_rounds))
    for round_index in range(training_config.lbfgs_rounds):
        optimizer.step(closure)
        if training_config.verbose:
            prefix = f"L-BFGS round {round_index + 1:{round_width}d}/{training_config.lbfgs_rounds}"
            last_values = {name: accumulator[name][-1] for name in ("total",) + component_names}
            print(_format_progress_line(prefix, last_values))
    
    return LossHistory(**{name: tuple(values) for name, values in accumulator.items()})
