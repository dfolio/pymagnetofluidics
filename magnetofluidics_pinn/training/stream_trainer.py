r"""Training loop for stream-function flow around a sphere.

Only the physics that is *not* built into the network is penalized: the Stokes
momentum residual, the inlet velocity, and the outlet pressure. There is no wall,
particle, core or conservation term, because those hold by construction. Adam is
followed by L-BFGS on a fixed batch, as in `training.trainer`, and the run
honors `TrainingConfig.dtype` and `TrainingConfig.device` end to end.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import inlet_velocity_condition
from magnetofluidics_pinn.config import DEFAULT_BOUNDARY_LOSS_WEIGHT, FluidConfig, \
    TrainingConfig
from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.networks import StreamFunctionSphereFlow
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.exterior import sample_exterior_points, \
    sample_inlet_outlet_points
from magnetofluidics_pinn.training.losses import compose_loss, LossHistoryBase

# Ordered names of the recorded losses; "total" is recorded first.
STREAM_LOSS_NAMES: tuple[str, ...] = ("total", "momentum_r", "momentum_z", "continuity",
                                      "inlet", "outlet")


@dataclass(frozen=True)
class StreamLossHistory(LossHistoryBase):
    """Loss values recorded during a stream-function training run.

    Args:
    - `names`: Recorded loss names, in column order.
    - `adam`: One tuple of values per Adam epoch, in `names` order.
    - `lbfgs`: One tuple of values per L-BFGS closure call, in `names` order.
    - `probe`: Held-out probe loss after each L-BFGS round.
    - `best_probe`: Probe loss of the state that was restored (`nan` if no restoration took place).
    - `restored_stage`: `"adam"` if the Adam-end state was restored, `"lbfgs-round-k"` for round `k`,
      `"last"` if nothing was restored (the returned network is then the last iterate).
    """

    names: tuple[str, ...]
    adam: tuple[tuple[float, ...], ...]
    lbfgs: tuple[tuple[float, ...], ...]
    probe: tuple[float, ...] = ()
    best_probe: float = math.nan
    restored_stage: str = "last"

    def final(self) -> dict[str, float]:
        """Return the last recorded value of every loss.

        Returns:
        - Mapping from loss name to value.

        Raises:
        - `ValueError`: If nothing was recorded.
        """
        records = self.lbfgs or self.adam
        if not records:
            raise ValueError("The history is empty.")
        return dict(zip(self.names, records[-1]))

    def print_summary(self) -> None:
        """Prints a formatted summary table of initial, intermediate, and final loss values."""
        has_lbfgs = len(self.lbfgs) > 0
        has_adam = len(self.adam) > 0

        if not has_adam and not has_lbfgs:
            print("No training data to summarize.")
            return

        print("\n" + "=" * 78)
        print(
            f"{'Loss Component':<20} | {'Initial (Adam)':<14} | {'Post-Adam':<14} | {'Final Loss':<14} | {'Factor':<8}")
        print("-" * 78)

        for name in self.names:
            if has_adam:
                initial = self.adam[0][self.names.index(name)] if self.adam else float(
                    'nan')
                post_adam = self.adam[-1][
                    self.names.index(name)] if self.adam else float('nan')
            else:
                initial = float('nan')
                post_adam = float('nan')

            if has_lbfgs:
                final = self.lbfgs[-1][self.names.index(name)]
            else:
                final = post_adam

            if math.isfinite(initial) and initial > 0:
                reduction = initial / max(final, 1e-15)
            else:
                reduction = float('nan')

            if has_adam and has_lbfgs:
                print(
                    f"{name.upper():<20} | {initial:14.4e} | {post_adam:14.4e} | {final:14.4e} | {reduction:7.1f}x"
                )
            elif has_adam:
                print(
                    f"{name.upper():<20} | {initial:14.4e} | {post_adam:14.4e} | {reduction:7.1f}x"
                )
            else:
                print(
                    f"{name.upper():<20} | {'N/A':<14} | {'N/A':<14} | {final:14.4e} | {'N/A':<8}"
                )

        print("=" * 78)
        if math.isfinite(self.best_probe):
            print(
                f"Best probe loss: {self.best_probe:.4e} (restored from {self.restored_stage})")
        print()


@dataclass(frozen=True)
class _StreamBatch:
    """Interior, inlet and outlet points with the inlet target.

    Args:
    - `interior`: Interior points requiring gradients.
    - `inlet_points`: Inlet points.
    - `inlet_target`: Target `(u_r, u_z)` at the inlet.
    - `outlet_points`: Outlet points.
    """

    interior: torch.Tensor
    inlet_points: torch.Tensor
    inlet_target: torch.Tensor
    outlet_points: torch.Tensor


def _build_batch(
    domain: Domain, particle_z: float, particle_radius: float,
    inlet_peak_velocity: float,
    n_interior: int, n_boundary: int, seed: int, device: torch.device,
    dtype: torch.dtype,
    shell_fraction: float = 0.25, shell_factor: float = 2.5, band_fraction: float = 0.0,
    band_factor: float = 1.5,
    axis_clearance_fraction: float = 0.0,
) -> _StreamBatch:
    r"""Sample one batch.

    Args:
    - `domain`: Nondimensional channel.
    - `particle_z`, `particle_radius`: Sphere centre and radius.
    - `inlet_peak_velocity`: Peak of the imposed inlet profile (may be zero).
    - `n_interior`, `n_boundary`: Number of interior points and of inlet plus outlet points.
    - `seed`: Sampling seed.
    - `device`, `dtype`: Placement of the tensors.
    - `shell_fraction`, `shell_factor`: Interior-point distribution, forwarded to `sample_exterior_points`.
    - `band_fraction`, `band_factor`: Dedicated near-surface share of the interior batch (`0.0` disables it).
    - `axis_clearance_fraction`: NEW. Fraction of $R$ kept free around the axis, forwarded to the sampler.

    Returns:
    - A `_StreamBatch`.
    """
    n_band = round(band_fraction * n_interior)
    interior = sample_exterior_points(
        domain, particle_z, particle_radius, n_interior - n_band, seed, shell_fraction,
        shell_factor, axis_clearance_fraction=axis_clearance_fraction,
        device=device, dtype=dtype
    )
    if n_band > 0:
        band_points = _sample_near_surface_band(
            domain, particle_z, particle_radius, band_factor, n_band, seed + 717_717,
            device=device, dtype=dtype)
        interior = torch.cat((interior, band_points))
    interior = interior.requires_grad_(True)
    inlet, outlet = sample_inlet_outlet_points(domain, n_boundary, seed + 1,
                                               device=device, dtype=dtype)
    return _StreamBatch(interior, inlet,
                        inlet_velocity_condition(domain, inlet, inlet_peak_velocity),
                        outlet)


def _evaluate_losses(
    network: nn.Module | StreamFunctionSphereFlow,
    batch: _StreamBatch,
    fluid_config: FluidConfig,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    r"""Compute every loss term and the weighted total objective.

    When `config.residual_form == "native"`, queries `StreamFunctionSphereFlow.stokes_residual_sz`
    directly, narrowing the type to suppress PyCharm inspection warnings.

    Args:
    - `network`: Flow model, expected to be an instance of `StreamFunctionSphereFlow`
      when `config.residual_form == "native"`.
    - `batch`: Collocation batches for interior, inlet, and outlet points.
    - `fluid_config`: Stokes fluid configuration.
    - `config`: Hyperparameter configuration specifying loss weights and `residual_form`.

    Returns:
    * Dictionary mapping names in `STREAM_LOSS_NAMES` to scalar tensors attached to the graph.

    Raises:
    * `TypeError`: If `config.residual_form == "native"` but `network` is not an
      instance of `StreamFunctionSphereFlow`.
    """
    # residual = stokes_residual(network, batch.interior, fluid_config, residual_form=config.residual_form)
    if config.residual_form == "native":  # CHANGED
        if not isinstance(network, StreamFunctionSphereFlow):
            raise TypeError(
                f"residual_form='native' requires a StreamFunctionSphereFlow instance; "
                f"got {type(network).__name__}."
            )
        residual = network.stokes_residual_sz(batch.interior)
    else:
        residual = stokes_residual(network, batch.interior, fluid_config,
                                   residual_form=config.residual_form)

    parts = {
        "momentum_r": residual[:, 0:1].square().mean(),
        "momentum_z": residual[:, 1:2].square().mean(),
        "continuity": residual[:, 2:3].square().mean(),
        # Monitored only: holds by construction.
        "inlet"     : (network(batch.inlet_points)[
                           :, :2] - batch.inlet_target).square().mean(),
        "outlet"    : network(batch.outlet_points)[:, 2:3].square().mean(),
    }
    weights = {"momentum_r": config.momentum_loss_weight,
               "momentum_z": config.momentum_loss_weight,
               "inlet"     : DEFAULT_BOUNDARY_LOSS_WEIGHT,
               "outlet"    : DEFAULT_BOUNDARY_LOSS_WEIGHT}
    total = compose_loss({name: (lambda value=parts[name]: value) for name in weights},
                         weights)
    return {"total": total, **parts}


def _record(losses: dict[str, torch.Tensor]) -> tuple[float, ...]:
    """Transfer all losses to the host with a single synchronization.

    Args:
    - `losses`: Mapping from loss name to scalar tensor.

    Returns:
    - Values in `STREAM_LOSS_NAMES` order.
    """
    return tuple(
        torch.stack([losses[name].detach() for name in STREAM_LOSS_NAMES]).tolist())


def _sample_near_surface_band(  # NEW
    domain: Domain, particle_z: float, particle_radius: float, band_factor: float,
    n_points: int, seed: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    r"""Draw interior collocation points uniform in area inside the near-surface band.

    The band is the meridian annulus $a \le \rho \le \mathrm{band\_factor} \cdot a$
    around the sphere, where $\rho$ is the distance to the sphere center. Points are
    uniform in area, so equator and poles are sampled without bias, and
    $\rho \ge a$ holds by construction, so no point falls inside the sphere.

    Args:
    - `domain`: Nondimensional channel; candidates outside the open fluid box
      are rejected.
    - `particle_z`, `particle_radius`: Sphere center and radius, in domain units.
    - `band_factor`: Band outer radius in units of the sphere radius; must exceed 1.
    - `n_points`: Number of points requested.
    - `seed`: Seed of a device-local generator, disjoint from the main sampler's.
    - `device`, `dtype`: Placement and dtype of the returned tensor.

    Returns:
    - Tensor `(n_points, 2)` of `(r, z)` points inside the band and the channel.

    Raises:
    - `ValueError`: If `band_factor` does not exceed 1.
    - `RuntimeError`: If fewer than `n_points` candidates survive the channel guards.
    """
    if band_factor <= 1.0:
        raise ValueError(f"band_factor must exceed 1; got {band_factor!r}.")
    rho_inner, rho_outer = particle_radius, band_factor * particle_radius
    generator = torch.Generator(device=device).manual_seed(seed)
    n_candidates = 4 * n_points + 16
    rho_sq = rho_inner ** 2 + (rho_outer ** 2 - rho_inner ** 2) * torch.rand(
        n_candidates, 1, generator=generator, device=device, dtype=dtype)
    theta = math.pi * torch.rand(n_candidates, 1, generator=generator, device=device,
                                 dtype=dtype)
    rho = torch.sqrt(rho_sq)
    candidates = torch.cat(
        [rho * torch.sin(theta), particle_z + rho * torch.cos(theta)], dim=1)
    keep = ((candidates[:, 0] < domain.radius) & (candidates[:, 1] > 0.0)
            & (candidates[:, 1] < domain.length))
    accepted = candidates[keep]
    if accepted.shape[0] < n_points:
        raise RuntimeError(
            f"Only {accepted.shape[0]} of {n_points} near-surface band points survived "
            "the channel guards; shrink the band or move the sphere away from the ends.")
    return accepted[:n_points]


def train_stream_function_flow(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    particle_axial_position: float,
    particle_radius: float,
    inlet_peak_velocity: float,
    resample_each_round: bool = True,
    min_relative_improvement: float = 1.0e-2,
    patience: int = 2,  # NEW
    stagnation_probe_seed: int | None = None,
    shell_fraction: float = 0.25,
    shell_factor: float = 2.5,
    band_fraction: float = 0.0,
    band_factor: float = 1.5,
    restore_best_probe: bool = True,
    *,
    axis_clearance_fraction: float = 0.0,  # NEW
    refine_dtype: torch.dtype | None = None,  # NEW
    restore_adam_end: bool = True,  # NEW
) -> tuple[nn.Module, StreamLossHistory]:
    """Train a stream-function network on a deep copy and return it.

    Args:
    - `network`: A `StreamFunctionSphereFlow`; never modified.
    - `domain`: Nondimensional channel.
    - `fluid_config`: Fluid configuration with `regime == "stokes"`.
    - `training_config`: Epochs, points, learning rate, L-BFGS settings, `dtype`, `device`.
    - `particle_axial_position`, `particle_radius`: Sphere centre and radius.
    - `inlet_peak_velocity`: Peak inlet velocity; it must match the network's `flow_amplitude`.
    - `resample_each_round`: Draw a new fixed L-BFGS batch at every round.
    - `min_relative_improvement`, `patience`: Early-stopping rule on the probe loss.
    - `stagnation_probe_seed`: Seed of the held-out probe batch; defaults to `random_seed + 999_999`.
    - `shell_fraction`, `shell_factor`, `band_fraction`, `band_factor`: Interior-point distribution, shared by every batch.
    - `restore_best_probe`: Restore the state with the lowest probe loss at the end.
    - `axis_clearance_fraction`: NEW. Fraction of $R$ kept free around the axis in every batch (default `0.0`).
    - `refine_dtype`: NEW. dtype of the L-BFGS phase, for example `torch.float64`; the network is cast back to the training
      dtype before it is returned. `None` keeps the training dtype.
    - `restore_adam_end`: NEW. Include the Adam-end state among the restoration candidates.

    Returns:
    - The trained copy, in the training dtype, and a `StreamLossHistory`.

    Raises:
    - `ValueError`: If L-BFGS is enabled with a non-positive setting.
    """
    if training_config.use_lbfgs_refinement and min(
        training_config.lbfgs_n_interior_points,
        training_config.lbfgs_n_boundary_points,
        training_config.lbfgs_rounds, training_config.lbfgs_iterations_per_round,
    ) <= 0:
        raise ValueError(
            "All L-BFGS settings must be strictly positive when refinement is enabled.")

    device, dtype = training_config.torch_device, training_config.torch_dtype
    trained = copy.deepcopy(network).to(device=device, dtype=dtype)

    def batch(n_interior: int, n_boundary: int, seed: int,
              batch_dtype: torch.dtype) -> _StreamBatch:  # CHANGED: dtype argument
        return _build_batch(domain, particle_axial_position, particle_radius,
                            inlet_peak_velocity, n_interior, n_boundary,
                            seed, device, batch_dtype, shell_fraction, shell_factor,
                            band_fraction, band_factor,
                            axis_clearance_fraction)

    optimizer = torch.optim.Adam(trained.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.n_epochs,
        eta_min=1.0e-2 * training_config.learning_rate)

    adam_records: list[tuple[float, ...]] = []
    for epoch in range(training_config.n_epochs):
        losses = _evaluate_losses(
            trained,
            batch(training_config.n_interior_points, training_config.n_boundary_points,
                  training_config.random_seed + epoch, dtype), fluid_config,
            training_config)
        optimizer.zero_grad()
        losses["total"].backward()
        if training_config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trained.parameters(),
                                           training_config.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        adam_records.append(_record(losses))
        if training_config.verbose and epoch % training_config.log_every == 0:
            print(f"Epoch {epoch:5d} | total {adam_records[-1][0]:.3e}")

    refine = dtype if refine_dtype is None else refine_dtype  # NEW
    if training_config.use_lbfgs_refinement and refine != dtype:
        trained = trained.to(dtype=refine)  # NEW: the L-BFGS phase runs in `refine`

    probe_seed = (
        training_config.random_seed + 999_999 if stagnation_probe_seed is None else stagnation_probe_seed)
    probe_batch = batch(training_config.lbfgs_n_interior_points,
                        training_config.lbfgs_n_boundary_points, probe_seed, refine)

    def probe_loss_now() -> float:
        return _evaluate_losses(trained, probe_batch, fluid_config, training_config)[
            "total"].item()

    best_state, best_probe, restored_stage, probe_history = None, math.inf, "last", []
    lbfgs_records: list[tuple[float, ...]] = []
    if training_config.use_lbfgs_refinement:
        if restore_adam_end:  # NEW: the Adam-end state competes with every L-BFGS round
            best_probe, best_state, restored_stage = probe_loss_now(), copy.deepcopy(
                trained.state_dict()), "adam"
        previous_probe_loss: float | None = None
        stalled_rounds = 0
        for round_index in range(training_config.lbfgs_rounds):
            seed = training_config.random_seed + 100_000 + (
                round_index if resample_each_round else 0)
            fixed = batch(training_config.lbfgs_n_interior_points,
                          training_config.lbfgs_n_boundary_points, seed, refine)
            lbfgs = torch.optim.LBFGS(
                trained.parameters(), lr=1.0,
                max_iter=training_config.lbfgs_iterations_per_round,
                history_size=100, line_search_fn="strong_wolfe", tolerance_grad=1.0e-12,
                tolerance_change=1.0e-14,
            )

            def closure(batch_=fixed, optimizer=lbfgs) -> torch.Tensor:
                optimizer.zero_grad()
                _losses = _evaluate_losses(trained, batch_, fluid_config,
                                           training_config)
                _losses["total"].backward()
                lbfgs_records.append(_record(_losses))
                return _losses["total"]

            lbfgs.step(closure)
            probe_loss = probe_loss_now()
            probe_history.append(probe_loss)
            if probe_loss < best_probe:
                best_probe, best_state = probe_loss, copy.deepcopy(trained.state_dict())
                restored_stage = f"lbfgs-round-{round_index + 1}"  # NEW
            if training_config.verbose:
                print(
                    f"L-BFGS round {round_index + 1} | train {lbfgs_records[-1][0]:.3e} | probe {probe_loss:.3e}")
            if previous_probe_loss is not None:
                improvement = (previous_probe_loss - probe_loss) / abs(
                    previous_probe_loss)
                stalled_rounds = stalled_rounds + 1 if improvement < min_relative_improvement else 0
                if stalled_rounds >= patience:
                    break
            previous_probe_loss = probe_loss

    restored = restore_best_probe and best_state is not None
    if restored:
        trained.load_state_dict(best_state)
    if next(
        trained.parameters()).dtype != dtype:  # NEW: hand the network back in the training dtype
        trained = trained.to(dtype=dtype)
    return trained, StreamLossHistory(
        STREAM_LOSS_NAMES, tuple(adam_records), tuple(lbfgs_records),
        tuple(probe_history),
        best_probe=best_probe if restored else math.nan,
        restored_stage=restored_stage if restored else "last",  # NEW
    )


def diagnose_loss_scale(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    particle_axial_position: float,
    particle_radius: float,
    inlet_peak_velocity: float,
    shell_fraction: float = 0.25,
    shell_factor: float = 2.5,
    band_fraction: float = 0.0,
    band_factor: float = 1.5,
    axis_clearance_fraction: float = 0.0,  # NEW
) -> dict[str, float]:
    """Evaluate every loss on one batch *before* training, to catch scaling defects.

    Args:
    - `network`: Stream-function network (not modified).
    - `domain`, `fluid_config`, `training_config`: As in `train_stream_function_flow`.
    - `particle_axial_position`, `particle_radius`, `inlet_peak_velocity`: As in `train_stream_function_flow`.
    - `shell_fraction`, `shell_factor`, `band_fraction`, `band_factor`: Distribution of the audit batch; pass the values of the run.
    - `axis_clearance_fraction`: NEW. Axis clearance of the audit batch; pass the value of the run.

    Returns:
    - Mapping from each name in `STREAM_LOSS_NAMES` to its initial value.
    """
    device, dtype = training_config.torch_device, training_config.torch_dtype
    probe = copy.deepcopy(network).to(device=device, dtype=dtype)
    batch = _build_batch(
        domain, particle_axial_position, particle_radius, inlet_peak_velocity,
        training_config.n_interior_points,
        training_config.n_boundary_points, training_config.random_seed, device, dtype,
        shell_fraction, shell_factor,
        band_fraction, band_factor, axis_clearance_fraction,
    )
    return dict(zip(STREAM_LOSS_NAMES, _record(
        _evaluate_losses(probe, batch, fluid_config, training_config))))
