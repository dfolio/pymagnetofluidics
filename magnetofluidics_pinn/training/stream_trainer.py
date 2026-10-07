r"""Training loop for stream-function flow around a sphere.

Only the physics that is *not* built into the network is penalized: the Stokes
momentum residual, the inlet velocity, and the outlet pressure. There is no wall,
particle, core or conservation term, because those hold by construction. Adam is
followed by L-BFGS on a fixed batch, as in `training.trainer`, and the run
honors `TrainingConfig.dtype` and `TrainingConfig.device` end to end.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from magnetofluidics_pinn.boundary_conditions.flow_bc import inlet_velocity_condition
from magnetofluidics_pinn.config import DEFAULT_BOUNDARY_LOSS_WEIGHT, FluidConfig, TrainingConfig
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.exterior import sample_exterior_points, sample_inlet_outlet_points
from magnetofluidics_pinn.training.losses import compose_loss
from magnetofluidics_pinn.types import Domain

# Ordered names of the recorded losses; "total" is recorded first.
STREAM_LOSS_NAMES: tuple[str, ...] = ("total", "momentum_r", "momentum_z", "continuity", "inlet", "outlet")


@dataclass(frozen=True)
class StreamLossHistory:
    """Loss values recorded during a stream-function training run.

    Args:
    - `names`: Recorded loss names, in column order.
    - `adam`: One tuple of values per Adam epoch, in `names` order.
    - `lbfgs`: One tuple of values per L-BFGS closure call, in `names` order.
    """

    names: tuple[str, ...]
    adam: tuple[tuple[float, ...], ...]
    lbfgs: tuple[tuple[float, ...], ...]

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
    domain: Domain, particle_z: float, particle_radius: float, inlet_peak_velocity: float,
    n_interior: int, n_boundary: int, seed: int, device: torch.device, dtype: torch.dtype,
) -> _StreamBatch:
    """Sample one batch.

    Args:
    - `domain`: Nondimensional channel.
    - `particle_z`: Sphere center.
    - `particle_radius`: Sphere radius.
    - `inlet_peak_velocity`: Peak of the imposed inlet profile (may be zero).
    - `n_interior`: Number of interior points.
    - `n_boundary`: Number of inlet plus outlet points.
    - `seed`: Sampling seed.
    - `device`: Target device.
    - `dtype`: Floating-point dtype.

    Returns:
    - A `_StreamBatch`.
    """
    interior = sample_exterior_points(
        domain, particle_z, particle_radius, n_interior, seed, device=device, dtype=dtype
    ).requires_grad_(True)
    inlet, outlet = sample_inlet_outlet_points(domain, n_boundary, seed + 1, device=device, dtype=dtype)
    return _StreamBatch(interior, inlet, inlet_velocity_condition(domain, inlet, inlet_peak_velocity), outlet)


def _evaluate_losses(
    network: nn.Module, batch: _StreamBatch, fluid_config: FluidConfig, config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Compute every loss term and the weighted total.

    Args:
    - `network`: Stream-function network.
    - `batch`: Collocation batch.
    - `fluid_config`: Stokes fluid configuration.
    - `config`: Supplies the momentum weight and `residual_form`.

    Returns:
    - Mapping from `STREAM_LOSS_NAMES` to scalar tensors.
    """
    residual = stokes_residual(network, batch.interior, fluid_config, residual_form=config.residual_form)
    parts = {
        "momentum_r": residual[:, 0:1].square().mean(),
        "momentum_z": residual[:, 1:2].square().mean(),
        "continuity": residual[:, 2:3].square().mean(),  # Monitored only: holds by construction.
        "inlet": (network(batch.inlet_points)[:, :2] - batch.inlet_target).square().mean(),
        "outlet": network(batch.outlet_points)[:, 2:3].square().mean(),
    }
    weights = {"momentum_r": config.momentum_loss_weight, "momentum_z": config.momentum_loss_weight,
               "inlet": DEFAULT_BOUNDARY_LOSS_WEIGHT, "outlet": DEFAULT_BOUNDARY_LOSS_WEIGHT}
    total = compose_loss({name: (lambda value=parts[name]: value) for name in weights}, weights)
    return {"total": total, **parts}


def _record(losses: dict[str, torch.Tensor]) -> tuple[float, ...]:
    """Transfer all losses to the host with a single synchronization.

    Args:
    - `losses`: Mapping from loss name to scalar tensor.

    Returns:
    - Values in `STREAM_LOSS_NAMES` order.
    """
    return tuple(torch.stack([losses[name].detach() for name in STREAM_LOSS_NAMES]).tolist())


def train_stream_function_flow(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    particle_axial_position: float,
    particle_radius: float,
    inlet_peak_velocity: float,
    resample_each_round: bool = True,  # NEW
    min_relative_improvement: float = 1.0e-2,  # NEW
    patience: int = 2,  # NEW
    stagnation_probe_seed: int | None = None,
) -> tuple[nn.Module, StreamLossHistory]:
    """Train a stream-function network on a deep copy and return it.

    Args:
    - `network`: A [`StreamFunctionSphereFlow`][magnetofluidics_pinn.networks.stream_function.StreamFunctionSphereFlow]; never modified.
    - `domain`: Nondimensional channel.
    - `fluid_config`: Fluid configuration with `regime == "stokes"`.
    - `training_config`: Epochs, points, learning rate, L-BFGS settings, `dtype`, `device`.
    - `particle_axial_position`: Sphere center.
    - `particle_radius`: Sphere radius.
    - `inlet_peak_velocity`: Peak inlet velocity; it must match the network's `flow_amplitude`.

    Returns:
    - The trained copy and a [`StreamLossHistory`][magnetofluidics_pinn.training.stream_trainer.StreamLossHistory].

    Raises:
    - `ValueError`: If L-BFGS is enabled with a non-positive setting.
    """
    if training_config.use_lbfgs_refinement and min(
        training_config.lbfgs_n_interior_points, training_config.lbfgs_n_boundary_points,
        training_config.lbfgs_rounds, training_config.lbfgs_iterations_per_round,
    ) <= 0:
        raise ValueError("All L-BFGS settings must be strictly positive when refinement is enabled.")

    device, dtype = training_config.torch_device, training_config.torch_dtype
    trained = copy.deepcopy(network).to(device=device, dtype=dtype)

    def batch(n_interior: int, n_boundary: int, seed: int) -> _StreamBatch:
        return _build_batch(domain, particle_axial_position, particle_radius, inlet_peak_velocity,
                            n_interior, n_boundary, seed, device, dtype)

    optimizer = torch.optim.Adam(trained.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # NEW: damps resampling noise
        optimizer, T_max=training_config.n_epochs, eta_min=1.0e-2 * training_config.learning_rate)
    
    adam_records: list[tuple[float, ...]] = []
    for epoch in range(training_config.n_epochs):
        losses = _evaluate_losses(
            trained, batch(training_config.n_interior_points, training_config.n_boundary_points,
                           training_config.random_seed + epoch),
            fluid_config, training_config,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        if training_config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trained.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        adam_records.append(_record(losses))
        if training_config.verbose and epoch % training_config.log_every == 0:
            print(f"Epoch {epoch:5d} | total {adam_records[-1][0]:.3e}")
    
    probe_seed = (
        training_config.random_seed + 999_999 if stagnation_probe_seed is None else stagnation_probe_seed
    )
    probe_batch = batch(training_config.lbfgs_n_interior_points, training_config.lbfgs_n_boundary_points,
                        probe_seed)  # NEW: fixed across every round
    
    lbfgs_records: list[tuple[float, ...]] = []
    if training_config.use_lbfgs_refinement:
        previous_probe_loss: float | None = None
        stalled_rounds = 0
        for round_index in range(training_config.lbfgs_rounds):
            seed = training_config.random_seed + 100_000 + (round_index if resample_each_round else 0)
            fixed = batch(training_config.lbfgs_n_interior_points,
                          training_config.lbfgs_n_boundary_points, seed)
            lbfgs = torch.optim.LBFGS(
                trained.parameters(), lr=1.0, max_iter=training_config.lbfgs_iterations_per_round,
                history_size=100, line_search_fn="strong_wolfe",
                tolerance_grad=1.0e-12, tolerance_change=1.0e-14,
            )
            
            def closure(batch_=fixed, optimizer=lbfgs) -> torch.Tensor:
                optimizer.zero_grad()
                _losses = _evaluate_losses(trained, batch_, fluid_config, training_config)
                _losses["total"].backward()
                lbfgs_records.append(_record(_losses))
                return _losses["total"]
            
            lbfgs.step(closure)
            # current = lbfgs_records[-1][0]
            # with torch.no_grad():
            probe_loss = _evaluate_losses(trained, probe_batch, fluid_config, training_config)["total"].item()
            
            if training_config.verbose:
                print(f"L-BFGS round {round_index + 1} | train {lbfgs_records[-1][0]:.3e} | probe {probe_loss:.3e}")
            
            if previous_probe_loss is not None:
                improvement = (previous_probe_loss - probe_loss) / abs(previous_probe_loss)
                stalled_rounds = stalled_rounds + 1 if improvement < min_relative_improvement else 0
                if stalled_rounds >= patience:
                    break
            previous_probe_loss = probe_loss

    return trained, StreamLossHistory(STREAM_LOSS_NAMES, tuple(adam_records), tuple(lbfgs_records))


def diagnose_loss_scale(  # NEW
    network: nn.Module, domain: Domain, fluid_config: FluidConfig, training_config: TrainingConfig,
    particle_axial_position: float, particle_radius: float, inlet_peak_velocity: float,
) -> dict[str, float]:
    """Evaluate every loss on one batch *before* training, to catch scaling defects.

    Args:
    - `network`: Stream-function network (not modified).
    - `domain`, `fluid_config`, `training_config`: As in `train_stream_function_flow`.
    - `particle_axial_position`, `particle_radius`, `inlet_peak_velocity`: As in `train_stream_function_flow`.

    Returns:
    - Mapping from each name in `STREAM_LOSS_NAMES` to its initial value.
    """
    device, dtype = training_config.torch_device, training_config.torch_dtype
    probe = copy.deepcopy(network).to(device=device, dtype=dtype)
    batch = _build_batch(domain, particle_axial_position, particle_radius, inlet_peak_velocity,
                         training_config.n_interior_points, training_config.n_boundary_points,
                         training_config.random_seed, device, dtype)
    losses = _evaluate_losses(probe, batch, fluid_config, training_config)
    return dict(zip(STREAM_LOSS_NAMES, _record(losses)))
