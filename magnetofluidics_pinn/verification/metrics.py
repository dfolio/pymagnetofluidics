r"""Independent, held-out evaluators for a trained flow network's physical fidelity.

`magnetofluidics_pinn.verification.metrics` provides the reusable
measurement layer a verification notebook builds its quantitative claims
on: relative and pointwise velocity error against an analytical reference,
profile invariance across axial stations, radial-leakage quantification,
global mass-flux conservation, pressure-gradient fidelity, and a held-out
PDE residual check, plus a structural hard-constraint probe and a
trajectory-vs-reference comparator. Every function is a pure evaluator: it
consumes an already-trained (or, for
[`evaluate_structural_constraints`][magnetofluidics_pinn.verification.metrics.evaluate_structural_constraints],
already-initialized) network and returns a plain Python or `pandas`
summary, with no side effects and no dependency on how that network was
produced — [`training.trainer.train`][magnetofluidics_pinn.training.trainer.train],
a checkpoint loaded via
[`io_utils.load_checkpoint`][magnetofluidics_pinn.io_utils.load_checkpoint],
or a hand-built network for a unit test are all equally valid inputs.

Every function that consumes an `nn.Module` resolves that module's device
and dtype internally, via
[`device_utils.resolve_module_device`][magnetofluidics_pinn.device_utils.resolve_module_device]
and
[`device_utils.resolve_module_dtype`][magnetofluidics_pinn.device_utils.resolve_module_dtype],
and moves any caller-supplied grid onto them before use — the same
single-resolution-point convention
[`visualization.plotting.plot_streamlines`][magnetofluidics_pinn.visualization.plotting.plot_streamlines]
already follows, avoiding the cross-device and cross-dtype mismatches
[`device_utils`][magnetofluidics_pinn.device_utils] warns against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
from magnetofluidics_pinn.config import FluidConfig
from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.physics.conservation import axial_flow_rate
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.collocation import sample_collocation_points
from magnetofluidics_pinn.types import Domain, ParticleState
from magnetofluidics_pinn.verification.analytical import (
    analytical_pressure_gradient,
    analytical_tracer_streamline,
)

# Column order `stokes_residual` returns; shared by every residual-facing
# function in this module so the mapping from column index to physical
# equation is declared exactly once.
_RESIDUAL_COMPONENT_NAMES = ("momentum_r", "momentum_z", "continuity")


def summarize_residual(residual: torch.Tensor, component_names: tuple[str, ...]) -> pd.DataFrame:
    r"""Reduce a per-point PDE residual tensor to per-component max/RMS statistics.

    Args:
    - `residual`: Tensor of shape `(n_points, len(component_names))`, e.g.
      the output of
      [`physics.fluid_residuals.stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual].
    - `component_names`: Name of each column of `residual`, in order.

    Returns:
    - A `pandas.DataFrame` indexed by `component_names` plus a `"total"`
      row (statistics over every entry, across all components), with
      columns `"max_abs"` and `"rms"`:

      $$ \text{max\_abs} = \max_i \abs{r_i}, \qquad \text{rms} =
      \sqrt{\frac{1}{N}\sum_i r_i^2}. $$

    Raises:
    - `ValueError`: If `residual` is not 2-D, or if its second dimension
      does not match `len(component_names)`.
    """
    if residual.ndim != 2 or residual.shape[1] != len(component_names):
        raise ValueError(
            f"residual must have shape (n_points, {len(component_names)}); got {tuple(residual.shape)}."
        )
    detached = residual.detach()
    records = [
        {
            "component": name,
            "max_abs": detached[:, index].abs().max().item(),
            "rms": torch.sqrt(torch.mean(detached[:, index] ** 2)).item(),
        }
        for index, name in enumerate(component_names)
    ]
    records.append(
        {
            "component": "total",
            "max_abs": detached.abs().max().item(),
            "rms": torch.sqrt(torch.mean(detached ** 2)).item(),
        }
    )
    return pd.DataFrame.from_records(records).set_index("component")


def evaluate_structural_constraints(
    network: nn.Module,
    domain: Domain,
    n_probe_points: int,
    random_seed: int = 0,
) -> dict[str, float]:
    r"""Probe the axis- and wall-regularity conditions at random axial positions.

    Checks $u_r(0,z)=0$, $u_r(R,z)=0$, $u_z(R,z)=0$, and, via automatic
    differentiation, $\pdv{u_z}{r}\big|_{r=0}=0$ and
    $\pdv{p}{r}\big|_{r=0}=0$ — the conditions
    [`networks.constraints.apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint]
    is meant to enforce structurally. Meant to be run on a **freshly
    initialized, untrained** network: these conditions are asserted to
    hold for *every* parameter value, so verifying them before any
    training has occurred is what actually tests the structural claim,
    rather than a trained network's incidental behavior.

    Args:
    - `network`: A flow network, typically wrapped with
      `apply_hard_wall_constraint`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `n_probe_points`: Number of random axial positions to probe.
    - `random_seed`: Seed for the axial-position sampling.

    Returns:
    - A dict with keys `"max_abs_ur_axis"`, `"max_abs_ur_wall"`,
      `"max_abs_uz_wall"`, `"max_abs_duz_dr_axis"`, and
      `"max_abs_dp_dr_axis"`, each the maximum absolute value observed
      across every probed axial position.

    Raises:
    - `ValueError`: If `n_probe_points` is not strictly positive.
    """
    if n_probe_points <= 0:
        raise ValueError(f"n_probe_points must be strictly positive; got {n_probe_points!r}.")

    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    generator = torch.Generator(device=resolved_device).manual_seed(random_seed)
    axial_probe = domain.length * torch.rand(
        n_probe_points, 1, generator=generator, device=resolved_device, dtype=resolved_dtype
    )

    axis_points = torch.cat([torch.zeros_like(axial_probe), axial_probe], dim=1).requires_grad_(True)
    wall_points = torch.cat([torch.full_like(axial_probe, domain.radius), axial_probe], dim=1)

    axis_output = network(axis_points)
    grad_axis_uz = scalar_field_gradient(axis_output[:, 1:2], axis_points)[:, 0:1]
    grad_axis_p = scalar_field_gradient(axis_output[:, 2:3], axis_points)[:, 0:1]
    with torch.no_grad():
        wall_output = network(wall_points)

    return {
        "max_abs_ur_axis": axis_output[:, 0:1].detach().abs().max().item(),
        "max_abs_ur_wall": wall_output[:, 0:1].abs().max().item(),
        "max_abs_uz_wall": wall_output[:, 1:2].abs().max().item(),
        "max_abs_duz_dr_axis": grad_axis_uz.detach().abs().max().item(),
        "max_abs_dp_dr_axis": grad_axis_p.detach().abs().max().item(),
    }


@dataclass(frozen=True)
class ProfileEvaluation:
    """Predicted and analytical axial-velocity profiles at a set of axial stations.

    Shared, immutable data container consumed both by the profile-based
    metrics in this module
    ([`compute_profile_invariance`][magnetofluidics_pinn.verification.metrics.compute_profile_invariance],
    [`compute_radial_leakage`][magnetofluidics_pinn.verification.metrics.compute_radial_leakage])
    and by the matching plotting functions in
    [`visualization.plotting`][magnetofluidics_pinn.visualization.plotting]
    (`plot_velocity_profile_comparison`, `plot_profile_error`,
    `plot_radial_leakage`), so the network is evaluated on the profile grid
    exactly once regardless of how many metrics or plots are derived from
    it.

    Args:
    - `z_stations`: Axial stations at which profiles were evaluated.
    - `radial_grid`: Tensor of shape `(n_radial,)`, the shared radial grid.
    - `predicted_uz`, `predicted_ur`: Tensors of shape
      `(n_stations, n_radial)`, the network's predicted axial and radial
      velocity at each station and radial position.
    - `analytical_uz`: Tensor of shape `(n_radial,)`, the analytical
      (Hagen-Poiseuille) axial-velocity profile — shared across every
      station, since the profile is $z$-independent.

    Raises:
    - `ValueError`: If any tensor's shape is inconsistent with the others.
    """

    z_stations: tuple[float, ...]
    radial_grid: torch.Tensor
    predicted_uz: torch.Tensor
    predicted_ur: torch.Tensor
    analytical_uz: torch.Tensor

    def __post_init__(self) -> None:
        n_stations = len(self.z_stations)
        n_radial = self.radial_grid.shape[0]
        if self.predicted_uz.shape != (n_stations, n_radial):
            raise ValueError(
                f"predicted_uz must have shape ({n_stations}, {n_radial}); got "
                f"{tuple(self.predicted_uz.shape)}."
            )
        if self.predicted_ur.shape != (n_stations, n_radial):
            raise ValueError(
                f"predicted_ur must have shape ({n_stations}, {n_radial}); got "
                f"{tuple(self.predicted_ur.shape)}."
            )
        if self.analytical_uz.shape != (n_radial,):
            raise ValueError(
                f"analytical_uz must have shape ({n_radial},); got {tuple(self.analytical_uz.shape)}."
            )


def evaluate_velocity_profiles(
    network: nn.Module,
    domain: Domain,
    z_stations: tuple[float, ...],
    radial_grid: torch.Tensor,
    peak_velocity: float,
) -> ProfileEvaluation:
    """Evaluate predicted and analytical axial-velocity profiles at a set of axial stations.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `z_stations`: Axial stations at which to evaluate profiles.
    - `radial_grid`: 1-D tensor of radial positions shared across every
      station.
    - `peak_velocity`: Dimensionless centerline velocity $u_\\text{peak}^*$.

    Returns:
    - A populated
      [`ProfileEvaluation`][magnetofluidics_pinn.verification.metrics.ProfileEvaluation].

    Raises:
    - `ValueError`: If `z_stations` is empty, or if `radial_grid` is not a
      1-D tensor.
    """
    if not z_stations:
        raise ValueError("z_stations must contain at least one axial station.")
    if radial_grid.ndim != 1:
        raise ValueError(f"radial_grid must be a 1-D tensor; got shape {tuple(radial_grid.shape)}.")

    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    radial_grid = radial_grid.to(device=resolved_device, dtype=resolved_dtype)

    n_radial = radial_grid.shape[0]
    station_tensor = torch.as_tensor(z_stations, dtype=resolved_dtype, device=resolved_device)
    radial_mesh = radial_grid.unsqueeze(0).expand(len(z_stations), -1).reshape(-1, 1)
    axial_mesh = station_tensor.unsqueeze(1).expand(-1, n_radial).reshape(-1, 1)
    coordinates = torch.cat([radial_mesh, axial_mesh], dim=1)

    with torch.no_grad():
        prediction = network(coordinates)
    predicted_ur = prediction[:, 0].reshape(len(z_stations), n_radial)
    predicted_uz = prediction[:, 1].reshape(len(z_stations), n_radial)
    analytical_uz = peak_velocity * (1.0 - (radial_grid / domain.radius) ** 2)

    return ProfileEvaluation(
        z_stations=z_stations,
        radial_grid=radial_grid.detach(),
        predicted_uz=predicted_uz.detach(),
        predicted_ur=predicted_ur.detach(),
        analytical_uz=analytical_uz.detach(),
    )


def _cartesian_grid(radial_grid: torch.Tensor, axial_grid: torch.Tensor) -> torch.Tensor:
    """Build a `(n_radial * n_axial, 2)` coordinate grid from two independent 1-D grids.

    Private helper shared by
    [`compute_global_l2_error`][magnetofluidics_pinn.verification.metrics.compute_global_l2_error],
    [`compute_max_pointwise_error`][magnetofluidics_pinn.verification.metrics.compute_max_pointwise_error],
    and
    [`evaluate_residual_grid`][magnetofluidics_pinn.verification.metrics.evaluate_residual_grid].

    Args:
    - `radial_grid`: 1-D tensor of radial coordinates.
    - `axial_grid`: 1-D tensor of axial coordinates.

    Returns:
    - Tensor of shape `(radial_grid.numel() * axial_grid.numel(), 2)`, the
      Cartesian product of both grids, `(r, z)` per row, in row-major
      (`r`-outer) order.

    Raises:
    - `ValueError`: If either input is not a 1-D tensor.
    """
    if radial_grid.ndim != 1 or axial_grid.ndim != 1:
        raise ValueError(
            "radial_grid and axial_grid must both be 1-D tensors; got shapes "
            f"{tuple(radial_grid.shape)} and {tuple(axial_grid.shape)}."
        )
    radial_mesh, axial_mesh = torch.meshgrid(radial_grid, axial_grid, indexing="ij")
    return torch.stack([radial_mesh.reshape(-1), axial_mesh.reshape(-1)], dim=1)


def _analytical_velocity(coordinates: torch.Tensor, radius: float, peak_velocity: float) -> torch.Tensor:
    """Evaluate the `(u_r, u_z)` Hagen-Poiseuille velocity vector at given coordinates.

    A private helper shared by
    [`compute_global_l2_error`][magnetofluidics_pinn.verification.metrics.compute_global_l2_error]
    and
    [`compute_max_pointwise_error`][magnetofluidics_pinn.verification.metrics.compute_max_pointwise_error].
    Callers needing the full `(u_r, u_z, p)` field should use
    [`verification.analytical.hagen_poiseuille_velocity_field`][magnetofluidics_pinn.verification.analytical.hagen_poiseuille_velocity_field]
    instead.
    """
    radial_position = coordinates[:, 0:1]
    axial_velocity = peak_velocity * (1.0 - (radial_position / radius) ** 2)
    return torch.cat([torch.zeros_like(axial_velocity), axial_velocity], dim=1)


def compute_global_l2_error(
    network: nn.Module,
    domain: Domain,
    radial_grid: torch.Tensor,
    axial_grid: torch.Tensor,
    peak_velocity: float,
) -> float:
    r"""Compute the global relative $L^2$ velocity error against Hagen-Poiseuille flow.

    $$ \varepsilon_{L^2} = \frac{\norm{\vb{u}_\text{predicted} -
    \vb{u}_\text{analytical}}_2}{\norm{\vb{u}_\text{analytical}}_2}, $$

    evaluated over the full Cartesian product of `radial_grid` and
    `axial_grid`.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `radial_grid`, `axial_grid`: 1-D tensors spanning the evaluation grid.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.

    Returns:
    - The scalar relative $L^2$ error.

    Raises:
    - `ValueError`: If `radial_grid` or `axial_grid` is not 1-D, or if the
      analytical velocity field's norm is zero (a degenerate
      `peak_velocity`, already excluded by `hagen_poiseuille_velocity_field`
      elsewhere, but checked here too since this function does not go
      through that validation path).
    """
    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    coordinates = _cartesian_grid(
        radial_grid.to(device=resolved_device, dtype=resolved_dtype),
        axial_grid.to(device=resolved_device, dtype=resolved_dtype),
    )
    with torch.no_grad():
        predicted_velocity = network(coordinates)[:, :2]
    analytical_velocity = _analytical_velocity(coordinates, domain.radius, peak_velocity)

    denominator = torch.linalg.norm(analytical_velocity)
    if denominator.item() == 0.0:
        raise ValueError("Analytical velocity field norm is zero; the relative L2 error is undefined.")
    return (torch.linalg.norm(predicted_velocity - analytical_velocity) / denominator).item()


def compute_max_pointwise_error(
    network: nn.Module,
    domain: Domain,
    radial_grid: torch.Tensor,
    axial_grid: torch.Tensor,
    peak_velocity: float,
) -> float:
    r"""Compute the maximum pointwise velocity error against Hagen-Poiseuille flow.

    $$ \varepsilon_\infty = \max_{(r,z)} \abs{\vb{u}_\text{predicted}(r,z) -
    \vb{u}_\text{analytical}(r,z)}, $$

    evaluated over the full Cartesian product of `radial_grid` and
    `axial_grid`. Complements
    [`compute_global_l2_error`][magnetofluidics_pinn.verification.metrics.compute_global_l2_error]:
    a small global $L^2$ error can still hide a large localized deviation
    this maximum would catch.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `radial_grid`, `axial_grid`: 1-D tensors spanning the evaluation grid.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.

    Returns:
    - The scalar maximum pointwise absolute error.

    Raises:
    - `ValueError`: If `radial_grid` or `axial_grid` is not 1-D.
    """
    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    coordinates = _cartesian_grid(
        radial_grid.to(device=resolved_device, dtype=resolved_dtype),
        axial_grid.to(device=resolved_device, dtype=resolved_dtype),
    )
    with torch.no_grad():
        predicted_velocity = network(coordinates)[:, :2]
    analytical_velocity = _analytical_velocity(coordinates, domain.radius, peak_velocity)
    return (predicted_velocity - analytical_velocity).abs().max().item()


def compute_profile_invariance(profile_evaluation: ProfileEvaluation) -> pd.DataFrame:
    r"""Quantify each station's error against the analytical profile and its deviation from the mean.

    $$ \Delta u_z(r, z_k) = u_z^\text{predicted}(r, z_k) - u_z^*(r), $$

    reported per station alongside the deviation from the per-radius mean
    profile across stations — a direct measure of the $z$-independence
    Hagen-Poiseuille flow requires: a perfectly trained network would show
    identical profiles at every station.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].

    Returns:
    - A `pandas.DataFrame` with one row per axial station and columns
      `"z_station"`, `"max_abs_error_vs_analytical"`,
      `"rms_error_vs_analytical"`, and
      `"max_abs_deviation_from_mean_profile"`.
    """
    error = profile_evaluation.predicted_uz - profile_evaluation.analytical_uz.unsqueeze(0)
    mean_profile = profile_evaluation.predicted_uz.mean(dim=0, keepdim=True)
    deviation_from_mean = profile_evaluation.predicted_uz - mean_profile
    records = [
        {
            "z_station": z_station,
            "max_abs_error_vs_analytical": error[index].abs().max().item(),
            "rms_error_vs_analytical": torch.sqrt(torch.mean(error[index] ** 2)).item(),
            "max_abs_deviation_from_mean_profile": deviation_from_mean[index].abs().max().item(),
        }
        for index, z_station in enumerate(profile_evaluation.z_stations)
    ]
    return pd.DataFrame.from_records(records)


def compute_radial_leakage(profile_evaluation: ProfileEvaluation, peak_velocity: float) -> dict[str, float]:
    r"""Quantify radial "leakage" — the spurious $u_r$ a perfectly mass-conserving field would lack.

    Args:
    - `profile_evaluation`: Profile data produced by
      [`evaluate_velocity_profiles`][magnetofluidics_pinn.verification.metrics.evaluate_velocity_profiles].
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$,
      used to normalize the leakage into a dimensionless fraction.

    Returns:
    - A dict with keys `"max_abs_radial_leakage"` (in velocity units) and
      `"normalized_radial_leakage"` (a fraction of `peak_velocity`).

    Raises:
    - `ValueError`: If `peak_velocity` is not finite and strictly positive.
    """
    if not (np.isfinite(peak_velocity) and peak_velocity > 0.0):
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    max_abs_radial_leakage = profile_evaluation.predicted_ur.abs().max().item()
    return {
        "max_abs_radial_leakage": max_abs_radial_leakage,
        "normalized_radial_leakage": max_abs_radial_leakage / peak_velocity,
    }


def compute_flow_rate_curve(
    network: nn.Module,
    domain: Domain,
    axial_grid: torch.Tensor,
    n_quadrature_points: int = 64,
) -> torch.Tensor:
    """Evaluate the predicted volumetric flow rate $Q(z)$ at a set of axial stations.

    Thin, no-grad wrapper around
    [`physics.conservation.axial_flow_rate`][magnetofluidics_pinn.physics.conservation.axial_flow_rate],
    for post-hoc diagnostic use where differentiability is not needed
    (unlike the training-time conservation loss, which calls
    `axial_flow_rate` directly to keep its computation graph).

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `axial_grid`: 1-D tensor of axial stations at which $Q(z)$ is
      evaluated.
    - `n_quadrature_points`: Radial trapezoidal-quadrature resolution per
      station; forwarded to `axial_flow_rate`.

    Returns:
    - Tensor of shape `(axial_grid.numel(),)` with $Q(z)$ at each station.

    Raises:
    - `ValueError`: If `axial_grid` is not a 1-D tensor.
    """
    if axial_grid.ndim != 1:
        raise ValueError(f"axial_grid must be a 1-D tensor; got shape {tuple(axial_grid.shape)}.")
    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    axial_grid = axial_grid.to(device=resolved_device, dtype=resolved_dtype)
    with torch.no_grad():
        return axial_flow_rate(network, domain.radius, axial_grid, n_quadrature_points)


def compute_flow_rate_errors(flow_rate_curve: torch.Tensor, reference_flow_rate: float) -> dict[str, float]:
    r"""Quantify how far a predicted flow-rate curve deviates from its analytical reference.

    Args:
    - `flow_rate_curve`: Tensor of shape `(n_stations,)`, e.g. from
      [`compute_flow_rate_curve`][magnetofluidics_pinn.verification.metrics.compute_flow_rate_curve].
    - `reference_flow_rate`: The analytical reference flow rate
      $Q_\text{ref}$, e.g. from
      [`physics.conservation.poiseuille_reference_flow_rate`][magnetofluidics_pinn.physics.conservation.poiseuille_reference_flow_rate].

    Returns:
    - A dict with keys `"max_abs_flow_rate_error"`,
      `"max_relative_flow_rate_error"`, `"flow_rate_span"` (max - min
      across stations, an internal-consistency check independent of the
      reference value), and `"flow_rate_std"`.

    Raises:
    - `ValueError`: If `reference_flow_rate` is not finite and strictly
      positive.
    """
    if not (np.isfinite(reference_flow_rate) and reference_flow_rate > 0.0):
        raise ValueError(f"reference_flow_rate must be finite and strictly positive; got {reference_flow_rate!r}.")
    deviation = flow_rate_curve.detach() - reference_flow_rate
    max_abs_error = deviation.abs().max().item()
    return {
        "max_abs_flow_rate_error": max_abs_error,
        "max_relative_flow_rate_error": max_abs_error / reference_flow_rate,
        "flow_rate_span": (flow_rate_curve.max() - flow_rate_curve.min()).item(),
        "flow_rate_std": flow_rate_curve.std(unbiased=False).item(),
    }


def evaluate_axis_pressure_profile(
    network: nn.Module,
    domain: Domain,
    n_axial_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a network's predicted pressure along the symmetry axis, $p(r=0, z)$.

    Pressure has no radial dependence in the analytical (or, ideally, the
    trained) solution, so the axis is a representative, singularity-free
    place to sample the axial pressure profile.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `n_axial_points`: Number of evenly-spaced axial samples spanning
      `[0, domain.length]`.

    Returns:
    - A tuple `(axial_positions, predicted_pressure)`, each a 1-D `numpy`
      array of length `n_axial_points`.

    Raises:
    - `ValueError`: If `n_axial_points` is fewer than 2 (a line cannot be
      fit through fewer than two points).
    """
    if n_axial_points < 2:
        raise ValueError(f"n_axial_points must be at least 2 to fit a line; got {n_axial_points!r}.")
    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    axial_positions = torch.linspace(
        0.0, domain.length, n_axial_points, device=resolved_device, dtype=resolved_dtype
    )
    coordinates = torch.stack([torch.zeros_like(axial_positions), axial_positions], dim=1)
    with torch.no_grad():
        predicted_pressure = network(coordinates)[:, 2]
    return axial_positions.cpu().numpy(), predicted_pressure.cpu().numpy()


def compute_pressure_gradient_error(
    network: nn.Module,
    domain: Domain,
    peak_velocity: float,
    n_axial_points: int = 50,
) -> dict[str, float]:
    r"""Fit the network's axial pressure profile and compare its slope to the analytical gradient.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$.
    - `n_axial_points`: Forwarded to
      [`evaluate_axis_pressure_profile`][magnetofluidics_pinn.verification.metrics.evaluate_axis_pressure_profile].

    Returns:
    - A dict with keys `"predicted_pressure_gradient"`,
      `"fitted_intercept"`, `"analytical_pressure_gradient"`,
      `"absolute_error"`, and `"relative_error"`.
    """
    axial_positions, predicted_pressure = evaluate_axis_pressure_profile(network, domain, n_axial_points)
    fitted_slope, fitted_intercept = np.polyfit(axial_positions, predicted_pressure, deg=1)
    analytical_slope = analytical_pressure_gradient(domain.radius, peak_velocity)
    absolute_error = abs(fitted_slope - analytical_slope)
    return {
        "predicted_pressure_gradient": float(fitted_slope),
        "fitted_intercept": float(fitted_intercept),
        "analytical_pressure_gradient": float(analytical_slope),
        "absolute_error": float(absolute_error),
        "relative_error": float(absolute_error / abs(analytical_slope)),
    }


def evaluate_held_out_residual(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    random_seed: int,
    n_points: int,
    residual_form: Literal["standard", "r_weighted"] = "standard",
    axis_clearance_fraction: float | None = None,
) -> pd.DataFrame:
    """Evaluate the Stokes residual on a fresh, independent collocation batch.

    Draws interior points under the same volume-uniform measure
    [`sampling.collocation.sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]
    uses during training, with a `random_seed` the caller controls
    independently of every seed used during training itself — a genuine
    held-out check rather than a re-evaluation of the training batch.

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `random_seed`: Seed for the held-out collocation batch; should differ
      from every seed used during training.
    - `n_points`: Number of interior points to draw.
    - `residual_form`: `"standard"` or `"r_weighted"`; see
      [`physics.fluid_residuals.stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual].
    - `axis_clearance_fraction`: Forwarded to `sample_collocation_points`;
      pass the same value used during training for an apples-to-apples
      comparison.

    Returns:
    - A `pandas.DataFrame`, as returned by
      [`summarize_residual`][magnetofluidics_pinn.verification.metrics.summarize_residual].

    Raises:
    - `ValueError`: If `n_points` is not strictly positive.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be strictly positive; got {n_points!r}.")
    resolved_device = resolve_module_device(network)
    # `sample_collocation_points` also draws a small boundary batch; it is
    # requested here (minimum size 3) purely to satisfy that function's own
    # `n_boundary > 0` contract, and discarded immediately below — only the
    # interior points, drawn under the same volume-uniform measure training
    # uses, are relevant to a held-out interior-residual check.
    collocation = sample_collocation_points(
        domain=domain, n_interior=n_points, n_boundary=3, random_seed=random_seed,
        axis_clearance_fraction=axis_clearance_fraction, device=resolved_device,
    )
    interior = collocation.interior.clone().requires_grad_(True)
    residual = stokes_residual(network, interior, fluid_config, residual_form=residual_form)
    return summarize_residual(residual, _RESIDUAL_COMPONENT_NAMES)


def evaluate_residual_grid(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    radial_grid: torch.Tensor,
    axial_grid: torch.Tensor,
    residual_form: Literal["standard", "r_weighted"] = "standard",
) -> dict[str, torch.Tensor]:
    """Evaluate the Stokes residual over a structured `(r, z)` grid, for visualization.

    Unlike
    [`evaluate_held_out_residual`][magnetofluidics_pinn.verification.metrics.evaluate_held_out_residual],
    which draws scattered points under the training-time sampling measure,
    this function evaluates on a regular grid suitable for
    [`visualization.plotting.plot_residual_heatmap`][magnetofluidics_pinn.visualization.plotting.plot_residual_heatmap].

    Args:
    - `network`: Trained flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Already-nondimensionalized vessel geometry.
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `radial_grid`, `axial_grid`: 1-D tensors spanning the evaluation grid.
      Under `residual_form="standard"`, `radial_grid` must not touch or
      cross the symmetry axis (see
      [`physics.fluid_residuals.stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]).
    - `residual_form`: `"standard"` or `"r_weighted"`.

    Returns:
    - A dict mapping each of `"momentum_r"`, `"momentum_z"`,
      `"continuity"` to a `(radial_grid.numel(), axial_grid.numel())`
      tensor of the residual's absolute value.

    Raises:
    - `ValueError`: If `residual_form == "standard"` and `radial_grid`
      contains a value `<= 0.0`.
    """
    if residual_form == "standard" and torch.any(radial_grid <= 0.0):
        raise ValueError(
            "evaluate_residual_grid received a radial_grid touching or crossing the symmetry "
            "axis (r <= 0) under residual_form='standard'; exclude r=0 from radial_grid, or use "
            "residual_form='r_weighted'."
        )
    resolved_device = resolve_module_device(network)
    resolved_dtype = resolve_module_dtype(network)
    radial_grid = radial_grid.to(device=resolved_device, dtype=resolved_dtype)
    axial_grid = axial_grid.to(device=resolved_device, dtype=resolved_dtype)
    coordinates = _cartesian_grid(radial_grid, axial_grid).requires_grad_(True)
    residual = stokes_residual(network, coordinates, fluid_config, residual_form=residual_form)
    n_radial, n_axial = radial_grid.shape[0], axial_grid.shape[0]
    return {
        name: residual[:, index].detach().abs().reshape(n_radial, n_axial)
        for index, name in enumerate(_RESIDUAL_COMPONENT_NAMES)
    }


def compare_trajectory_to_analytical(
    trajectory: list[ParticleState],
    initial_position: tuple[float, float],
    peak_velocity: float,
    radius: float,
    particle_radius: float = 0.0,
) -> pd.DataFrame:
    """Compare an integrated particle trajectory against its closed-form reference.

    Args:
    - `trajectory`: A single particle's trajectory, as returned by one
      entry of
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]'s
      return value.
    - `initial_position`: Tuple $(r_0^*, z_0^*)$ matching the trajectory's
      first state.
    - `peak_velocity`: Dimensionless centerline velocity $u_\\text{peak}^*$.
    - `radius`: Dimensionless channel radius $R^*$.
    - `particle_radius`: Dimensionless particle radius $a^*$; forwarded to
      [`verification.analytical.analytical_tracer_streamline`][magnetofluidics_pinn.verification.analytical.analytical_tracer_streamline].

    Returns:
    - A `pandas.DataFrame` with one row per recorded state and columns
      `"time"`, `"predicted_r"`, `"analytical_r"`, `"radial_error"`,
      `"predicted_z"`, `"analytical_z"`, `"axial_error"`.

    Raises:
    - `ValueError`: If `trajectory` is empty.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one ParticleState.")
    time_grid = np.array([state.time for state in trajectory], dtype=np.float64)
    analytical_r, analytical_z = analytical_tracer_streamline(
        initial_position=initial_position, peak_velocity=peak_velocity, radius=radius,
        time_grid=time_grid, initial_time=float(trajectory[0].time), particle_radius=particle_radius,
    )
    predicted_r = np.array([state.position[0].item() for state in trajectory])
    predicted_z = np.array([state.position[1].item() for state in trajectory])
    records = [
        {
            "time": t, "predicted_r": pr, "analytical_r": ar, "radial_error": abs(pr - ar),
            "predicted_z": pz, "analytical_z": az, "axial_error": abs(pz - az),
        }
        for t, pr, ar, pz, az in zip(time_grid, predicted_r, analytical_r, predicted_z, analytical_z)
    ]
    return pd.DataFrame.from_records(records)
