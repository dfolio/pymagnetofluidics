r"""Resistance-pair mobility model replacing the `brentq` retraining loop.

Stokes flow is linear, so the axial force on an on-axis sphere is exactly

$$ F(U_p) = F_A + U_p\,F_B, $$

where $F_A$ is the force on the fixed sphere in the ambient flow and $F_B < 0$
is the force per unit velocity of the sphere translating in still fluid
[@happel1983low]. Two trainings therefore give the closed form
$U_p = -(F_A + F_\text{applied})/F_B$. The pair depends on $z_p$ only through
end effects, so it is reused along the trajectory away from the ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable

import torch
from torch import nn
import pandas as pd  # NEW

from magnetofluidics_pinn.config import FluidConfig, TrainingConfig
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import apply_stream_function_constraint
from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory, train_stream_function_flow
from magnetofluidics_pinn.types import Domain, ParticleState
from magnetofluidics_pinn.verification.force_checks import sphere_traction_force


@dataclass(frozen=True)
class ResistancePair:
    r"""Resistance pair $(F_A, F_B)$.

    Args:
    - `ambient_force`: $F_A$, the force on the fixed sphere in the ambient flow.
    - `translation_force`: $F_B < 0$, the force per unit sphere velocity in still fluid.

    Raises:
    - `ValueError`: If a value is not finite or `translation_force >= 0`.
    """

    ambient_force: float
    translation_force: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ambient_force) and math.isfinite(self.translation_force)):
            raise ValueError("Both resistances must be finite.")
        if self.translation_force >= 0.0:
            raise ValueError("translation_force must be strictly negative (it is a drag).")

    @property
    def mobility(self) -> float:
        r"""Return $M = 1/\abs{F_B}$, the velocity per unit applied force."""
        return -1.0 / self.translation_force

    @property
    def free_velocity(self) -> float:
        r"""Return the passive velocity $U_0 = -F_A / F_B$."""
        return -self.ambient_force / self.translation_force


def force_balanced_velocity(pair: ResistancePair, applied_force: float) -> float:
    r"""Return $U_p = -(F_A + F_\text{applied})/F_B$.

    Args:
    - `pair`: The resistance pair.
    - `applied_force`: Total external axial force.

    Returns:
    - The force-balanced axial velocity.

    Raises:
    - `ValueError`: If `applied_force` is not finite.
    """
    if not math.isfinite(applied_force):
        raise ValueError(f"applied_force must be finite; got {applied_force!r}.")
    return pair.free_velocity + pair.mobility * applied_force


@dataclass(frozen=True)
class ResistanceSolution:
    """Result of [`solve_resistance_pair`][magnetofluidics_pinn.trajectory.mobility.solve_resistance_pair].

    Args:
    - `pair`: Resistance pair averaged over the control spheres.
    - `ambient_network`, `translation_network`: The two trained networks.
    - `ambient_history`, `translation_history`: Training histories.
    - `force_spread`: Relative max-min spread of `(F_A, F_B)` over control radii. Large
      values mean the field is not a Stokes solution, so the pair is unreliable.
    """

    pair: ResistancePair
    ambient_network: nn.Module
    translation_network: nn.Module
    ambient_history: StreamLossHistory
    translation_history: StreamLossHistory
    force_spread: tuple[float, float]


def _averaged_force(network: nn.Module, z_p: float, radii: tuple[float, ...], n_quadrature: int) -> tuple[float, float]:
    """Average the traction force over control radii.

    Args:
    - `network`: Trained flow field.
    - `z_p`: Sphere center.
    - `radii`: Control radii.
    - `n_quadrature`: Polar nodes.

    Returns:
    - The pair `(mean_force, relative_spread)`.
    """
    forces = tuple(sphere_traction_force(network, z_p, radius, n_quadrature) for radius in radii)
    mean = sum(forces) / len(forces)
    return mean, (max(forces) - min(forces)) / max(abs(mean), 1.0e-12)


def solve_resistance_pair(
    domain: Domain,
    fluid_config: FluidConfig,
    particle_radius: float,
    particle_axial_position: float,
    training_config: TrainingConfig,
    hidden_layers: tuple[int, ...] = (64, 64, 64),
    activation: str = "tanh",
    control_radius_factors: tuple[float, ...] = (1.25, 1.5, 2.0),
    n_quadrature_points: int = 361,
) -> ResistanceSolution:
    """Train the ambient and translation problems and extract $(F_A, F_B)$.

    Args:
    - `domain`: Nondimensional channel; `domain.u_max` sets the ambient amplitude.
    - `fluid_config`: Stokes fluid configuration.
    - `particle_radius`: Sphere radius.
    - `particle_axial_position`: Sphere center.
    - `training_config`: Budget, device and dtype for both trainings.
    - `hidden_layers`, `activation`: MLP architecture.
    - `control_radius_factors`: Control radii in units of `particle_radius`; factors that do
      not fit inside the channel are dropped.
    - `n_quadrature_points`: Polar nodes of each traction integral.

    Returns:
    - A [`ResistanceSolution`][magnetofluidics_pinn.trajectory.mobility.ResistanceSolution].

    Raises:
    - `ValueError`: If no control radius fits inside the domain.
    """
    room = 0.95 * min(domain.radius, particle_axial_position, domain.length - particle_axial_position)
    radii = tuple(f * particle_radius for f in control_radius_factors if f > 1.0 and f * particle_radius < room)
    if not radii:
        raise ValueError("No control radius fits inside the domain; use smaller factors.")

    def solve(u_particle: float, amplitude: float) -> tuple[nn.Module, StreamLossHistory]:
        raw = build_mlp(2, 2, hidden_layers, training_config=training_config, activation=activation)
        wrapped = apply_stream_function_constraint(
            raw, domain, particle_axial_position, particle_radius, u_particle, amplitude)
        return train_stream_function_flow(
            wrapped, domain, fluid_config, training_config,
            particle_axial_position, particle_radius, amplitude)

    ambient_net, ambient_hist = solve(0.0, domain.u_max)
    translation_net, translation_hist = solve(1.0, 0.0)
    f_a, spread_a = _averaged_force(ambient_net, particle_axial_position, radii, n_quadrature_points)
    f_b, spread_b = _averaged_force(translation_net, particle_axial_position, radii, n_quadrature_points)
    return ResistanceSolution(ResistancePair(f_a, f_b), ambient_net, translation_net,
                              ambient_hist, translation_hist, (spread_a, spread_b))


def advance_particle_resistance(
    pair: ResistancePair,
    applied_force_fn: Callable[[float], float],
    initial_axial_position: float,
    n_steps: int,
    time_step: float,
    admissible_range: tuple[float, float],
) -> list[ParticleState]:
    """Advance the particle with the midpoint (RK2) rule using the force-balanced velocity.

    Args:
    - `pair`: The resistance pair, valid away from the channel ends.
    - `applied_force_fn`: Maps axial position to the total external axial force.
    - `initial_axial_position`: Starting axial position.
    - `n_steps`: Number of steps, at least 1.
    - `time_step`: Dimensionless step, strictly positive.
    - `admissible_range`: `(z_min, z_max)` over which the pair is trusted.

    Returns:
    - `n_steps + 1` on-axis `ParticleState` objects with float64 CPU tensors.

    Raises:
    - `ValueError`: If `n_steps < 1`, `time_step <= 0`, or the range is invalid.
    - `RuntimeError`: If the particle leaves `admissible_range`.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if not (math.isfinite(time_step) and time_step > 0.0):
        raise ValueError("time_step must be finite and strictly positive.")
    z_min, z_max = admissible_range
    if not z_min < z_max:
        raise ValueError("admissible_range must satisfy z_min < z_max.")

    def velocity(z: float) -> float:
        return force_balanced_velocity(pair, applied_force_fn(z))

    def step(state: tuple[float, float, float], _: int) -> tuple[float, float, float]:
        z, t, _u = state
        if not z_min < z < z_max:
            raise RuntimeError(f"Particle left the admissible range [{z_min}, {z_max}] at z={z}.")
        midpoint = z + 0.5 * time_step * velocity(z)
        u_mid = velocity(midpoint)
        return z + time_step * u_mid, t + time_step, u_mid

    states = accumulate(range(n_steps), step, initial=(initial_axial_position, 0.0, velocity(initial_axial_position)))
    return [
        ParticleState(
            position=torch.tensor([0.0, z], dtype=torch.float64),
            velocity=torch.tensor([0.0, u], dtype=torch.float64),
            time=t,
        )
        for z, t, u in states
    ]


def resistance_position_sweep(
    domain: Domain,
    fluid_config: FluidConfig,
    particle_radius: float,
    positions: tuple[float, ...],
    training_config: TrainingConfig,
    **solver_kwargs,
) -> pd.DataFrame:
    """Solve the resistance pair at several axial positions.

    Used to test the assumption behind `advance_particle_resistance`, that $(F_A, F_B)$ does not
    depend on $z_p$ away from the channel ends.

    Args:
    - `domain`: Nondimensional channel.
    - `fluid_config`: Stokes fluid configuration.
    - `particle_radius`: Sphere radius.
    - `positions`: Sphere centers to test (at least two).
    - `training_config`: Budget, device and dtype for each solve.
    - `**solver_kwargs`: Forwarded to `solve_resistance_pair`.

    Returns:
    - A `DataFrame` with columns `z_p`, `F_A`, `F_B`, `spread_A`, `spread_B`, `U_free`.

    Raises:
    - `ValueError`: If fewer than two positions are given.
    """
    if len(positions) < 2:
        raise ValueError("At least two positions are required for a sweep.")
    solutions = tuple(
        solve_resistance_pair(domain, fluid_config, particle_radius, z, training_config, **solver_kwargs)
        for z in positions
    )
    return pd.DataFrame({
        "z_p": positions,
        "F_A": [s.pair.ambient_force for s in solutions],
        "F_B": [s.pair.translation_force for s in solutions],
        "spread_A": [s.force_spread[0] for s in solutions],
        "spread_B": [s.force_spread[1] for s in solutions],
        "U_free": [s.pair.free_velocity for s in solutions],
    })


def position_independence(table: pd.DataFrame) -> float:
    """Return the largest relative variation of `F_A` and `F_B` across positions.

    Args:
    - `table`: Output of `resistance_position_sweep`.

    Returns:
    - $\\max_{k\\in\\{A,B\\}} (\\max F_k - \\min F_k)/\\abs{\\overline{F_k}}$.

    Raises:
    - `ValueError`: If `F_A` or `F_B` is missing, or a column has zero mean.
    """
    if not {"F_A", "F_B"}.issubset(table.columns):
        raise ValueError("table must contain the columns 'F_A' and 'F_B'.")
    variations = []
    for name in ("F_A", "F_B"):
        column = table[name]
        mean = column.mean()
        if mean == 0.0:
            raise ValueError(f"{name} has zero mean; relative variation is undefined.")
        variations.append((column.max() - column.min()) / abs(mean))
    return float(max(variations))
