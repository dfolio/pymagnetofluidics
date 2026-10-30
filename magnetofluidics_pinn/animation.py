r"""Quasi-static free-particle trajectories and flow animation.

The particle is free to move with the flow: no-slip remains the kinematic
interface condition $\mathbf{u}|_S = (0, U_p)$ for any $U_p$, and the freedom
is closed by the overdamped force balance
$$F_{\mathrm{applied}}(z_p) + F_0 + U_p\,F_1 = 0
\iff U_p = -\frac{F_{\mathrm{applied}} + F_0}{F_1},$$
exact at zero Reynolds number [@maxey1983equation;@brady1988stokesiana].
Because steady Stokes flow is linear in the boundary data, the field for any
$(u_{\max}, U_p)$ is the superposition
$$\mathbf{u} = u_{\max}\,\Phi(1, 0) + U_p\,\Phi(0, 1),$$
so the two cached resistance networks render the free-particle field without
retraining. Transferring the raw corrections to a new position uses the
translation-invariant network input $(s/R^2, (z - z_p)/R)$; this is accurate in
the channel's central region and degrades near the inlet and outlet.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from magnetofluidics_pinn.networks.stream_function import (
    apply_stream_function_constraint,
)


@dataclass(frozen=True)
class FreeParticleTrajectory:
    """Immutable quasi-static trajectory of the free particle.

    Args:
    - `times`: Time samples, shape `(n,)`, nondimensional.
    - `positions`: Axial positions $z_p(t)$, shape `(n,)`.
    - `velocities`: Balanced velocities $U_p(t)$, shape `(n,)`.
    - `applied_forces`: External axial forces $F_{\mathrm{applied}}(z_p)$, shape `(n,)`.
    - `exited_domain`: Whether the particle left the admissible span before `max_steps`.
    """

    times: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    applied_forces: torch.Tensor
    exited_domain: bool


def free_particle_velocity(resistance, applied_force: float) -> float:
    """Return the force-balanced velocity $U_p = -(F_{\mathrm{applied}} + F_0)/F_1$.

    Args:
    - `resistance`: `ResistanceResult` carrying $F_0$ and $F_1$.
    - `applied_force`: External axial force at the current position.

    Returns:
    - Balanced velocity $U_p$.

    Raises:
    - `ValueError`: If `applied_force` is not finite, or via `balanced_velocity`
      if the resistance data is non-dissipative.
    """
    if not math.isfinite(applied_force):
        raise ValueError(f"applied_force must be finite; got {applied_force!r}.")
    return resistance.balanced_velocity(applied_force)


def free_particle_trajectory(
    resistance,
    applied_force_fn: Callable[[float], float],
    axial_span: tuple[float, float],
    time_step: float,
    max_steps: int,
    initial_position: float | None = None,
) -> FreeParticleTrajectory:
    """Integrate the overdamped trajectory $\mathrm{d}z_p/\mathrm{d}t = U_p(z_p)$ with RK4.

    Args:
    - `resistance`: `ResistanceResult` carrying $F_0$ and $F_1$.
    - `applied_force_fn`: Pure callable $z_p \mapsto F_{\mathrm{applied}}(z_p)$.
    - `axial_span`: Admissible open interval for $z_p$.
    - `time_step`: Fixed nondimensional time step, strictly positive.
    - `max_steps`: Upper bound on integration steps.
    - `initial_position`: Starting position; defaults to the span's lower bound.

    Returns:
    - Frozen `FreeParticleTrajectory` with the sampled states.

    Raises:
    - `ValueError`: On invalid span, step, or non-finite force input.
    """
    z_low, z_high = (float(axial_span[0]), float(axial_span[1]))
    if not z_low < z_high:
        raise ValueError(f"axial_span must be increasing; got {axial_span!r}.")
    if time_step <= 0.0:
        raise ValueError(f"time_step must be strictly positive; got {time_step!r}.")

    def velocity_at(z: float) -> float:
        return free_particle_velocity(resistance, float(applied_force_fn(z)))

    position = float(z_low if initial_position is None else initial_position)
    times: list[float] = [0.0]
    positions: list[float] = [position]
    velocities: list[float] = [velocity_at(position)]
    forces: list[float] = [float(applied_force_fn(position))]
    exited = False

    for step in range(max_steps):
        if not z_low < position < z_high:
            exited = True
            break
        t = step * time_step
        k1 = velocity_at(position)
        k2 = velocity_at(position + 0.5 * time_step * k1)
        k3 = velocity_at(position + 0.5 * time_step * k2)
        k4 = velocity_at(position + time_step * k3)
        position = position + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        times.append(t + time_step)
        positions.append(position)
        velocities.append(velocity_at(position))
        forces.append(float(applied_force_fn(position)))

    to_tensor = torch.tensor  # CPU time series; field work stays on CUDA.
    return FreeParticleTrajectory(
        times=to_tensor(times, dtype=torch.float64),
        positions=to_tensor(positions, dtype=torch.float64),
        velocities=to_tensor(velocities, dtype=torch.float64),
        applied_forces=to_tensor(forces, dtype=torch.float64),
        exited_domain=exited,
    )


class SuperposedFlow(nn.Module):
    """Field module for the free-particle superposition $u_{\max}\Phi_A + U_p\Phi_B$.

    Both wrapped networks enforce their rigid-body face exactly, so the
    superposition enforces $\mathbf{u}|_S = (0, U_p)$ by linearity. Both gauges
    anchor at $(0, L)$, so the superposed pressure keeps the same gauge.

    Args:
    - `network_ambient`: Trained $\Phi(1, 0)$ wrapper (fixed sphere, ambient flow).
    - `network_translating`: Trained $\Phi(0, 1)$ wrapper (unit translation, quiescent).
    - `ambient_amplitude`: Background amplitude $u_{\max}$.
    - `particle_velocity`: Free-particle velocity $U_p$.
    """

    def __init__(
        self,
        network_ambient: nn.Module,
        network_translating: nn.Module,
        ambient_amplitude: float,
        particle_velocity: float,
    ) -> None:
        super().__init__()
        self.network_ambient = network_ambient
        self.network_translating = network_translating
        self.ambient_amplitude = float(ambient_amplitude)
        self.particle_velocity = float(particle_velocity)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the superposed $(u_r, u_z, p)$ field at `coordinates`."""
        return (
            self.ambient_amplitude * self.network_ambient(coordinates)
            + self.particle_velocity * self.network_translating(coordinates)
        )


def rebuild_at_position(
    network: nn.Module,
    domain,
    particle_axial_position: float,
    particle_radius: float,
    *,
    particle_axial_velocity: float,
    flow_amplitude: float,
    taper_margin: float,
) -> nn.Module:
    """Re-wrap a trained raw MLP at a new position without mutating `network`.

    The analytic lifting is rebuilt at the new geometry; the learned raw
    correction transfers through the translation-invariant input.

    Args:
    - `network`: Trained `StreamFunctionSphereFlow` wrapper.
    - `domain`: Nondimensional channel geometry.
    - `particle_axial_position`: New sphere center $z_p$.
    - `particle_radius`: Sphere radius $a$.
    - `particle_axial_velocity`: Sphere velocity $U_p$ for this sub-problem.
    - `flow_amplitude`: Background amplitude for this sub-problem.
    - `taper_margin`: Taper support margin, identical to training.

    Returns:
    - A fresh wrapper; `network` is left untouched.

    Raises:
    - `AttributeError`: If `network` exposes no `raw_network`.
    """
    raw = getattr(network, "raw_network")
    if raw is None:
        raise AttributeError("network exposes no raw_network to transfer.")
    return apply_stream_function_constraint(
        raw, domain, particle_axial_position, particle_radius,
        particle_axial_velocity=particle_axial_velocity,
        flow_amplitude=flow_amplitude,
        taper_margin=taper_margin,
    )


def superposed_flow_at(
    network_ambient: nn.Module,
    network_translating: nn.Module,
    domain,
    particle_axial_position: float,
    particle_radius: float,
    *,
    ambient_amplitude: float,
    particle_velocity: float,
    taper_margin: float,
) -> SuperposedFlow:
    """Assemble the free-particle field at $(z_p, U_p)$ from the cached pair.

    Args:
    - `network_ambient`, `network_translating`: Trained resistance wrappers.
    - `domain`: Nondimensional channel geometry.
    - `particle_axial_position`: Sphere center $z_p$ for this frame.
    - `particle_radius`: Sphere radius $a$.
    - `ambient_amplitude`: Background amplitude $u_{\max}$.
    - `particle_velocity`: Free-particle velocity $U_p$ for this frame.
    - `taper_margin`: Taper support margin, identical to training.

    Returns:
    - `SuperposedFlow` evaluating $(u_r, u_z, p)$ at the requested geometry.
    """
    rebuilt_ambient = rebuild_at_position(
        network_ambient, domain, particle_axial_position, particle_radius,
        particle_axial_velocity=0.0, flow_amplitude=ambient_amplitude,
        taper_margin=taper_margin,
    )
    rebuilt_translating = rebuild_at_position(
        network_translating, domain, particle_axial_position, particle_radius,
        particle_axial_velocity=1.0, flow_amplitude=0.0,
        taper_margin=taper_margin,
    )
    return SuperposedFlow(
        rebuilt_ambient, rebuilt_translating, ambient_amplitude, particle_velocity,
    )
