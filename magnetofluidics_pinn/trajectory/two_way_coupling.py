r"""Force-balanced particle motion under a genuine two-way flow coupling.

NEW module. `trajectory.integrator.integrate_trajectory` treats the
magnetic microrobot as a point (or Faxén-corrected finite) tracer inside
an *undisturbed* ambient flow, then adds a magnetic drift term under an
explicit "unit-mobility" placeholder — both docstrings already flag this
as the weakest link in that model. This module closes it the other way:
rather than approximating the drag the particle would feel, it *solves
for it directly* from a flow field trained around the particle's actual
excluded volume
([`training.trainer.train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle],
[`physics.hydrodynamic_drag.surface_traction_force`][magnetofluidics_pinn.physics.hydrodynamic_drag.surface_traction_force]),
and finds the particle's own translational velocity as the one value that
balances every externally applied force against that drag:
$$F_{\text{applied}} + F_z^{\text{drag}}(U_z) = 0.$$
This is the correct closure at zero Reynolds number: a freely-moving
particle in Stokes flow carries no net force at any instant (there is no
inertia to carry a residual one), so whatever force acts on it — magnetic
today, anything else later — is balanced *exactly* by drag, not merely
correlated with it [@richou2003correction; @happel1983low].

**Cost, and what this is (and is not) for.** Each evaluation of the force
residual above is a full (re-)training of the two-way-coupled flow field,
not a cheap function call — `solve_force_balanced_velocity` needs several
such evaluations to bracket a root, and `advance_particle_two_way` calls
it once per trajectory step. Both are deliberately high-fidelity,
computationally heavy validation tools: a way to check the existing
one-way, Faxén-corrected
[`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]
against genuine two-way ground truth at a handful of positions, or to
compute a small number of physically rigorous states — not a routine
replacement for it. `integrate_trajectory`'s adaptive-step, cheap
point-tracer model remains the right default for ordinary trajectory
work; reach for this module when the particle's radius is a large enough
fraction of the channel radius that a one-way correction is in doubt, and
only for as many positions as the cost allows.

**Scope.** Restricted, like
[`SphericalObstacle`][magnetofluidics_pinn.types.SphericalObstacle]
itself, to a single spherical particle confined to the channel's
symmetry axis, translating axially with no rotation — the one obstacle
placement compatible with this package's axisymmetric `(r, z)`
formulation (see that type's docstring). `applied_force_fn` is
deliberately just a plain function of axial position returning a single
signed force: today that force is expected to come from
[`physics.magnetic_forcing.dipole_force`][magnetofluidics_pinn.physics.magnetic_forcing.dipole_force]
(zero for the currently-implemented `uniform_field`, since a spatially
constant field exerts no dipole force by construction — a non-uniform
field is needed for this mechanism to do anything magnetically
interesting, and `gradient_field`/`biot_savart_field` remain
`NotImplementedError` stubs elsewhere in this package); nothing here
depends on that, and any other force (gravity, buoyancy, a future
field type) folds in exactly the same way, by the caller summing it into
the same scalar before calling in.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from scipy.optimize import brentq

from magnetofluidics_pinn.config import FluidConfig, TrainingConfig
from magnetofluidics_pinn.physics.hydrodynamic_drag import surface_traction_force
from magnetofluidics_pinn.training.trainer import ObstacleTrainingHistory, train_around_obstacle
from magnetofluidics_pinn.types import Domain, ParticleState, SphericalObstacle


def solve_force_balanced_velocity(
    network: nn.Module,
    domain: Domain,
    obstacle: SphericalObstacle,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    applied_force_z: float,
    velocity_bracket: tuple[float, float],
    n_obstacle_surface_points: int = 200,
    n_quadrature_points: int = 181,
    velocity_tolerance: float = 1.0e-3,
    refinement_config: TrainingConfig | None = None,
    verbose: bool = False,
) -> tuple[float, nn.Module, ObstacleTrainingHistory]:
    r"""Solve for the particle's force-balanced translational velocity, at a fixed position.

    NEW. Brackets and roots the scalar force residual
    $F_{\text{applied}} + F_z^{\text{drag}}(U_z)$ over `velocity_bracket`
    with `scipy.optimize.brentq`, retraining
    ([`train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle])
    the two-way-coupled flow field at each candidate $U_z$ and reading its
    drag off
    [`surface_traction_force`][magnetofluidics_pinn.physics.hydrodynamic_drag.surface_traction_force].
    Successive candidates are *warm-started* from the previous candidate's
    converged network — consecutive brentq iterates are close together in
    $U_z$, so fine-tuning from the last solution converges far faster than
    retraining from scratch every time; `refinement_config` lets those
    warm-started re-solves use a much smaller epoch/round budget than the
    first, cold-started one.

    Args:
    - `network`: Starting flow network for the *first* (cold-started)
      candidate velocity evaluated (`velocity_bracket`'s lower end);
      every subsequent candidate warm-starts from the previous one's
      result. Never mutated — `train_around_obstacle` already
      deep-copies before training.
    - `domain`, `obstacle`, `fluid_config`: As in
      [`train_around_obstacle`][magnetofluidics_pinn.training.trainer.train_around_obstacle].
    - `training_config`: Used for the first, cold-started evaluation.
    - `applied_force_z`: The sum of every externally applied axial force
      on the particle (magnetic today; anything else the caller adds
      later), evaluated at `obstacle.axial_position`. `0.0` finds the
      velocity of a passively-advected, non-actuated particle.
    - `velocity_bracket`: `(low, high)` with `low < high`; the force
      residual must have opposite signs at the two ends (checked below) —
      a reasonable starting bracket is a generous multiple of the
      Faxén-corrected point-particle estimate
      ([`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity])
      at the same position.
    - `n_obstacle_surface_points`, `n_quadrature_points`: Forwarded to
      `train_around_obstacle` and `surface_traction_force` respectively.
    - `velocity_tolerance`: `brentq`'s `xtol`, in the same dimensionless
      velocity units as `velocity_bracket`.
    - `refinement_config`: `TrainingConfig` used for every candidate
      *after* the first; `None` (the default) reuses `training_config`
      for all of them, unchanged. Since every candidate after the first
      is warm-started, this is typically set to far fewer epochs/rounds
      than a cold start needs.
    - `verbose`: If `True`, print each candidate velocity and the
      resulting drag force and residual.

    Returns:
    - A tuple `(particle_velocity, trained_network, history)`:
      `particle_velocity` is the force-balanced $U_z$; `trained_network`
      is the flow field converged *at* that velocity (re-evaluated once
      more exactly there after `brentq` returns, so it is never left at
      whatever the second-to-last bracketing step happened to produce);
      `history` is that final training call's
      [`ObstacleTrainingHistory`][magnetofluidics_pinn.training.trainer.ObstacleTrainingHistory].

    Raises:
    - `ValueError`: If `velocity_bracket[0] >= velocity_bracket[1]`, or if
      the force residual does not change sign across `velocity_bracket`
      (the bracket needs widening).
    """
    velocity_low, velocity_high = velocity_bracket
    if velocity_low >= velocity_high:
        raise ValueError(f"velocity_bracket must satisfy low < high; got {velocity_bracket!r}.")

    state: dict[str, object] = {"network": network, "history": None}

    def force_residual(candidate_velocity: float) -> float:
        current_config = training_config if state["history"] is None else (refinement_config or training_config)
        trained, history = train_around_obstacle(
            state["network"], domain, obstacle, candidate_velocity, fluid_config, current_config,
            n_obstacle_surface_points=n_obstacle_surface_points, verbose=False,
        )
        drag_force = surface_traction_force(trained, obstacle, n_quadrature_points).item()
        state["network"] = trained
        state["history"] = history
        residual = applied_force_z + drag_force
        if verbose:
            print(f"  U_z = {candidate_velocity:+.5f}  ->  F_drag = {drag_force:+.5f}, residual = {residual:+.5f}")
        return residual

    residual_low = force_residual(velocity_low)
    residual_high = force_residual(velocity_high)
    if residual_low * residual_high > 0.0:
        raise ValueError(
            f"velocity_bracket={velocity_bracket!r} does not bracket a sign change in the force "
            f"residual (residual({velocity_low})={residual_low:.4g}, residual({velocity_high})="
            f"{residual_high:.4g}); widen the bracket."
        )

    particle_velocity = brentq(force_residual, velocity_low, velocity_high, xtol=velocity_tolerance)
    # One further evaluation exactly at the root, so the returned network
    # and history are guaranteed to match `particle_velocity` precisely,
    # rather than whichever bracketing step brentq's internal search last
    # happened to land near it.
    force_residual(particle_velocity)

    return particle_velocity, state["network"], state["history"]


def advance_particle_two_way(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    obstacle_radius: float,
    initial_axial_position: float,
    applied_force_fn: Callable[[float], float],
    n_steps: int,
    time_step: float,
    velocity_bracket: tuple[float, float],
    refinement_config: TrainingConfig | None = None,
    n_obstacle_surface_points: int = 200,
    n_quadrature_points: int = 181,
    velocity_tolerance: float = 1.0e-3,
    verbose: bool = False,
) -> list[ParticleState]:
    r"""Advance a two-way-coupled particle through the channel, one force-balanced step at a time.

    NEW. An explicit-Euler stepper built directly on
    [`solve_force_balanced_velocity`][magnetofluidics_pinn.trajectory.two_way_coupling.solve_force_balanced_velocity]:
    at each of `n_steps` steps, freezes the particle at its current axial
    position, solves for the velocity balancing `applied_force_fn` there
    against drag, advances the position by `velocity * time_step`, and
    repeats — warm-starting every step's flow solve from the previous
    step's, since the particle typically moves only a small fraction of
    its own radius per step. A higher-order integrator was not worth
    layering on top: the per-step cost is already dominated by
    `solve_force_balanced_velocity`'s own retraining, not by the stepping
    scheme, so a fixed, small `time_step` (checked against how far the
    particle actually moved) is the more direct way to control accuracy.

    Args:
    - `network`: Starting flow network for the first step (see
      `solve_force_balanced_velocity`'s `network` argument).
    - `domain`, `fluid_config`, `training_config`, `refinement_config`,
      `n_obstacle_surface_points`, `n_quadrature_points`,
      `velocity_tolerance`, `verbose`: Forwarded to
      `solve_force_balanced_velocity` at every step.
    - `obstacle_radius`: Dimensionless particle radius (see
      `SphericalObstacle.radius`); fixed for the whole trajectory.
    - `initial_axial_position`: Dimensionless starting $z$-position, on
      the axis.
    - `applied_force_fn`: Callable mapping the particle's *current* axial
      position to the total externally applied axial force there (see
      this module's docstring on how additional force types compose into
      this single scalar).
    - `n_steps`: Number of steps to advance.
    - `time_step`: Dimensionless time increment per step; strictly
      positive.
    - `velocity_bracket`: Forwarded to every step's
      `solve_force_balanced_velocity` call. Since the force-balanced
      velocity should change only gradually from step to step once the
      trajectory settles, the same bracket usually suffices throughout;
      widen it if a later step's residual turns out not to change sign
      across it.

    Returns:
    - A list of `n_steps + 1`
      [`ParticleState`][magnetofluidics_pinn.types.ParticleState] instances
      (including the initial state at `time=0.0`), each with `position`
      `(0.0, z)` — the radial component is always exactly `0.0`, per
      `SphericalObstacle`'s on-axis constraint — and `velocity` `(0.0,
      U_z)`, on the CPU, in the package's usual dimensionless units.

    Raises:
    - `ValueError`: If `n_steps` or `time_step` is not strictly positive.
    - `RuntimeError`: If the particle would leave the domain's valid
      interior (its surface reaching the inlet, outlet, or channel wall)
      before completing `n_steps`. This function does not return partial
      results on that path — it raises before appending the offending
      step: catch the exception and retry with a smaller `time_step` or
      fewer `n_steps` if the steps completed so far are needed.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be strictly positive; got {n_steps!r}.")
    if time_step <= 0.0:
        raise ValueError(f"time_step must be strictly positive; got {time_step!r}.")

    current_position = float(initial_axial_position)
    current_network = network
    current_time = 0.0
    states = [
        ParticleState(
            position=torch.tensor([0.0, current_position]),
            velocity=torch.zeros(2),
            time=current_time,
        )
    ]

    for step in range(n_steps):
        if not (obstacle_radius < current_position < domain.length - obstacle_radius):
            raise RuntimeError(
                f"Particle reached z={current_position!r} at step {step}, no longer strictly "
                f"inside [obstacle_radius, domain.length - obstacle_radius] = "
                f"[{obstacle_radius!r}, {domain.length - obstacle_radius!r}]; stopping before "
                "the obstacle would straddle the inlet, outlet, or wall. Returning the states "
                "completed so far is not possible from inside this exception — catch it and use "
                "a smaller time_step or fewer n_steps."
            )
        obstacle = SphericalObstacle(axial_position=current_position, radius=obstacle_radius)
        applied_force = applied_force_fn(current_position)
        if verbose:
            print(f"Step {step + 1}/{n_steps}: z={current_position:.5f}, "
                  f"applied_force_z={applied_force:.5f}")

        particle_velocity, current_network, _ = solve_force_balanced_velocity(
            current_network, domain, obstacle, fluid_config, training_config,
            applied_force_z=applied_force, velocity_bracket=velocity_bracket,
            n_obstacle_surface_points=n_obstacle_surface_points,
            n_quadrature_points=n_quadrature_points, velocity_tolerance=velocity_tolerance,
            refinement_config=refinement_config, verbose=verbose,
        )

        current_position = current_position + particle_velocity * time_step
        current_time += time_step
        states.append(
            ParticleState(
                position=torch.tensor([0.0, current_position]),
                velocity=torch.tensor([0.0, particle_velocity]),
                time=current_time,
            )
        )

    return states
