r"""Force-balanced particle motion under a genuine two-way flow coupling.

`trajectory.integrator.integrate_trajectory` treats the
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
[`ParticleState`][magnetofluidics_pinn.types.ParticleState]
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

**CHANGED — `train()` replaces `train_around_obstacle()`.** Both
functions below previously called a dedicated `train_around_obstacle()`
entry point in `training.trainer`. That function has been folded into
`train()` (see that module's docstring): the flow field for a given
candidate velocity is now obtained by calling `train()` with an
[`ObstacleConfig`][magnetofluidics_pinn.config.ObstacleConfig] built from
`obstacle` and the candidate `obstacle_velocity`. `verbose`, which
`train_around_obstacle` previously took as a separate keyword argument
hard-wired to `False` here (retraining silently at every bracketing step,
independent of whatever the caller's own `training_config.verbose` was
set to), now lives on `TrainingConfig` itself; the same silencing is
preserved below via `dataclasses.replace(current_config, verbose=False)`
rather than mutating the caller's config in place.
"""

from __future__ import annotations

import math
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Callable, Any

import torch
from torch import nn
from scipy.optimize import brentq

from magnetofluidics_pinn.config import (
    FluidConfig,
    ParticleConfig,
    TwoWayCouplingConfig,
    TrainingConfig,
)
from magnetofluidics_pinn.physics.hydrodynamic_drag import surface_traction_force
from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force  # NEW
from magnetofluidics_pinn.training.trainer import TrainingHistory, train
from magnetofluidics_pinn.data_types import Domain, MagneticFieldSample, ParticleState
from magnetofluidics_pinn.verification.benchmarks import shell_force_profile, shell_force_spread
from magnetofluidics_pinn.networks.mlp import build_mlp  # NEW
from magnetofluidics_pinn.networks.stream_function import StreamFunctionSphereFlow  # NEW
from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory, train_stream_function_flow

# CHANGED: ambient centreline velocity of the "fixed sphere in Poiseuille flow" problem.
_AMBIENT_PEAK_VELOCITY = 1.0


def _expand_velocity_bracket(
        force_residual_fn, low: float, high: float, max_iter: int = 5, expansion_factor: float = 1.8
) -> tuple[float, float]:
    r_low = force_residual_fn(low)
    r_high = force_residual_fn(high)

    step = 0
    while r_low * r_high > 0.0 and step < max_iter:
        if r_high > 0:
            # Need higher positive velocity to generate stronger opposing drag
            high = high * expansion_factor + 0.5
            r_high = force_residual_fn(high)
        else:
            # Need lower/negative velocity
            low = low * expansion_factor - 0.5
            r_low = force_residual_fn(low)
        step += 1

    if r_low * r_high > 0.0:
        raise ValueError(
            f"Could not bracket root after {max_iter} expansions. "
            f"Last bracket: ({low:.2f}, {high:.2f}) with residuals ({r_low:.3e}, {r_high:.3e})."
        )
    return low, high


def solve_force_balanced_velocity(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    particle_config: ParticleConfig,
    particle_state: ParticleState,
    training_config: TrainingConfig,
    applied_force_z: float,
    velocity_bracket: tuple[float, float],
    n_surface_points: int = 200,
    n_quadrature_points: int = 181,
    velocity_tolerance: float = 1.0e-3,
    refinement_config: TrainingConfig | None = None,
) -> tuple[float, nn.Module, TrainingHistory]:
    r"""Solve for the particle's force-balanced translational velocity, at a fixed position.

    Brackets and roots the scalar force residual
    $F_{\text{applied}} + F_z^{\text{drag}}(U_z)$ over `velocity_bracket`
    with `scipy.optimize.brentq`, retraining
    ([`train`][magnetofluidics_pinn.training.trainer.train] with
    [`TwoWayCouplingConfig`][magnetofluidics_pinn.config.TwoWayCouplingConfig])
    the two-way-coupled flow field at each candidate $U_z$ and evaluating hydrodynamic
    drag via
    [`surface_traction_force`][magnetofluidics_pinn.physics.hydrodynamic_drag.surface_traction_force].

    Successive candidates are warm-started from the previous iterate's converged
    network parameters. `refinement_config` allows warm-started re-solves to use a
    reduced optimization budget compared to the cold start.

    Args:
    - `network`: Starting flow network for the *first* (cold-started)
      candidate velocity evaluated (`velocity_bracket`'s lower end);
      every subsequent candidate warm-starts from the previous one's
      result. Never mutated — `train` already deep-copies before training.
    - `domain`, `object`, `fluid_config`: As in
      [`train`][magnetofluidics_pinn.training.trainer.train]; a fresh
      [`TwoWayCouplingConfig`][magnetofluidics_pinn.config.TwoWayCouplingConfig]
      is built internally for each candidate velocity
    - `fluid_config`: Physical properties of the fluid.
    - `particle_config`: Intrinsic physical/geometric properties of the microrobot.
    - `particle_state`: Instantaneous position and kinematic state.
    - `training_config`: Used for the first, cold-started evaluation.
      Regardless of its own `verbose` setting, every retraining call this
      function makes runs silently (see this module's docstring) — pass
      `verbose=True` to *this* function instead, to report progress across
      candidates rather than within any single retraining.
    - `applied_force_z`: The sum of every externally applied axial force
      on the particle (magnetic today; anything else the caller adds
      later), evaluated at `particle_state.axial_position`. `0.0` finds the
      velocity of a passively-advected, non-actuated particle.
    - `velocity_bracket`: `(low, high)` with `low < high`; the force
      residual must have opposite signs at the two ends (checked below) —
      a reasonable starting bracket is a generous multiple of the
      Faxén-corrected point-particle estimate
      ([`physics.hydrodynamic_drag.faxen_corrected_velocity`][magnetofluidics_pinn.physics.hydrodynamic_drag.faxen_corrected_velocity])
      at the same position.
    - `n_surface_points`: Forwarded to each candidate's
      `TwoWayCouplingConfig`.
    - `n_quadrature_points`: Forwarded to `surface_traction_force`.
    - `velocity_tolerance`: `brentq`'s `xtol`, in the same dimensionless
      velocity units as `velocity_bracket`.
    - `refinement_config`: `TrainingConfig` used for every candidate
      *after* the first; `None` (the default) reuses `training_config`
      for all of them, unchanged. Since every candidate after the first
      is warm-started, this is typically set to far fewer epochs/rounds
      than a cold start needs.

    Returns:
    - A tuple `(particle_velocity, trained_network, history)` containing the root $U_z$,
      the flow network converged at $U_z$, and the final training history.

    Raises:
    - `ValueError`: If `velocity_bracket[0] >= velocity_bracket[1]`, or if
      the force residual does not change sign across `velocity_bracket`
      (the bracket needs widening).
    """
    velocity_low, velocity_high = velocity_bracket
    if velocity_low >= velocity_high:
        raise ValueError(f"velocity_bracket must satisfy low < high; got {velocity_bracket!r}.")

    device = training_config.torch_device
    dtype = training_config.torch_dtype
    network = network.to(device=device, dtype=dtype)
    state: dict[str, Any] = {"network": network, "history": None}

    def force_residual(candidate_velocity: float) -> float:
        current_config = training_config if state["history"] is None else (refinement_config or training_config)
        silent_config = dataclasses.replace(current_config, verbose=False)
        candidate_state = ParticleState(
            position=particle_state.position.to(device=device, dtype=dtype),
            velocity=torch.tensor([0.0, candidate_velocity], device=device, dtype=dtype),
            time=particle_state.time,
        )
        pos_tuple = (
            float(candidate_state.position[0].item()),
            float(candidate_state.position[1].item()),
        )
        vel_tuple = (0.0, float(candidate_velocity))

        coupling_config = TwoWayCouplingConfig(
            particle_config=particle_config,
            particle_velocity=vel_tuple,
            particle_position=pos_tuple,
            n_surface_points=n_surface_points,
        )

        trained, history = train(
            state["network"],
            domain,
            fluid_config=fluid_config,
            training_config=silent_config,
            coupling_config=coupling_config,
        )
        # Evaluate drag force with disabled network parameter gradients
        trained.eval()
        #with torch.no_grad():
        drag_force = surface_traction_force(
            trained, particle_config=particle_config, particle_state=candidate_state,
            n_quadrature_points=n_quadrature_points
        ).item()
        state["network"] = trained
        state["history"] = history
        residual = applied_force_z + drag_force
        # GPU cache cleanup between brentq candidate evaluations
        # device = next(trained.parameters()).device
        # if device.type == "cuda":
        #     torch.cuda.empty_cache()
        if training_config.verbose:
            print(f"  U_z = {candidate_velocity:+.3e}  ->  F_drag = {drag_force:+.3e}, residual = {residual:+.3e}")
        return residual

    # Evaluate initial residuals at bracket endpoints
    residual_low = force_residual(velocity_low)
    residual_high = force_residual(velocity_high)

    # CHANGED: Removed the premature `raise ValueError` that previously rendered
    # `_expand_velocity_bracket` unreachable dead code. Automatic expansion now proceeds.
    if residual_low * residual_high > 0.0:
        if training_config.verbose:
            print("Initial bracket does not enclose a sign change; expanding dynamically...")
        velocity_low, velocity_high = _expand_velocity_bracket(
            force_residual, velocity_low, velocity_high
        )

    # Resolve certified root with scalar Brent method
    particle_velocity = brentq(force_residual, velocity_low, velocity_high, xtol=velocity_tolerance)
    force_residual(particle_velocity)

    return particle_velocity, state["network"], state["history"]


def advance_particle_two_way(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    training_config: TrainingConfig,
    particle_config: ParticleConfig,
    initial_axial_position: float,
    applied_force_fn: Callable[[float], float],
    n_steps: int,
    time_step: float,
    velocity_bracket: tuple[float, float],
    refinement_config: TrainingConfig | None = None,
    n_surface_points: int = 200,
    n_quadrature_points: int = 181,
    velocity_tolerance: float = 1.0e-3,
) -> list[ParticleState]:
    r"""Advance a two-way-coupled particle through the channel, one force-balanced step at a time.

    An explicit-Euler stepper built directly on
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
    - domain`, `fluid_config`, `training_config`, `particle_config`,
      `refinement_config`, `n_object_surface_points`, `n_quadrature_points`,
      `velocity_tolerance`: Forwarded to `solve_force_balanced_velocity` at
      every step (`training_config.verbose` governs progress printing;
      there is no separate `verbose` parameter on this function).
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
      `ParticleState`'s on-axis constraint — and `velocity` `(0.0,
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

    device = training_config.torch_device
    dtype = training_config.torch_dtype

    current_position = float(initial_axial_position)
    current_network = network
    current_time = 0.0
    particle_state = ParticleState(
            position=torch.tensor([0.0, current_position], device=device, dtype=dtype),
            velocity=torch.zeros(2, device=device, dtype=dtype),
            time=current_time,
        )
    states = [particle_state]
    for step in range(n_steps):
        if not (particle_config.radius < current_position < domain.length - particle_config.radius):
            raise RuntimeError(
                f"Particle reached z={current_position!r} at step {step}, no longer strictly "
                f"inside [object_radius, domain.length - particle_radius] = "
                f"[{particle_config.radius!r}, {domain.length - particle_config.radius!r}]; stopping before "
                "the object would straddle the inlet, outlet, or wall. Returning the states "
                "completed so far is not possible from inside this exception — catch it and use "
                "a smaller time_step or fewer n_steps."
            )
        particle_state = ParticleState(
            position=torch.tensor([0.0, current_position], device=device, dtype=dtype),
            velocity=torch.zeros(2, device=device, dtype=dtype),
            time=current_time,
        )
        applied_force = applied_force_fn(current_position)
        if training_config.verbose:
            print(f"Step {step + 1}/{n_steps}: z={current_position:.5f}, "
                  f"applied_force_z={applied_force:.5f}")

        particle_velocity, current_network, _ = solve_force_balanced_velocity(
            current_network, domain, fluid_config,
            particle_config=particle_config, particle_state=particle_state,
            training_config=training_config,
            applied_force_z=applied_force, velocity_bracket=velocity_bracket,
            n_surface_points=n_surface_points,
            n_quadrature_points=n_quadrature_points, velocity_tolerance=velocity_tolerance,
            refinement_config=refinement_config,
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


def make_dipole_applied_force_fn(  # NEW
    field_fn: Callable[[torch.Tensor], MagneticFieldSample],
    magnetic_moment: torch.Tensor,
) -> Callable[[float], float]:
    r"""Build an `applied_force_fn` for `advance_particle_two_way` from a prescribed magnetic field.

    The one-way path,
    [`trajectory.integrator.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory],
    derives the magnetic drift force from a `field_fn`/`magnetic_moment`
    pair via
    [`physics.magnetic_forcing.dipole_force`][magnetofluidics_pinn.physics.magnetic_forcing.dipole_force].
    `advance_particle_two_way` instead takes an arbitrary scalar-valued
    `applied_force_fn`, so nothing previously forced a one-way/two-way
    comparison under "the same" actuation to actually use the same force
    model — one call site could prescribe a real field while the other
    silently fell back to `lambda z: 0.0`. This function removes that gap
    by building the two-way `applied_force_fn` from the identical
    primitives the one-way path already uses.

    Args:
    - `field_fn`: Callable returning a
      [`MagneticFieldSample`][magnetofluidics_pinn.types.MagneticFieldSample]
      for a batch of coordinates, e.g. `uniform_field` or `biot_savart_field`
      — the same callable passed to `integrate_trajectory`.
    - `magnetic_moment`: Tensor of shape `(1, 2)` or `(2,)`, the particle's
      magnetic moment, in the same dimensionless units `dipole_force`
      expects — the same value passed to `integrate_trajectory`.

    Returns:
    - A callable mapping an axial position `z` (on-axis, `r = 0`, matching
      this module's on-axis confinement) to the axial component of the
      dipole force there, suitable for `advance_particle_two_way`'s
      `applied_force_fn`. Under a spatially uniform field the dipole force
      is exactly zero everywhere [@abbott2020magnetic], so the returned
      callable reduces to `lambda z: 0.0` in that case — now as a physical
      consequence of the field being uniform, not by omission.
    """
    def applied_force_fn(axial_position: float) -> float:
        with torch.enable_grad():
            position = torch.tensor(
                [[0.0, axial_position]], dtype=magnetic_moment.dtype, device=magnetic_moment.device
            ).requires_grad_(True)
            force = dipole_force(field_fn, position, magnetic_moment)
            return force[0, 1].item()

    return applied_force_fn


def affine_force_balance(
    force_ambient: float,
    force_translating: float,
    applied_force: float,
) -> float:
    r"""Solve the scalar force balance $F_{\mathrm{applied}} + F_0 + U_p F_1 = 0$ in closed form.

    By linearity of the Stokes equations [@happel1983low]:
    $$F_{\mathrm{drag}}(U_p) = F_0 + U_p \, F_1.$$
    At force balance:
    $$F_{\mathrm{applied}} + F_0 + U_p F_1 = 0 \iff U_p = -\frac{F_{\mathrm{applied}} + F_0}{F_1}.$$

    Args:
    - `force_ambient`: Hydrodynamic force $F_0$ on a stationary particle ($U_p = 0$) in
      ambient flow ($U_{\infty} = 1$).
    - `force_translating`: Drag force per unit velocity $F_1$ on a translating particle
      ($U_p = 1$) in stagnant fluid ($U_{\infty} = 0$). Must be strictly negative.
    - `applied_force`: External axial force $F_{\mathrm{applied}}$ acting on the particle.

    Returns:
    - Force-balanced particle velocity $U_p \in \mathbb{R}$.

    Raises:
    - `ValueError`: If `force_translating` is non-negative (dissipative Stokes drag
       must oppose translation).
    """
    if force_translating >= 0.0:
        raise ValueError(
            "force_translating must be strictly negative (drag opposes motion in Stokes flow); "
            f"got {force_translating!r}."
        )
    return -(applied_force + force_ambient) / force_translating


@dataclass(frozen=True)
class ResistanceResult:
    r"""Immutable container holding solutions to the canonical linear Stokes resistance problems.

    Stores the solutions of the two canonical subproblems:
    $$
    F(U_p) = F_0 + U_p \, F_1,
    $$
    where $F_0$ is `force_ambient` and $F_1$ is `force_translating`.

    Args:
    - `force_ambient`: Drag $F_0$ on a stationary particle in ambient flow ($U_p = 0, U_{\infty} = 1$).
    - `force_translating`: Drag per unit velocity $F_1$ in quiescent fluid ($U_p = 1, U_{\infty} = 0$).
    - `spread_ambient`: Relative variation of $F_0$ across concentric control shells.
    - `spread_translating`: Relative variation of $F_1$ across concentric control shells.
    - `network_ambient`: Trained neural model for the stationary obstacle problem.
    - `network_translating`: Trained neural model for the unit translation problem.
    - `history_ambient`: Training history for the stationary obstacle problem.
    - `history_translating`: Training history for the unit translation problem.
    """

    force_ambient: float
    force_translating: float
    spread_ambient: float
    spread_translating: float
    network_ambient: nn.Module
    network_translating: nn.Module
    history_ambient: Any
    history_translating: Any

    # --------------------------------------------------------------------------
    # Physical Properties & Closed-Form Solvers
    # --------------------------------------------------------------------------

    @property
    def resistance(self) -> float:
        r"""Hydrodynamic drag coefficient $\zeta = -F_1 > 0$."""
        return -self.force_translating

    @property
    def mobility(self) -> float:
        r"""Effective hydrodynamic mobility $\mathcal{M} = 1 / \zeta = -1 / F_1$."""
        return 1.0 / self.resistance

    @property
    def passive_velocity(self) -> float:
        r"""Passive force-free velocity $U_{\mathrm{free}} = -F_0 / F_1$ when $F_{\mathrm{applied}} = 0$."""
        return self.balanced_velocity(0.0)

    def balanced_velocity(self, applied_force: float) -> float:
        r"""Compute force-balanced particle velocity $U_p = -(F_{\mathrm{applied}} + F_0) / F_1$.

        Args:
        - `applied_force`: Total external axial actuation force $F_{\mathrm{applied}}^*$.

        Returns:
        - Resulting force-balanced velocity $U_p^*$.
        """
        return affine_force_balance(self.force_ambient, self.force_translating, applied_force)

    def drag_force(self, particle_velocity: float) -> float:
        r"""Evaluate hydrodynamic drag force at arbitrary velocity: $F(U_p) = F_0 + U_p F_1$.

        Args:
        - `particle_velocity`: Particle translational velocity $U_p^*$.

        Returns:
        - Total axial hydrodynamic force $F_z^*$.
        """
        return self.force_ambient + particle_velocity * self.force_translating

    @property
    def force_fixed(self) -> float:
        """Alias for `force_ambient` ($F_0$)."""
        return self.force_ambient

    @property
    def force_unit_translation(self) -> float:
        """Alias for `force_translating` ($F_1$)."""
        return self.force_translating

    @property
    def shell_spread_fixed(self) -> float:
        """Alias for `spread_ambient`."""
        return self.spread_ambient

    @property
    def shell_spread_unit(self) -> float:
        """Alias for `spread_translating`."""
        return self.spread_translating

    @property
    def network_fixed(self) -> nn.Module:
        """Alias for `network_ambient`."""
        return self.network_ambient

    @property
    def network_unit(self) -> nn.Module:
        """Alias for `network_translating`."""
        return self.network_translating

    @property
    def history_fixed(self) -> Any:
        """Alias for `history_ambient`."""
        return self.history_ambient

    @property
    def history_unit(self) -> Any:
        """Alias for `history_translating`."""
        return self.history_translating


def solve_resistance_problems(
    build_network: Callable[[Domain], nn.Module],
    domain: Domain,
    fluid_config: FluidConfig,
    particle_config: ParticleConfig,
    particle_state: ParticleState,
    training_config: TrainingConfig,
    n_surface_points: int = 250,          # CHANGED: Added missing parameter with robust default
    particle_loss_weight: float = 20.0,   # CHANGED: Added missing parameter with robust default
    n_quadrature_points: int = 181,
    shell_factors: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
) -> ResistanceResult:
    r"""Train the two canonical linear resistance problems using the primitive-variable PINN.

    By linearity of the steady Stokes equations [@happel1983low], the hydrodynamic
    force $F(U_p)$ on an on-axis translating spherical particle decomposes into an
    exact affine relation:
    $$F(U_p) = F_0 + U_p \, F_1,$$
    where:

    - $F_0$: Force on a stationary particle ($U_p = 0$) in active Poiseuille flow (
      $U_{\infty} = 1$).
    - $F_1$: Drag per unit velocity on a translating particle ($U_p = 1$) in
      quiescent fluid ($U_{\infty} = 0$).

    Solving these two problems once analytically closes the force-balanced velocity
    for any applied force $F_{\mathrm{applied}}$:
    $$F_{\mathrm{applied}} + F_0 + U_p F_1 = 0 \iff U_p = -\frac{F_{\mathrm{applied}} + F_0}{F_1},$$
    eliminating the computational overhead and instability of iterative nonlinear root-finding.

    Args:
    - `build_network`: Factory callable `(Domain) -> nn.Module` instantiating a fresh
      hard-wall-constrained flow network.
    - `domain`: Nondimensional channel domain geometry.
    - `fluid_config`: Physical configuration specifying fluid properties.
    - `particle_config`: Physical particle geometry and properties.
    - `particle_state`: Instantaneous particle kinematic state.
    - `training_config`: Hyperparameter configuration for Adam and L-BFGS training.
    - `n_surface_points`: Number of collocation points on the sphere boundary.
    - `particle_loss_weight`: Weight multiplying the soft particle surface loss.
    - `n_quadrature_points`: Meridian quadrature resolution for surface traction integration.
    - `shell_factors`: Scale multipliers for momentum-conserving control shells.

    Returns:
    - Populated `ResistanceResult` containing trained models, canonical forces $F_0$ and $F_1$,
      and shell force spreads.

    Raises:
    - `ValueError`: If `shell_factors` are invalid or exceed domain boundaries.
    """
    if any(factor < 1.0 for factor in shell_factors):
        raise ValueError(
            "Every shell factor must be >= 1.0 (on or outside the sphere surface).")

    particle_axial_position = float(particle_state.position[1].item())
    particle_radius = particle_config.radius
    largest_shell_radius = max(shell_factors) * particle_radius

    # Verify that the largest integration shell fits strictly inside channel margins
    channel_clearance = min(domain.radius, particle_axial_position,
                            domain.length - particle_axial_position)
    if largest_shell_radius >= channel_clearance:
        raise ValueError(
            f"Largest shell radius ({largest_shell_radius:.3e}) must be strictly smaller "
            f"than channel clearance ({channel_clearance:.3e})."
        )

    position = (float(particle_state.position[0].item()),
                float(particle_state.position[1].item()))

    # Problem 0: Fixed particle (Up = 0) in Poiseuille flow (u_max from domain)
    # Problem 1: Translating particle (Up = 1) in quiescent fluid (u_max = 0)
    problems = (
        (domain, 0.0),
        (dataclasses.replace(domain, u_max=0.0), 1.0),
    )

    def solve_one(problem: tuple[Domain, float]) -> tuple[nn.Module, float, float, TrainingHistory]:
        problem_domain, velocity = problem

        # CHANGED: Forward validated parameters to TwoWayCouplingConfig without undefined names
        coupling_config = TwoWayCouplingConfig(
            particle_config=particle_config,
            particle_velocity=(0.0, velocity),
            particle_position=position,
            n_surface_points=n_surface_points,
            particle_loss_weight=particle_loss_weight,
            positivity_loss_weight_override=0.0,  # Physical flow reversal allowed
        )

        start = build_network(problem_domain)
        trained, history = train(
            network=start,
            domain=problem_domain,
            fluid_config=fluid_config,
            training_config=training_config,
            coupling_config=coupling_config,
        )

        profile = shell_force_profile(trained, particle_config, particle_state, shell_factors, n_quadrature_points)
        force = float(profile["force"].mean())
        spread = shell_force_spread(profile)
        return trained, force, spread, history

    trained_ambient, force_ambient, spread_ambient, history_ambient = solve_one(problems[0])
    trained_translating, force_translating, spread_translating, history_translating = solve_one(problems[1])
    return ResistanceResult(
        force_ambient=force_ambient,
        force_translating=force_translating,
        spread_ambient=spread_ambient,
        spread_translating=spread_translating,
        network_ambient=trained_ambient,
        network_translating=trained_translating,
        history_ambient=history_ambient,
        history_translating=history_translating,
    )


def solve_resistance_problems_stream(
    domain: Domain,
    fluid_config: FluidConfig,
    particle_config: ParticleConfig,
    particle_state: ParticleState,
    training_config: TrainingConfig,
    hidden_layers: tuple[int, ...] = (64, 64, 64, 64),
    activation: str = "tanh",
    taper_margin: float = 0.9,
    n_quadrature_points: int = 181,
    shell_factors: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
    shell_fraction: float = 0.25,
    shell_factor: float = 2.5,
    band_fraction: float = 0.0,
    band_factor: float = 1.5,
) -> ResistanceResult:
    r"""Train the two canonical linear resistance problems using the stream-function PINN.

    Constructs and trains `StreamFunctionSphereFlow` instances where incompressibility
    $\nabla \cdot \mathbf{u} \equiv 0$ and boundary conditions are satisfied structurally.

    CHANGED: Removed redundant duplicated validation checks on `shell_factors` and verified
    spatial bounding against the channel boundaries.

    Args:
    - `domain`: Nondimensional channel geometry.
    - `fluid_config`: Physical configuration specifying fluid properties.
    - `particle_config`: Particle geometry providing radius $a$.
    - `particle_state`: Instantaneous particle kinematic state.
    - `training_config`: Hyperparameter configuration for Adam and L-BFGS training.
    - `hidden_layers`: Width and depth specification for underlying raw network.
    - `activation`: Activation function identifier (e.g., `"tanh"`).
    - `taper_margin`: Radial thickness $\Delta$ bounding the compact taper ball.
    - `n_quadrature_points`: Meridian quadrature resolution for surface traction integration.
    - `shell_factors`: Scale multipliers for momentum-conserving control shells.
    - `shell_fraction`, `shell_factor`: Collocation distribution parameters for outer shell.
    - `band_fraction`, `band_factor`: Collocation distribution parameters for near-surface band.

    Returns:
    - Populated `ResistanceResult` containing stream-function models, canonical forces,
      and shell force spreads.

    Raises:
    - `ValueError`: If shell factors are $< 1.0$ or exceed physical channel clearance.
    """
    # CHANGED: Removed duplicate line of this check
    if any(factor < 1.0 for factor in shell_factors):
        raise ValueError("Every shell factor must be >= 1.0 (on or outside the sphere surface).")

    particle_axial_position = float(particle_state.position[1].item())
    particle_radius = particle_config.radius
    largest_shell_radius = max(shell_factors) * particle_radius

    # Verify that the largest integration shell fits strictly inside channel margins
    channel_clearance = min(domain.radius, particle_axial_position, domain.length - particle_axial_position)
    if largest_shell_radius >= channel_clearance:
        raise ValueError(
            f"Largest shell radius ({largest_shell_radius:.3e}) must be strictly smaller "
            f"than channel clearance ({channel_clearance:.3e})."
        )

    # Problem 0: Fixed particle (ambient = 1.0, Up = 0.0)
    # Problem 1: Translating particle (ambient = 0.0, Up = 1.0)
    problems = (
        (1.0, 0.0),
        (0.0, 1.0),
    )

    def solve_one(problem: tuple[float, float]) -> tuple[nn.Module, float, float, StreamLossHistory]:
        ambient, velocity = problem
        raw = build_mlp(
            n_inputs=2,
            n_outputs=2,
            hidden_layers=hidden_layers,
            training_config=training_config,
            activation=activation,
        )
        start = StreamFunctionSphereFlow(
            raw_network=raw,
            domain=domain,
            particle_axial_position=particle_axial_position,
            particle_radius=particle_radius,
            particle_axial_velocity=velocity,
            flow_amplitude=ambient,
            taper_margin=taper_margin,
        )
        trained, history = train_stream_function_flow(
            network=start,
            domain=domain,
            fluid_config=fluid_config,
            training_config=training_config,
            particle_axial_position=particle_axial_position,
            particle_radius=particle_radius,
            inlet_peak_velocity=ambient,
            shell_fraction=shell_fraction,
            shell_factor=shell_factor,
            band_fraction=band_fraction,
            band_factor=band_factor,
        )
        profile = shell_force_profile(trained, particle_config, particle_state, shell_factors, n_quadrature_points)
        force = float(profile["force"].mean())
        spread = shell_force_spread(profile)
        return trained, force, spread, history

    trained_ambient, force_ambient, spread_ambient, history_ambient = solve_one(problems[0])
    trained_translating, force_translating, spread_translating, history_translating = solve_one(problems[1])

    return ResistanceResult(
        force_ambient=force_ambient,
        force_translating=force_translating,
        spread_ambient=spread_ambient,
        spread_translating=spread_translating,
        network_ambient=trained_ambient,
        network_translating=trained_translating,
        history_ambient=history_ambient,
        history_translating=history_translating,
    )
