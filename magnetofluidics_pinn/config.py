"""Immutable configuration objects for `magnetofluidics_pinn`.

Every stage of the pipeline (geometry, sampling, network, training) is
parameterized by one of the frozen dataclasses defined here. Passing
configuration explicitly, rather than through global state, keeps every
downstream function pure and independently testable

.Every config below validates itself in `__post_init__`: constructing an
invalid one raises immediately, rather than surfacing as a confusing
failure much later (e.g., inside `scaling.compute_scales` or a training
loop many collocation points and epochs downstream of the actual mistake).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch

from magnetofluidics_pinn.types import ParticleState
from magnetofluidics_pinn.device_utils import resolve_device

# Numerical tolerance for "is this vector a unit vector" checks; loose
# enough to tolerate float rounding, tight enough to catch a genuinely
# mis-specified direction. Matches the tolerance used at the point of use
# in `boundary_conditions.magnetic_field_bc.uniform_field`.
_UNIT_NORM_TOLERANCE = 1.0e-6

# CHANGED: was `reference_length: float = DomainConfig.radius`, which looked
# dynamically linked to DomainConfig's default but was actually baked to a
# plain float at class-definition time — identical runtime behavior to a
# literal, just less honest about it.
_DEFAULT_REFERENCE_LENGTH = 7.5e-4  # meter;
_DEFAULT_REFERENCE_VELOCITY = 1.0e-3  # meter/second

# NEW: promoted out of `training.trainer` (previously a module-private
# `_BOUNDARY_LOSS_WEIGHT`) so that `ObstacleConfig.obstacle_loss_weight`'s
# default can reference it here, alongside every other config default,
# without `config` importing from `training.trainer` (which would create
# a circular import, since `training.trainer` already imports from
# `config`). `training.trainer` now imports this constant instead of
# defining its own copy, so the wall/inlet/outlet/obstacle boundary weight
# is declared in exactly one place.
DEFAULT_BOUNDARY_LOSS_WEIGHT: float = 10.0


def _check_if_finite_positive(value: float, name: str="") -> None:
    """Raise ValueError if `value` is not finite and strictly positive."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}  must be finite and strictly positive; got {value!r}.")


@dataclass(frozen=True)
class FluidConfig:
    """Configuration of the viscous flow model.

    All physical fields follow the MKSA/SI convention. `reference_velocity`
    and `reference_length` are not free choices: they are the scales that
    [`scaling.compute_scales`][magnetofluidics_pinn.scaling.compute_scales]
    uses to nondimensionalize every quantity the solvers see, which keeps
    the network's inputs and outputs near unit order despite the microscale
    geometry (SI values around `1.0e-4` to `1.0e-3` would otherwise starve
    gradient-based optimization of numerical precision).

    Args:
    - `regime`: Either `"stokes"` for the steady, low-Reynolds-number
      approximation or `"navier_stokes"` for the full unsteady model.
    - `dynamic_viscosity`: Dynamic viscosity, in `pascal-second`, of the
      carrier fluid (e.g., `1.0e-3` for water-like viscosity, matching the
      target application).
    - `density`: Fluid density, in `kilogram-per-cubic-meter`. Only used when
      `regime == "navier_stokes"`.
    - `reference_velocity`: Characteristic velocity scale, in
      `meter-per-second`, used to nondimensionalize velocity and (with
      `reference_length`) time and pressure.
    - `reference_length`: Characteristic length scale, in `meter`, used to
      nondimensionalize every spatial coordinate. Defaults to the vessel
      radius order of magnitude (a few hundred micrometers).

    Raises:
    - `ValueError`: If `dynamic_viscosity`, `density`, `reference_velocity`,
      or `reference_length` is not finite and strictly positive.
    """
    
    regime: Literal["stokes", "navier_stokes"] = "stokes"
    dynamic_viscosity: float = 1.0e-3  # [Pa.s]
    density: float = 1.0e3  # [kg/m^3]
    
    def __post_init__(self) -> None:
        for field_name in ("dynamic_viscosity", "density"):
            value = getattr(self, field_name)
            _check_if_finite_positive(value, field_name)


@dataclass(frozen=True)
class DomainConfig:
    """Configuration of the vessel geometry.

    All physical fields follow the MKSA/SI convention (meter, kilogram,
    second, ampere) unless stated otherwise. The solvers never consume this
    configuration directly: [`scaling.nondimensionalize_domain`]
    [magnetofluidics_pinn.scaling.nondimensionalize_domain] converts it to a
    dimensionless [`Domain`][magnetofluidics_pinn.types.Domain] first, which
    is what `sampling` and `physics` actually operate on.

    Args:
    - `kind`: Either `"channel"` or `"bifurcation"`.
    - `length`: Axial length of the vessel, in `meter` (e.g., a few
      millimeters, so `1.0e-3` to `1.0e-2`).
    - `radius`: Radius (or half-width) of the vessel, in `meter` (e.g., a few
      hundred micrometers, so around `1.0e-4`).
    - `u_max`: Maximum velocity in the flow, in `meter/second`.
    - `branch_angle`: Branch half-angle in radians, required only when
      `kind == "bifurcation"`.
    Raises:
    - `ValueError`: If `length` or `radius` is not finite and strictly
      positive, if `kind == "bifurcation"` and `branch_angle` is not a
      finite number, or if `kind == "channel"` and `branch_angle` is not
      `None`.
    """
    
    kind: Literal["channel", "bifurcation"] = "channel"
    length: float = _DEFAULT_REFERENCE_LENGTH * 8.0  # m
    radius: float = _DEFAULT_REFERENCE_LENGTH        # m
    u_max: float = _DEFAULT_REFERENCE_VELOCITY  # m/s
    branch_angle: float | None = None
    fluid: FluidConfig = FluidConfig()  # Fluid embedded by default
    
    def __post_init__(self) -> None:
        _check_if_finite_positive(self.length, "length")
        _check_if_finite_positive(self.radius, "radius")
        # _check_if_finite_positive(self.u_max, "u_max")
        if self.kind == "bifurcation":
            if self.branch_angle is None or not math.isfinite(self.branch_angle):
                raise ValueError("kind='bifurcation' requires a finite branch_angle.")
        elif self.branch_angle is not None:
            raise ValueError(
                "branch_angle is only meaningful when kind='bifurcation'; "
                f"got kind={self.kind!r} with branch_angle={self.branch_angle!r}."
            )
        
    @property
    def reference_length(self) -> float:
        """Characteristic length scale, in meter, used to nondimensionalize every spatial coordinate."""
        return self.radius
    
    @property
    def reference_velocity(self) -> float:
        """Characteristic velocity scale, in meter/second, used to nondimensionalize
        velocity and (with reference_length) time and pressure."""
        return self.u_max

    @property
    def reynolds(self) -> float:
        r"""Reynolds number of the flow, $Re = \rho U_c L_c / \mu$.

        Uses the same characteristic velocity and length
        ([`scaling.compute_scales`][magnetofluidics_pinn.scaling.compute_scales]
        scales `velocity`/`length` with) as every other nondimensionalization
        in this package, so this is exactly the coefficient that must
        multiply the unsteady and convective terms of the dimensionless
        Navier-Stokes momentum equation once it is expressed on the same
        *viscous* pressure scale $P_c = \mu U_c / L_c$ that
        [`physics.fluid_residuals.stokes_residual`]
        [magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
        already assumes (dimensionless viscosity exactly $1$).

        Returns:
        - The (dimensionless) Reynolds number
          $Re = \rho U_c L_c / \mu$, always strictly positive since every
          factor is validated strictly positive in `__post_init__`.
        """
        return ((self.fluid.density * self.reference_velocity * self.reference_length)
                / self.fluid.dynamic_viscosity)


@dataclass(frozen=True)
class MagneticFieldConfig:
    """Configuration of the prescribed magnetic field input.

    `magnitude` already follows the MKSA/SI convention (`tesla`); no further
    normalization is required here; `scaling.compute_scales` re-expresses it
    relative to `magnitude` itself, so the dimensionless field stays near
    unit order.

    Args:
    - `source`: Either `"uniform"`, `"gradient"`, or `"biot_savart"`,
      selecting how the field-generating function is built.
    - `magnitude`: Reference field magnitude, in `tesla`.
    - `orientation`: Unit vector giving the field direction, used when
      `source == "uniform"`.
    - `time_dependent`: Whether the field varies with time.
    
    Raises:
    - `ValueError`: If `magnitude` is not finite and strictly positive, or
      if `orientation` is not (approximately) a unit vector.
      
    TODO: manage 2D/3D fields
    """
    
    source: Literal["uniform", "gradient", "biot_savart"] = "uniform"
    magnitude: float = 1.0e-2  # [T]
    orientation: tuple[float, float] = (1.0, 0.0)
    time_dependent: bool = False
    
    def __post_init__(self) -> None:
        _check_if_finite_positive(self.magnitude, "magnitude")
        if not all(math.isfinite(component) for component in self.orientation):
            raise ValueError(f"orientation components must be finite; got {self.orientation!r}.")
        orientation_norm = math.hypot(*self.orientation)
        if abs(orientation_norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError(
                f"orientation must be a unit vector; got a norm of {orientation_norm:.6g}."
            )


@dataclass(frozen=True)
class ParticleConfig:
    """Physical properties of the single magnetic microrobot tracked in Phase 1.

    All fields follow the MKSA/SI convention, matching `DomainConfig` and
    `FluidConfig`. Neither is consumed directly by
    [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]:
    call
    [`scaling.nondimensionalize_particle`][magnetofluidics_pinn.scaling.nondimensionalize_particle]
    first to obtain the dimensionless radius that function's own
    `particle_radius` argument expects — see that function's docstring for
    why `magnetic_moment` is deliberately *not* handled the same way yet.

    Args:
    - `kind`: The shape of the particle, either "spherical", "swarms",
      "cylinder", or "spheroid".
    - `radius`: Particle radius for "spherical" or "swarms", in meter.
      Used only for the Faxén-law finite-size correction to the ambient flow velocity
      ([`physics.hydrodynamic_drag`][magnetofluidics_pinn.physics.hydrodynamic_drag]);
      `0.0` recovers the exact point-particle limit.
    - `length`: Cylinder length, in meter. Only meaningful (and required to
      be strictly positive) when `kind == "cylinder"`; unused otherwise.
    - `aspect_ratio`: Length-to-diameter ratio, for a future spheroidal drag
      model. Not yet consumed by `max_surface_extension` or by any Faxén-type
      correction — `kind == "spheroid"` currently falls back to the spherical
      case (see that property's implementation).
    - `number`: Number of particles the configuration nominally represents,
      for `kind == "swarms"`. Not yet consumed downstream —
      `magnetofluidics_pinn` currently applies one `ParticleConfig` per call
      to
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]
      regardless of `initial_states`'s length, so a genuinely heterogeneous
      swarm is not yet supported (see that function's docstring).
    - `position`: Initial position of the particle, in meter (e.g., `(0.0, 0.0)` for the center of the domain).
    - `magnetic_moment`: Magnetic dipole moment vector `(m_r, m_z)`, in
      ampere-square-meter (`A m^2`), giving both the particle's magnetic
      "strength" and its (fixed, for Phase 1) orientation. Passed to
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]
      as its `magnetic_moment` argument only once already expressed in the
      dimensionless units that function's docstring describes — this
      config field records the *physical* moment for bookkeeping, but does
      not itself perform that conversion (see the caveat above).
    - `magnetic_ratio`: volumic ratio of magnetic material.

    Raises:
    - `ValueError`: If `radius` is not finite and non-negative, if either
      component of `magnetic_moment` is not finite, if `magnetic_moment` is
      the zero vector, if `kind == "swarms"` and `number` is not strictly
      positive, or if `kind == "cylinder"` and `length` is not strictly
      positive.
      
    Todo: manage 2D/3D particle
    """
    
    kind: Literal["spherical", "swarms", "cylinder", "spheroid"] = "spherical"
    radius: float = 0.0       # [m]
    length: float = 0.0       # [m]
    aspect_ratio: float = 1.0
    number: int = 1
    magnetic_moment: tuple[float, float] = (1.0e-13, 0.0)  # [A m^2]
    magnetic_ratio: float = 1.0
    
    def __post_init__(self) -> None:
        _check_if_finite_positive(self.radius, "radius")
        if not all(math.isfinite(component) for component in self.magnetic_moment):
            raise ValueError(f"magnetic_moment components must be finite; got {self.magnetic_moment!r}.")
        if math.hypot(*self.magnetic_moment) <= 0.0:
            raise ValueError("magnetic_moment must be non-zero.")
        if self.kind == "swarms" and self.number <= 0:
            raise ValueError(f"number of particles must be positive for swarms; got {self.number!r}.")
        if self.kind == "cylinder" and self.length <= 0:
            raise ValueError(f"length of cylinder must be positive; got {self.length!r}.")
        if self.kind == "spheroid" and not (0 < self.aspect_ratio <= 1):
            raise ValueError(f"aspect_ratio of spheroid must be between 0 and 1; got {self.aspect_ratio!r}.")
        
    @property
    def max_surface_extension(self) -> float:
        """Returns the maximum radial extension from the centre of mass."""
        if self.kind == "cylinder":
            return math.hypot(self.radius, self.length / 2.0)
        return self.radius
    
    @property
    def volume(self) -> float :
        """Returns the particle volume in cubic meters."""
        if self.kind == "spherical":
            return (4.0 / 3.0) * math.pi * self.radius ** 3
        elif self.kind == "cylinder":
            return math.pi * self.radius ** 2 * self.length
        elif self.kind == "spheroid":
            return (4.0 / 3.0) * math.pi * self.radius ** 2 * (self.length / 2.0)
        elif self.kind == "swarms":
            return self.number * (4.0 / 3.0) * math.pi * self.radius ** 3
        raise ValueError(f"Unknown kind: {self.kind}")

    @property
    def magnetic_volume(self) -> float :
        """"Returns the magnetic volume"""
        return self.volume*self.magnetic_ratio
        
    @property
    def magnetization(self) -> tuple[float | int, ...]:
        """Returns the magnetization vector in A/m."""
        return tuple(m / self.magnetic_volume for m in self.magnetic_moment)
    
    def with_magnetization(self, magnetization: tuple[float | int, ...]) -> ParticleConfig:
        """Pure functional updater replacing unsafe object.__setattr__ mutations."""
        mag_vol = self.volume * self.magnetic_ratio
        new_moment = tuple(m * self.magnetic_volume for m in magnetization)
        return replace(self, magnetic_moment=new_moment)


@dataclass(frozen=True)
class TwoWayCouplingConfig:
    r"""Two-way-coupling parameters for training the flow around a fixed (spherical) object.

    NEW. Passing an `ObjectConfig` to
    [`training.trainer.train`][magnetofluidics_pinn.training.trainer.train]
    switches it from the ordinary (obstacle-free) Stokes/Navier-Stokes
    training problem to the two-way-coupled problem solved around an
    embedded [`SphericalObject`][magnetofluidics_pinn.types.SphericalObject]:
    interior points are drawn by rejection sampling around the excluded
    volume (see
    [`sampling.collocation.sample_collocation_points_with_obstacle`]
    [magnetofluidics_pinn.sampling.collocation.sample_collocation_points_with_obstacle]),
    and an additional loss term enforces the object's own rigid-body
    velocity on its surface (see
    [`boundary_conditions.flow_bc.rigid_body_velocity_condition`]
    [magnetofluidics_pinn.boundary_conditions.flow_bc.rigid_body_velocity_condition]).
    `train` itself stays a single entry point either way; this dataclass is
    what makes the two problems reuse it instead of needing a separate
    `train_around_obstacle` function.

    Kept as its own dataclass rather than folded into
    [`TrainingConfig`][magnetofluidics_pinn.config.TrainingConfig], since
    `object_velocity` in particular is expected to vary from call to call
    within a single outer search (see
    [`trajectory.two_way_coupling.solve_force_balanced_velocity`]
    [magnetofluidics_pinn.trajectory.two_way_coupling.solve_force_balanced_velocity],
    which retrains at a new candidate `object_velocity` on every
    bracketing step): a small, per-call dataclass keeps that variation
    local, instead of requiring a fresh `TrainingConfig` — an object
    otherwise meant to be reused unchanged across an entire optimization
    run — for every candidate velocity.

    Args:
    - `object`: The embedded sphere; must fit strictly inside the
      training domain. Checked by
      [`sampling.collocation.sample_collocation_points_with_obstacle`]
      [magnetofluidics_pinn.sampling.collocation.sample_collocation_points_with_obstacle]
      at training time, not here, since this dataclass has no `Domain` of
      its own to check `obstacle` against.
    - `object_velocity`: The sphere's own axial translational velocity,
      in the lab frame — the quantity
      [`trajectory.two_way_coupling.solve_force_balanced_velocity`]
      [magnetofluidics_pinn.trajectory.two_way_coupling.solve_force_balanced_velocity]
      searches over. `0.0` recovers a *fixed* (anchored) object.
    - `n_object_surface_points`: Number of collocation points drawn on
      the object's surface each Adam epoch (and once, fixed, for the
      L-BFGS phase).
    - `object_loss_weight`: Weight for the object-surface velocity
      term; defaults to
      [`DEFAULT_BOUNDARY_LOSS_WEIGHT`][magnetofluidics_pinn.config.DEFAULT_BOUNDARY_LOSS_WEIGHT],
      the same weight every other Dirichlet velocity boundary (wall,
      inlet) already uses — physically, the object surface is just
      another such boundary.

    Raises:
    - `ValueError`: If `object_velocity` is not finite, if
      `n_object_surface_points` is not strictly positive, or if
      `object_loss_weight` is not finite and non-negative.
    """
    
    particle_config: ParticleConfig
    particle_velocity: tuple[float, float] = (0.0, 0.0) # Lab frame velocity
    particle_position: tuple[float, float] = (0.0, 0.0) # Instantaneous position
    time: float = 0.0 # Current time
    n_surface_points: int = 200
    object_loss_weight: float = DEFAULT_BOUNDARY_LOSS_WEIGHT
    
    def __post_init__(self) -> None:
        if math.hypot(*self.particle_velocity) <= 0.0:
            raise ValueError("particle_velocity must be non-zero.")
        if self.n_surface_points <= 0:
            raise ValueError(
                "n_surface_points must be strictly positive; "
                f"got {self.n_surface_points!r}."
            )
        _check_if_finite_positive(self.object_loss_weight, "object_loss_weight")
    
    @property
    def particle_velocity_magnitude(self) -> float:
        """Returns the magnitude of the particle velocity."""
        return math.hypot(*self.particle_velocity)
    
    @property
    def axial_particle_velocity(self) -> float:
        """Returns the axial component of the particle velocity."""
        return self.particle_velocity[1]
    
    @property
    def radial_particle_velocity(self) -> float:
        """Returns the radial component of the particle velocity."""
        return self.particle_velocity[0]
    
    @property
    def particle_state(self):
        """Returns a ParticleState object representing the particle's current state."""
        return ParticleState(
            position=torch.tensor(self.particle_position),
            velocity=torch.tensor(self.particle_velocity),
            time=self.time
        )

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration of the training procedure.

    The default recipe is Adam (for global exploration) followed by L-BFGS
    (for local refinement), matching the original PINN training procedure
    [@raissi2019physics]. L-BFGS refinement and gradient clipping were
    added after a manufactured-solution audit traced Phase 1's convergence
    gap to optimization difficulty, not a formulation
    error (`stokes_residual` and the boundary-condition targets are exact
    for the analytical Poiseuille solution) — Adam alone plateaus with the
    PDE residual still orders of magnitude too large; adding an L-BFGS
    refinement phase closed most of that gap in testing (relative L2 error
    against the analytical profile: ~0.94 -> ~0.09 on the same problem).

    Args:
    - `n_interior_points`: Number of interior collocation points per Adam
      epoch.
    - `n_boundary_points`: Number of boundary collocation points per Adam
      epoch.
    - `n_axis_points`: Reserved for a future symmetry-axis collocation
      scheme; not yet consumed by `sampling` or `training` (axis and wall
      regularity are currently enforced structurally instead, via
      `networks.apply_hard_wall_constraint`). Kept here so an eventual
      soft-axis-penalty alternative doesn't require a config schema change.
    - `residual_form`: NEW. Which form of the interior PDE residual
      `training.trainer` evaluates, passed straight through to
      [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
      / [`navier_stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.navier_stokes_residual].
      `"standard"` (the default) reproduces every prior release's behavior
      exactly. `"r_weighted"` evaluates the same equations pre-multiplied by
      $r$ (continuity, $z$-momentum) or $r^2$ ($r$-momentum), which removes
      their $1/r$, $1/r^2$ singularity at the symmetry axis analytically
      instead of by sampling away from it, at the cost of implicitly
      down-weighting how hard the PDE residual is enforced near the axis
      relative to near the wall (the multiplier itself shrinks the residual
      there, independent of accuracy) — see that function's docstring for
      the full derivation and this trade-off.
    - `axis_clearance_fraction`: NEW. Overrides
      [`sampling.collocation._AXIS_CLEARANCE_FRACTION`]
      [magnetofluidics_pinn.sampling.collocation.sample_collocation_points]'s
      default clearance band (as a fraction of `domain.radius`) kept clear
      of the symmetry axis when drawing interior points. `None` (the
      default) keeps that module default and reproduces every prior
      release's sampling exactly. An explicit value lets interior points be
      drawn closer to (down to flush with, at `0.0`) the axis, which is
      only numerically safe under `residual_form="r_weighted"` — see the
      cross-field check in `__post_init__`.
    - `learning_rate`: Initial learning rate for the Adam phase.
    - `n_epochs`: Number of Adam epochs.
    - `gradient_clip_norm`: Maximum gradient norm during the Adam phase, or
      `None` to disable clipping. Guards against the occasional large
      gradient spike that nested second-order autograd (needed for the PDE
      residual) can produce [@wang2021understanding], without changing the
      loss being optimized.
    - `use_lbfgs_refinement`: Whether to follow the Adam phase with L-BFGS
      refinement on a larger, fixed collocation set (L-BFGS is a full-batch
      method: resampling points between its internal iterations, the way
      Adam's per-epoch resampling does, would violate its quasi-Newton
      curvature estimate).
    - `lbfgs_n_interior_points`: Interior points in the fixed L-BFGS batch.
    - `lbfgs_n_boundary_points`: Boundary points in the fixed L-BFGS batch.
    - `lbfgs_rounds`: Number of independent `LBFGS.step()` calls. Each round
      restarts the line search with the previous round's result; in
      testing, accuracy kept improving through 4 rounds (relative L2 error
      against the analytical profile: 0.94 with Adam alone -> 0.66 -> 0.39
      -> 0.24 -> 0.09 after rounds 1-4) before diminishing returns set in.
    - `lbfgs_iterations_per_round`: `max_iter` passed to `torch.optim.LBFGS`
      for each round.
    - `device`: Any device string accepted by `torch.device` (e.g. `"cpu"`,
      `"cuda"`, `"cuda:0"`); resolved automatically when `None`.
    - `random_seed`: Seed used for reproducible sampling and initialization.
    - `verbose`: If `True`, print one progress line every `log_every`
      epochs during the Adam phase (always including the last), and one
      line per round during the L-BFGS phase, plus a one-line summary at
      the end. If `False` (the default), `train` prints nothing.
    - `log_every`: Epoch interval between printed Adam-phase progress
      lines when `verbose=True`; ignored otherwise.
    Raises:
    - `ValueError`: If any of the strictly-positive fields is not strictly
      positive, if `axis_clearance_fraction` is invalid or paired with an
      unsafe `residual_form`, or if `device` does not resolve to `"cuda"`
      or `"cpu"`.
    """
    
    n_interior_points: int = 10_000
    n_boundary_points: int = 2_000
    n_axis_points: int = 512
    residual_form: Literal["standard", "r_weighted"] = "standard"
    axis_clearance_fraction: float | None = None
    learning_rate: float = 1.0e-3
    n_epochs: int = 20_000
    gradient_clip_norm: float | None = 1.0
    use_lbfgs_refinement: bool = True
    lbfgs_n_interior_points: int = 4_000
    lbfgs_n_boundary_points: int = 450
    lbfgs_rounds: int = 4
    lbfgs_iterations_per_round: int = 500
    momentum_loss_weight: float = 1.0
    continuity_loss_weight: float = 20.0
    positivity_loss_weight: float = 1.0
    conservation_loss_weight: float = 20.0
    n_conservation_stations: int = 8
    n_conservation_quadrature_points: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    random_seed: int = 42
    verbose: bool = False
    log_every: int = 100
    
    def __post_init__(self) -> None:
        """Validate configuration parameters and resolve target device."""
        if self.n_interior_points <= 0 or self.n_boundary_points <= 0:
            raise ValueError(
                "n_interior_points and n_boundary_points must be strictly"
                "positive."
            )
        if self.n_boundary_points < 3:
            raise ValueError("n_boundary_points must be at least 3.")
        if self.n_axis_points <= 0:
            raise ValueError("n_axis_points must be strictly positive.")
        if self.axis_clearance_fraction is not None:
            if not math.isfinite(self.axis_clearance_fraction) or not (0.0 <= self.axis_clearance_fraction < 1.0):
                raise ValueError(
                    "axis_clearance_fraction must be finite and lie in [0.0, 1.0); "
                    f"got {self.axis_clearance_fraction!r}."
                )
            if self.axis_clearance_fraction == 0.0 and self.residual_form == "standard":
                raise ValueError(
                    "axis_clearance_fraction=0.0 (sampling flush to the symmetry axis) "
                    "is only numerically safe with residual_form='r_weighted', since the "
                    "'standard' residual is singular at r=0. Either raise "
                    "axis_clearance_fraction above 0.0 or set residual_form='r_weighted'."
                )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be strictly positive.")
        for weight_name in (
                "momentum_loss_weight", "continuity_loss_weight",
                "positivity_loss_weight", "conservation_loss_weight",
        ):
            weight_value = getattr(self, weight_name)
            if not math.isfinite(weight_value) or weight_value < 0.0:
                raise ValueError(f"{weight_name} must be finite and non-negative; got {weight_value!r}.")
        if self.n_conservation_stations <= 0:
            raise ValueError(
                f"n_conservation_stations must be strictly positive; got {self.n_conservation_stations!r}.")
        if self.n_conservation_quadrature_points < 2:
            raise ValueError(
                f"n_conservation_quadrature_points must be at least 2 for trapezoidal "
                f"quadrature; got {self.n_conservation_quadrature_points!r}."
            )
        if self.verbose and self.log_every < 1:
            raise ValueError(f"log_every must be at least 1 when verbose=True; got {self.log_every!r}.")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(
                f"device type must be 'cuda' or 'cpu', got '{self.device}'."
            )
        
    @property
    def torch_device(self) -> torch.device:
        """Lazy device resolution without mutating frozen dataclass fields."""
        return resolve_device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float64


def set_random_seed(seeds: int | None = None) -> None:
    """Set seeds for reproducibility.

    Args:
    - `seeds`: If `None`, do not set any seeds. If an integer, set the
      Python, NumPy, and PyTorch random seeds to this value.
    """
    if seeds is not None:
        import random
        import numpy as np
        
        random.seed(seeds)
        np.random.seed(seeds)
        torch.manual_seed(seeds)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
