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
from dataclasses import dataclass
from typing import Literal

from magnetofluidics_pinn.device_utils import resolve_device

# Numerical tolerance for "is this vector a unit vector" checks; loose
# enough to tolerate float rounding, tight enough to catch a genuinely
# mis-specified direction. Matches the tolerance used at the point of use
# in `boundary_conditions.magnetic_field_bc.uniform_field`.
_UNIT_NORM_TOLERANCE = 1.0e-6


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
    - `branch_angle`: Branch half-angle in radians, required only when
      `kind == "bifurcation"`.
      
    Raises:
    - `ValueError`: If `length` or `radius` is not finite and strictly
      positive, if `kind == "bifurcation"` and `branch_angle` is not a
      finite number, or if `kind == "channel"` and `branch_angle` is not
      `None`.
    """
    
    kind: Literal["channel", "bifurcation"] = "channel"
    length: float = 3.0e-3  # m
    radius: float = 7.5e-4  # m
    branch_angle: float | None = None
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError(f"length must be finite and strictly positive; got {self.length!r}.")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(f"radius must be finite and strictly positive; got {self.radius!r}.")
        if self.kind == "bifurcation":
            if self.branch_angle is None or not math.isfinite(self.branch_angle):
                raise ValueError("kind='bifurcation' requires a finite branch_angle.")
        elif self.branch_angle is not None:
            raise ValueError(
                "branch_angle is only meaningful when kind='bifurcation'; "
                f"got kind={self.kind!r} with branch_angle={self.branch_angle!r}."
            )


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
    dynamic_viscosity: float = 1.0e-3
    density: float = 1.0e3
    reference_velocity: float = 1.0e-3
    reference_length: float = DomainConfig.radius
    
    def __post_init__(self) -> None:
        for field_name in ("dynamic_viscosity", "density", "reference_velocity", "reference_length"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and strictly positive; got {value!r}.")


@dataclass(frozen=True)
class FieldConfig:
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
    magnitude: float = 1.0e-2
    orientation: tuple[float, float] = (1.0, 0.0)
    time_dependent: bool = False
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude) or self.magnitude <= 0.0:
            raise ValueError(f"magnitude must be finite and strictly positive; got {self.magnitude!r}.")
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
    - `radius`: Particle radius, in meter (e.g., a few micrometers, so
      around `1.0e-6` to `1.0e-5`). Used only for the Faxén-law finite-size
      correction to the ambient flow velocity
      ([`physics.hydrodynamic_drag`][magnetofluidics_pinn.physics.hydrodynamic_drag]);
      `0.0` recovers the exact point-particle limit.
    - `magnetic_moment`: Magnetic dipole moment vector `(m_r, m_z)`, in
      ampere-square-meter (`A m^2`), giving both the particle's magnetic
      "strength" and its (fixed, for Phase 1) orientation. Passed to
      [`trajectory.integrate_trajectory`][magnetofluidics_pinn.trajectory.integrator.integrate_trajectory]
      as its `magnetic_moment` argument only once already expressed in the
      dimensionless units that function's docstring describes — this
      config field records the *physical* moment for bookkeeping, but does
      not itself perform that conversion (see the caveat above).
    - `position`: Initial position of the particle, in meter (e.g., `(0.0, 0.0)` for the center of the domain).

    Raises:
    - `ValueError`: If `radius` is not finite and non-negative, if either
      component of `magnetic_moment` is not finite, or if
      `magnetic_moment` is the zero vector.
      
    Todo: manage 2D/3D position
    """
    
    kind: Literal["spherical", "swarms", "cylinder", "spheroid"] = "spherical"
    radius: float = 0.0
    length: float = 0.0
    aspect_ratio: float = 1.0
    number: int = 1
    magnetic_moment: tuple[float, float] = (1.0e-13, 0.0)
    position: tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError(f"radius must be finite and non-negative; got {self.radius!r}.")
        if not all(math.isfinite(component) for component in self.magnetic_moment):
            raise ValueError(f"magnetic_moment components must be finite; got {self.magnetic_moment!r}.")
        if math.hypot(*self.magnetic_moment) <= 0.0:
            raise ValueError("magnetic_moment must be non-zero.")
        if self.kind == "swarms" and self.number <= 0:
            raise ValueError(f"number of particles must be positive for swarms; got {self.number!r}.")
        if self.kind == "cylinder" and self.length <= 0:
            raise ValueError(f"length of cylinder must be positive; got {self.length!r}.")

    @property
    def max_surface_extension(self) -> float:
        """Returns the maximum radial extension from the centre of mass."""
        if self.kind == "cylinder":
            return math.hypot(self.radius, self.length / 2.0)
        return self.radius
    

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
    - `learning_rate`: Initial learning rate for the Adam phase.
    - `n_epochs`: Number of Adam epochs.
    - `device`: Any device string accepted by `torch.device` (e.g. `"cpu"`,
      `"cuda"`, `"cuda:0"`); resolved automatically when `None`.
    - `random_seed`: Seed used for reproducible sampling and initialization.
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
    """
    
    n_interior_points: int = 10_000
    n_boundary_points: int = 2_000
    n_axis_points: int = 512
    learning_rate: float = 1.0e-3
    n_epochs: int = 20_000
    device: str | None = None
    random_seed: int = 42
    gradient_clip_norm: float | None = 1.0
    use_lbfgs_refinement: bool = True
    lbfgs_n_interior_points: int = 4_000
    lbfgs_n_boundary_points: int = 450
    lbfgs_rounds: int = 4
    lbfgs_iterations_per_round: int = 500
    
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
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be strictly positive.")
        resolved_device = resolve_device(self.device)
        if resolved_device.type not in ("cuda", "cpu"):
            raise ValueError(
                f"device type must be 'cuda' or 'cpu', got '{resolved_device.type}'."
            )
        
        # Dataclass is frozen; mutate through object.__setattr__
        object.__setattr__(self, "device", resolved_device)
