"""Immutable configuration objects for `magnetofluidics_pinn`.

Every stage of the pipeline (geometry, sampling, network, training) is
parameterized by one of the frozen dataclasses defined here. Passing
configuration explicitly, rather than through global state, keeps every
downstream function pure and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from magnetofluidics_pinn.device_utils import resolve_device


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
    """

    kind: Literal["channel", "bifurcation"] = "channel"
    length: float = 3.0e-3  # m
    radius: float = 7.5e-4  # m
    branch_angle: float | None = None


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
    """

    regime: Literal["stokes", "navier_stokes"] = "stokes"
    dynamic_viscosity: float = 1.0e-3
    density: float = 1.0e3
    reference_velocity: float = 1.0e-3
    reference_length: float = DomainConfig.radius
    
    def __post_init__(self) -> None:
        if self.dynamic_viscosity <= 0.0:
            raise ValueError("dynamic_viscosity must be strictly positive.")
        if self.reference_velocity <= 0.0:
            raise ValueError("reference_velocity must be strictly positive.")
        if self.reference_length <= 0.0:
            raise ValueError("reference_length must be strictly positive.")
        if self.regime == "navier_stokes" and self.density <= 0.0:
            raise ValueError("density must be strictly positive for Navier-Stokes flow.")

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
    """

    source: Literal["uniform", "gradient", "biot_savart"] = "uniform"
    magnitude: float = 1.0e-2
    orientation: tuple[float, float] = (1.0, 0.0)
    time_dependent: bool = False


# Add:
@dataclass(frozen=True)
class ParticleConfig:
    radius: float
    magnetic_moment: tuple[float, float]
    
    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("particle radius must be strictly positive.")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration of the training procedure.

    The default recipe is Adam (for global exploration) followed by L-BFGS
    (for local refinement), matching the original PINN training procedure
    (Raissi, Perdikaris, & Karniadakis, 2019). # NEW: L-BFGS refinement and
    gradient clipping were added after a manufactured-solution audit traced
    Phase 1's convergence gap to optimization difficulty, not a formulation
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
      residual) can produce, without changing the loss being optimized.
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
