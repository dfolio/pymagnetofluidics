"""Immutable configuration objects for `magnetofluidics_pinn`.

Every stage of the pipeline (geometry, sampling, network, training) is
parameterized by one of the frozen dataclasses defined here. Passing
configuration explicitly, rather than through global state, keeps every
downstream function pure and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    length: float = 3.0e-3
    radius: float = 7.5e-4
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
    # CHANGED: reference_length now documented as the nondimensionalization
    # length scale (kept at the vessel-radius order of magnitude, 1.0e-4 m).
    reference_length: float = 1.0e-6


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


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration of the training procedure.

    Args:
    - `n_interior_points`: Number of interior collocation points per epoch.
    - `n_boundary_points`: Number of boundary collocation points per epoch.
    - `learning_rate`: Initial learning rate for the optimizer.
    - `n_epochs`: Total number of training epochs.
    - `device`: Either `"cuda"` or `"cpu"`; resolved automatically when `None`.
    - `random_seed`: Seed used for reproducible sampling and initialization.
    """

    n_interior_points: int = 10_000
    n_boundary_points: int = 2_000
    learning_rate: float = 1.0e-3
    n_epochs: int = 20_000
    device: Literal["cuda", "cpu"] | None = None
    random_seed: int = 42
