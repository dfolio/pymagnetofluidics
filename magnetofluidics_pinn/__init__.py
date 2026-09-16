"""magnetofluidics_pinn: physics-informed neural networks for magnetic
particle transport in viscous microfluidic flows.

The public API is re-exported here so that common workflows only need the
top-level package, imported as:

```python
import magnetofluidics_pinn as mfp
```

Submodules remain independently importable (e.g.,
`magnetofluidics_pinn.physics.fluid_residuals`) for finer-grained access.
"""

# NOTE: every import below uses the explicit `as <same_name>` re-export idiom
# (PEP 484). Without it, static analyzers (PyCharm, mypy --no-implicit-reexport)
# cannot distinguish "re-exported public API" from "leftover unused import" and
# flag every symbol as unreferenced, even though `__all__` lists them.
from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient as scalar_field_gradient

from magnetofluidics_pinn.boundary_conditions import (
    biot_savart_field as biot_savart_field,
    gradient_field as gradient_field,
    inlet_velocity_condition as inlet_velocity_condition,
    no_slip_condition as no_slip_condition,
    outlet_pressure_condition as outlet_pressure_condition,
    uniform_field as uniform_field,
)
from magnetofluidics_pinn.config import (
    DomainConfig as DomainConfig,
    MagneticFieldConfig as FieldConfig,
    FluidConfig as FluidConfig,
    ParticleConfig as ParticleConfig,
    TrainingConfig as TrainingConfig,
)
from magnetofluidics_pinn.device_utils import (
    resolve_device as resolve_device,
    resolve_module_device as resolve_module_device,
    resolve_module_dtype as resolve_module_dtype,
)
from magnetofluidics_pinn.geometry import (
    build_bifurcation_domain as build_bifurcation_domain,
    build_channel_domain as build_channel_domain,
)
from magnetofluidics_pinn.networks import (
    apply_hard_wall_constraint as apply_hard_wall_constraint,
    build_mlp as build_mlp,
)
from magnetofluidics_pinn.physics import (
    dipole_force as dipole_force,
    faxen_corrected_velocity as faxen_corrected_velocity,
    navier_stokes_residual as navier_stokes_residual,
    stokes_residual as stokes_residual,
)
from magnetofluidics_pinn.sampling import (
    sample_collocation_points as sample_collocation_points,
)
# Normalization utilities, required to keep solvers dimensionless while
# accepting/reporting physical (SI) quantities at the package boundary.
from magnetofluidics_pinn.scaling import (compute_scales as compute_scales,
                                          nondimensionalize_domain as nondimensionalize_domain,
                                          nondimensionalize_particle as nondimensionalize_particle,
                                          redimensionalize_domain as redimensionalize_domain, Scales as Scales)
from magnetofluidics_pinn.training import (compose_loss as compose_loss,
                                           train as train,
                                           LossHistory as LossHistory,
                                           TrainingHistory as TrainingHistory,
                                           )
from magnetofluidics_pinn.trajectory import integrate_trajectory as integrate_trajectory
from magnetofluidics_pinn.types import (
    CollocationPoints as CollocationPoints,
    Domain as Domain,
    MagneticFieldSample as FieldSample,
    ParticleState as ParticleState,
)
from magnetofluidics_pinn.visualization import (
    plot_streamlines as plot_streamlines,
    plot_trajectories as plot_trajectories,
    plot_training_history as plot_training_history,
)

__version__ = "0.1.5"

__all__ = [
    "__version__",
    # Autodiff
    "scalar_field_gradient",
    # Configuration
    "DomainConfig",
    "FluidConfig",
    "MagneticFieldConfig",
    "ParticleConfig",
    "TrainingConfig",
    # Data types
    "Domain",
    "CollocationPoints",
    "MagneticFieldSample",
    "ParticleState",
    # Geometry
    "build_channel_domain",
    "build_bifurcation_domain",
    # Boundary conditions
    "inlet_velocity_condition",
    "outlet_pressure_condition",
    "no_slip_condition",
    "uniform_field",
    "gradient_field",
    "biot_savart_field",
    # Physics
    "stokes_residual",
    "navier_stokes_residual",
    "dipole_force",
    "faxen_corrected_velocity",
    # Sampling
    "sample_collocation_points",
    # Networks
    "build_mlp",
    "apply_hard_wall_constraint",
    # Training
    "compose_loss",
    "train",
    "LossHistory",
    "TrainingHistory",
    # Trajectory
    "integrate_trajectory",
    # Visualization
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
    # Scaling
    "Scales",
    "compute_scales",
    "nondimensionalize_domain",
    "redimensionalize_domain",
    "nondimensionalize_particle",
    # Device management
    "resolve_device",
    "resolve_module_device",
    "resolve_module_dtype",
]
