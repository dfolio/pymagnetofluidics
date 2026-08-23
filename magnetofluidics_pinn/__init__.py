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
    FieldConfig as FieldConfig,
    FluidConfig as FluidConfig,
    TrainingConfig as TrainingConfig,
)
from magnetofluidics_pinn.geometry import (
    build_bifurcation_domain as build_bifurcation_domain,
    build_channel_domain as build_channel_domain,
)
from magnetofluidics_pinn.networks import build_mlp as build_mlp
from magnetofluidics_pinn.physics import (
    dipole_force as dipole_force,
    navier_stokes_residual as navier_stokes_residual,
    stokes_residual as stokes_residual,
)
from magnetofluidics_pinn.sampling import (
    sample_collocation_points as sample_collocation_points,
)
# NEW: normalization utilities, required to keep solvers dimensionless while
# accepting/reporting physical (SI) quantities at the package boundary.
from magnetofluidics_pinn.scaling import (compute_scales as compute_scales,
                                          nondimensionalize_domain as nondimensionalize_domain,
                                          redimensionalize_domain as redimensionalize_domain, Scales as Scales)
from magnetofluidics_pinn.training import compose_loss as compose_loss, train as train
from magnetofluidics_pinn.trajectory import integrate_trajectory as integrate_trajectory
from magnetofluidics_pinn.types import (
    CollocationPoints as CollocationPoints,
    Domain as Domain,
    FieldSample as FieldSample,
    ParticleState as ParticleState,
)
from magnetofluidics_pinn.visualization import (
    plot_streamlines as plot_streamlines,
    plot_trajectories as plot_trajectories,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # Configuration
    "DomainConfig",
    "FluidConfig",
    "FieldConfig",
    "TrainingConfig",
    # Data types
    "Domain",
    "CollocationPoints",
    "FieldSample",
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
    # Sampling
    "sample_collocation_points",
    # Networks
    "build_mlp",
    # Training
    "compose_loss",
    "train",
    # Trajectory
    "integrate_trajectory",
    # Visualization
    "plot_streamlines",
    "plot_trajectories",
    # Scaling
    "Scales",
    "compute_scales",
    "nondimensionalize_domain",
    "redimensionalize_domain",
]
