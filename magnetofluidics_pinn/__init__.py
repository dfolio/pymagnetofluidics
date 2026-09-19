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
    MagneticFieldConfig as MagneticFieldConfig,
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
    axial_flow_rate as axial_flow_rate,
    poiseuille_reference_flow_rate as poiseuille_reference_flow_rate
)
from magnetofluidics_pinn.sampling import (
    sample_collocation_points as sample_collocation_points,
)
# Normalization utilities, required to keep solvers dimensionless while
# accepting/reporting physical (SI) quantities at the package boundary.
from magnetofluidics_pinn.scaling import (compute_scales as compute_scales,
                                          nondimensionalize_domain as nondimensionalize_domain,
                                          nondimensionalize_particle as nondimensionalize_particle,
                                          nondimensionalize_mobility as nondimensionalize_mobility,
                                          redimensionalize_domain as redimensionalize_domain, Scales as Scales)
from magnetofluidics_pinn.training import (compose_loss as compose_loss,
                                           train as train,
                                           LossHistory as LossHistory,
                                           TrainingHistory as TrainingHistory,
                                           LOSS_COMPONENT_NAMES as LOSS_COMPONENT_NAMES
                                           )
from magnetofluidics_pinn.trajectory import integrate_trajectory as integrate_trajectory
from magnetofluidics_pinn.types import (
    CollocationPoints as CollocationPoints,
    Domain as Domain,
    MagneticFieldSample as MagneticFieldSample,
    ParticleState as ParticleState,
)
# NEW: quantitative verification utilities (closed-form Hagen-Poiseuille
# reference, independent evaluators, and pass/fail checklist scaffolding).
# Imported before `visualization` since `visualization.plotting` consumes
# `verification.metrics.ProfileEvaluation` - see that module's docstring.
from magnetofluidics_pinn.verification import (
    ProfileEvaluation as ProfileEvaluation,
    VerificationCheck as VerificationCheck,
    VerificationThresholds as VerificationThresholds,
    analytical_pressure_gradient as analytical_pressure_gradient,
    analytical_tracer_streamline as analytical_tracer_streamline,
    compare_trajectory_to_analytical as compare_trajectory_to_analytical,
    compute_flow_rate_curve as compute_flow_rate_curve,
    compute_flow_rate_errors as compute_flow_rate_errors,
    compute_global_l2_error as compute_global_l2_error,
    compute_max_pointwise_error as compute_max_pointwise_error,
    compute_pressure_gradient_error as compute_pressure_gradient_error,
    compute_profile_invariance as compute_profile_invariance,
    compute_radial_leakage as compute_radial_leakage,
    default_thresholds as default_thresholds,
    evaluate_axis_pressure_profile as evaluate_axis_pressure_profile,
    evaluate_check as evaluate_check,
    evaluate_held_out_residual as evaluate_held_out_residual,
    evaluate_residual_grid as evaluate_residual_grid,
    evaluate_structural_constraints as evaluate_structural_constraints,
    evaluate_velocity_profiles as evaluate_velocity_profiles,
    faxen_offset_velocity as faxen_offset_velocity,
    hagen_poiseuille_pressure as hagen_poiseuille_pressure,
    hagen_poiseuille_velocity_field as hagen_poiseuille_velocity_field,
    pending_check as pending_check,
    render_verification_table as render_verification_table,
    summarize_residual as summarize_residual,
)
from magnetofluidics_pinn.visualization import (
    plot_flow_rate_deviation as plot_flow_rate_deviation,  # NEW
    plot_pressure_gradient_fit as plot_pressure_gradient_fit,  # NEW
    plot_profile_error as plot_profile_error,  # NEW
    plot_radial_leakage as plot_radial_leakage,  # NEW
    plot_residual_heatmap as plot_residual_heatmap,  # NEW
    plot_streamlines as plot_streamlines,
    plot_trajectories as plot_trajectories,
    plot_training_history as plot_training_history,
    plot_velocity_profile_comparison as plot_velocity_profile_comparison,  # NEW
)

__version__ = "0.1.6"  # CHANGED: was "0.1.5" - bumped for the new `verification` subpackage.

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
    "axial_flow_rate",
    "poiseuille_reference_flow_rate",
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
    "LOSS_COMPONENT_NAMES",
    # Trajectory
    "integrate_trajectory",
    # Visualization
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
    "plot_velocity_profile_comparison",  # NEW
    "plot_profile_error",  # NEW
    "plot_radial_leakage",  # NEW
    "plot_flow_rate_deviation",  # NEW
    "plot_pressure_gradient_fit",  # NEW
    "plot_residual_heatmap",  # NEW
    # Scaling
    "Scales",
    "compute_scales",
    "nondimensionalize_domain",
    "redimensionalize_domain",
    "nondimensionalize_particle",
    "nondimensionalize_mobility",  # NEW: was previously only reachable via the `scaling` submodule directly.
    # Device management
    "resolve_device",
    "resolve_module_device",
    "resolve_module_dtype",
    # NEW: Verification
    "ProfileEvaluation",
    "summarize_residual",
    "evaluate_structural_constraints",
    "evaluate_velocity_profiles",
    "compute_global_l2_error",
    "compute_max_pointwise_error",
    "compute_profile_invariance",
    "compute_radial_leakage",
    "compute_flow_rate_curve",
    "compute_flow_rate_errors",
    "evaluate_axis_pressure_profile",
    "compute_pressure_gradient_error",
    "evaluate_held_out_residual",
    "evaluate_residual_grid",
    "compare_trajectory_to_analytical",
    "hagen_poiseuille_velocity_field",
    "hagen_poiseuille_pressure",
    "analytical_pressure_gradient",
    "analytical_tracer_streamline",
    "faxen_offset_velocity",
    "VerificationCheck",
    "VerificationThresholds",
    "default_thresholds",
    "evaluate_check",
    "pending_check",
    "render_verification_table",
]
