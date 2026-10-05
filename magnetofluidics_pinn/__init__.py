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
    rigid_body_velocity_condition as rigid_body_velocity_condition,
    uniform_field as uniform_field,
)
# CHANGED: added ObstacleConfig and DEFAULT_BOUNDARY_LOSS_WEIGHT — both new
# in config.py, the former replacing the standalone train_around_obstacle()
# entry point, the latter promoted out of training.trainer as its default.
from magnetofluidics_pinn.config import (DEFAULT_BOUNDARY_LOSS_WEIGHT as DEFAULT_BOUNDARY_LOSS_WEIGHT,
                                         DomainConfig as DomainConfig, FluidConfig as FluidConfig,
                                         MagneticFieldConfig as MagneticFieldConfig,
                                         TwoWayCouplingConfig as TwoWayCouplingConfig,
                                         ParticleConfig as ParticleConfig,
                                         TrainingConfig as TrainingConfig,
                                         set_random_seed as set_random_seed)
from magnetofluidics_pinn.device_utils import (
    resolve_device as resolve_device,
    resolve_module_device as resolve_module_device,
    resolve_module_dtype as resolve_module_dtype,
    configure_cuda_matmul_precision as configure_cuda_matmul_precision,
)
from magnetofluidics_pinn.geometry import (
    build_bifurcation_domain as build_bifurcation_domain,
    build_channel_domain as build_channel_domain,
)
from magnetofluidics_pinn.networks import (
    apply_hard_wall_constraint as apply_hard_wall_constraint,
    apply_hard_particle_constraint as apply_hard_particle_constraint,
    build_mlp as build_mlp,
    apply_stream_function_constraint as apply_stream_function_constraint  # NEW
)
from magnetofluidics_pinn.physics import (annular_flow_rate as annular_flow_rate,
                                          axisymmetric_vector_laplacian as axisymmetric_vector_laplacian,
                                          compute_flow_derivatives as compute_flow_derivatives,
                                          axial_flow_rate as axial_flow_rate,
                                          dipole_force as dipole_force,
                                          axial_dipole_force_si as axial_dipole_force_si,
                                          force_scale as force_scale,
                                          make_dimensionless_dipole_force_fn as make_dimensionless_dipole_force_fn,
                                          brenner_poiseuille_wall_factor as brenner_poiseuille_wall_factor,
                                          faxen_corrected_velocity as faxen_corrected_velocity,
                                          navier_stokes_residual as navier_stokes_residual,
                                          poiseuille_reference_flow_rate as poiseuille_reference_flow_rate,
                                          stokes_residual as stokes_residual,
                                          surface_traction_force as surface_traction_force,
                                          haberman_sayre_wall_factor as haberman_sayre_wall_factor,
                                          stokes_drag_reference as stokes_drag_reference,
                                          stokes_sphere_field as stokes_sphere_field,
                                          tube_resistance_reference as tube_resistance_reference,
                                          stokes_reduced_stream_function as stokes_reduced_stream_function,
                                          stokes_sphere_pressure as stokes_sphere_pressure,
                                          oseen_drag_correction as oseen_drag_correction,
                                          )
from magnetofluidics_pinn.sampling import (
    sample_collocation_points as sample_collocation_points,
    sample_collocation_points_with_particle as sample_collocation_points_with_particle,
    sample_exterior_points as sample_exterior_points,  # NEW
)
# Normalization utilities, required to keep solvers dimensionless while
# accepting/reporting physical (SI) quantities at the package boundary.
from magnetofluidics_pinn.scaling import (compute_scales as compute_scales,
                                          nondimensionalize_domain as nondimensionalize_domain,
                                          nondimensionalize_mobility as nondimensionalize_mobility,
                                          nondimensionalize_particle as nondimensionalize_particle,
                                          redimensionalize_domain as redimensionalize_domain, Scales as Scales)
from magnetofluidics_pinn.training import (compose_loss as compose_loss, LOSS_COMPONENT_NAMES as LOSS_COMPONENT_NAMES,
                                           LossHistory as LossHistory, train as train,
                                           TrainingHistory as TrainingHistory,
                                           train_stream_function_flow as train_stream_function_flow,
                                           diagnose_loss_scale as diagnose_loss_scale,
                                           )
from magnetofluidics_pinn.trajectory import (
    integrate_trajectory as integrate_trajectory,
    solve_force_balanced_velocity as solve_force_balanced_velocity,
    advance_particle_two_way as advance_particle_two_way,
    make_dipole_applied_force_fn as make_dipole_applied_force_fn,
    ResistancePair as ResistancePair,
    advance_particle_resistance as advance_particle_resistance,
    force_balanced_velocity as force_balanced_velocity,
    solve_resistance_pair as solve_resistance_pair,
    resistance_position_sweep as resistance_position_sweep,
    position_independence as position_independence,
)
from magnetofluidics_pinn.types import (
    CollocationPoints as CollocationPoints,
    Domain as Domain,
    MagneticFieldSample as MagneticFieldSample,
    ParticleState as ParticleState,
)
# Quantitative verification utilities (closed-form Hagen-Poiseuille
# reference, independent evaluators, and pass/fail checklist scaffolding).
# Imported before `visualization` since `visualization.plotting` consumes
# `verification.metrics.ProfileEvaluation` - see that module's docstring.
from magnetofluidics_pinn.verification import (analytical_pressure_gradient as analytical_pressure_gradient,
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
                                               evaluate_particle_noslip_error as evaluate_particle_noslip_error,
                                               faxen_offset_velocity as faxen_offset_velocity,
                                               hagen_poiseuille_pressure as hagen_poiseuille_pressure,
                                               hagen_poiseuille_velocity_field as hagen_poiseuille_velocity_field,
                                               pending_check as pending_check, ProfileEvaluation as ProfileEvaluation,
                                               render_verification_table as render_verification_table,
                                               summarize_residual as summarize_residual,
                                               VerificationCheck as VerificationCheck,
                                               VerificationThresholds as VerificationThresholds,
                                               cross_section_momentum_force as cross_section_momentum_force,
                                               global_momentum_balance as global_momentum_balance,
                                               sphere_traction_force as sphere_traction_force,
                                               surface_independence_report as surface_independence_report,
                                               residual_by_region as residual_by_region,
                                               )
from magnetofluidics_pinn.visualization import (plot_flow_rate_deviation as plot_flow_rate_deviation,
                                                plot_pressure_gradient_fit as plot_pressure_gradient_fit,
                                                plot_profile_error as plot_profile_error,
                                                plot_radial_leakage as plot_radial_leakage,
                                                plot_residual_heatmap as plot_residual_heatmap,
                                                plot_streamlines as plot_streamlines,
                                                plot_training_history as plot_training_history,
                                                plot_two_way_history as plot_two_way_history,
                                                plot_trajectories as plot_trajectories,
                                                plot_velocity_profile_comparison as plot_velocity_profile_comparison,
                                                plot_particle_flow_field as plot_particle_flow_field,
                                                plot_particle_residual_heatmap as plot_particle_residual_heatmap,
                                                )

__version__ = "0.1.7"  # CHANGED: was "0.1.6" - bumped for the train()/train_around_obstacle() merge and the new ObstacleConfig.

__all__ = [
    "__version__",
    # Autodiff
    "scalar_field_gradient",
    # Configuration
    "DomainConfig",
    "FluidConfig",
    "MagneticFieldConfig",
    "ParticleConfig",
    "TwoWayCouplingConfig",
    "TrainingConfig",
    "set_random_seed",
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
    "rigid_body_velocity_condition",
    "no_slip_condition",
    "uniform_field",
    "gradient_field",
    "biot_savart_field",
    # Physics
    "annular_flow_rate",
    "axisymmetric_vector_laplacian",
    "compute_flow_derivatives",
    "axial_flow_rate",
    "poiseuille_reference_flow_rate",
    "stokes_residual",
    "navier_stokes_residual",
    "dipole_force",
    "axial_dipole_force_si",
    "force_scale",
    "make_dimensionless_dipole_force_fn",
    "brenner_poiseuille_wall_factor",
    "faxen_corrected_velocity",
    "surface_traction_force",
    "haberman_sayre_wall_factor",
    "stokes_drag_reference",
    "stokes_sphere_field",
    "tube_resistance_reference",
    "stokes_reduced_stream_function",
    "stokes_sphere_pressure",
    "oseen_drag_correction",
    # Sampling
    "sample_collocation_points",
    "sample_collocation_points_with_particle",
    "sample_exterior_points",
    # Networks
    "build_mlp",
    "apply_hard_wall_constraint",
    "apply_hard_particle_constraint",
    "apply_stream_function_constraint",
    # Training
    "compose_loss",
    "train",
    "LossHistory",
    "TrainingHistory",
    "LOSS_COMPONENT_NAMES",
    "train_stream_function_flow",
    "diagnose_loss_scale",
    # Trajectory
    "integrate_trajectory",
    "solve_force_balanced_velocity",
    "advance_particle_two_way",
    "make_dipole_applied_force_fn",
    "ResistancePair",
    "advance_particle_resistance",
    "force_balanced_velocity",
    "solve_resistance_pair",
    "resistance_position_sweep",
    "position_independence",
    # Visualization
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
    "plot_two_way_history",  # NEW
    "plot_velocity_profile_comparison",  # NEW
    "plot_profile_error",  # NEW
    "plot_radial_leakage",  # NEW
    "plot_flow_rate_deviation",  # NEW
    "plot_pressure_gradient_fit",  # NEW
    "plot_residual_heatmap",  # NEW
    "plot_particle_flow_field",
    "plot_particle_residual_heatmap",
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
    "configure_cuda_matmul_precision",
    # Verification
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
    "evaluate_particle_noslip_error",
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
    "cross_section_momentum_force",
    "global_momentum_balance",
    "sphere_traction_force",
    "surface_independence_report",
    "residual_by_region",
]
