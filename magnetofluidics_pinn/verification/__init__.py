"""Quantitative verification utilities for `magnetofluidics_pinn`.

This subpackage exposes the closed-form Hagen-Poiseuille reference solution
(`analytical`), the independent, held-out evaluators a verification
notebook measures a trained network against (`metrics`), and the
pass/fail checklist scaffolding those measurements are reported through
(`reporting`). Nothing here is specific to Phase 1's straight channel in
principle - `metrics` operates on any `(r, z) -> (u_r, u_z, p)` callable,
trained or analytical - though the current `analytical` module's closed
form is Hagen-Poiseuille-specific, matching the straight-channel geometry
Phase 1 targets.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.verification.analytical import (
    analytical_pressure_gradient as analytical_pressure_gradient,
    analytical_tracer_streamline as analytical_tracer_streamline,
    faxen_offset_velocity as faxen_offset_velocity,
    hagen_poiseuille_pressure as hagen_poiseuille_pressure,
    hagen_poiseuille_velocity_field as hagen_poiseuille_velocity_field,
)
from magnetofluidics_pinn.verification.benchmarks import (haberman_sayre_factor as haberman_sayre_factor,
                                                          reference_resistances as reference_resistances,
                                                          shell_force_profile as shell_force_profile,
                                                          shell_force_spread as shell_force_spread,
                                                          stokes_sphere_field_polar as stokes_sphere_field_polar)
from magnetofluidics_pinn.verification.force_checks import (
    cross_section_momentum_force as cross_section_momentum_force, global_momentum_balance as global_momentum_balance,
    residual_by_region as residual_by_region, sphere_traction_force as sphere_traction_force,
    surface_independence_report as surface_independence_report, traction_profile as traction_profile)  # NEW
from magnetofluidics_pinn.verification.metrics import (
    compare_trajectory_to_analytical as compare_trajectory_to_analytical,
    compute_flow_rate_curve as compute_flow_rate_curve, compute_flow_rate_errors as compute_flow_rate_errors,
    compute_global_l2_error as compute_global_l2_error, compute_max_pointwise_error as compute_max_pointwise_error,
    compute_pressure_gradient_error as compute_pressure_gradient_error,
    compute_profile_invariance as compute_profile_invariance, compute_radial_leakage as compute_radial_leakage,
    evaluate_axis_pressure_profile as evaluate_axis_pressure_profile,
    evaluate_held_out_residual as evaluate_held_out_residual,
    evaluate_particle_noslip_error as evaluate_particle_noslip_error, evaluate_residual_grid as evaluate_residual_grid,
    evaluate_structural_constraints as evaluate_structural_constraints,
    evaluate_velocity_profiles as evaluate_velocity_profiles, ProfileEvaluation as ProfileEvaluation,
    summarize_residual as summarize_residual)
from magnetofluidics_pinn.verification.physical_gates import (axis_pressure_jump as axis_pressure_jump,
                                                              divergence_probe as divergence_probe,
                                                              free_velocity_bounds_check as free_velocity_bounds_check,
                                                              generalization_gate as generalization_gate,
                                                              momentum_balance_gate as momentum_balance_gate,
                                                              pressure_drop_gate as pressure_drop_gate,
                                                              reference_tolerance as reference_tolerance)
from magnetofluidics_pinn.verification.reporting import (default_thresholds as default_thresholds,
                                                         evaluate_check as evaluate_check,
                                                         pending_check as pending_check,
                                                         render_verification_table as render_verification_table,
                                                         VerificationCheck as VerificationCheck,
                                                         VerificationThresholds as VerificationThresholds)
from magnetofluidics_pinn.verification.warm_start_gates import (
    warm_start_cost_ratio as warm_start_cost_ratio,
    warm_start_equivalence_gate as warm_start_equivalence_gate)

__all__ = [
    # Analytical reference
    "hagen_poiseuille_velocity_field",
    "hagen_poiseuille_pressure",
    "analytical_pressure_gradient",
    "analytical_tracer_streamline",
    "faxen_offset_velocity",
    # Metrics
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
    # Reporting
    "VerificationCheck",
    "VerificationThresholds",
    "default_thresholds",
    "evaluate_check",
    "pending_check",
    "render_verification_table",
    # Force checks
    "cross_section_momentum_force",
    "global_momentum_balance",
    "sphere_traction_force",
    "surface_independence_report",
    "residual_by_region",
    "traction_profile",
    # Benchmarks
    "stokes_sphere_field_polar",
    "haberman_sayre_factor",
    "reference_resistances",
    "shell_force_profile",
    "shell_force_spread",
    # Physical gates
    "divergence_probe",
    "axis_pressure_jump",
    "pressure_drop_gate",
    "free_velocity_bounds_check",
    "generalization_gate",
    "reference_tolerance",
    "momentum_balance_gate",
    # Warm start gates
    "warm_start_cost_ratio",
    "warm_start_equivalence_gate"
]
