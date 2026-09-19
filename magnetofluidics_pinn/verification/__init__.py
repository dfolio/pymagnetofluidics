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
from magnetofluidics_pinn.verification.metrics import (
    ProfileEvaluation as ProfileEvaluation,
    compare_trajectory_to_analytical as compare_trajectory_to_analytical,
    compute_flow_rate_curve as compute_flow_rate_curve,
    compute_flow_rate_errors as compute_flow_rate_errors,
    compute_global_l2_error as compute_global_l2_error,
    compute_max_pointwise_error as compute_max_pointwise_error,
    compute_pressure_gradient_error as compute_pressure_gradient_error,
    compute_profile_invariance as compute_profile_invariance,
    compute_radial_leakage as compute_radial_leakage,
    evaluate_axis_pressure_profile as evaluate_axis_pressure_profile,
    evaluate_held_out_residual as evaluate_held_out_residual,
    evaluate_residual_grid as evaluate_residual_grid,
    evaluate_structural_constraints as evaluate_structural_constraints,
    evaluate_velocity_profiles as evaluate_velocity_profiles,
    summarize_residual as summarize_residual,
)
from magnetofluidics_pinn.verification.reporting import (
    VerificationCheck as VerificationCheck,
    VerificationThresholds as VerificationThresholds,
    default_thresholds as default_thresholds,
    evaluate_check as evaluate_check,
    pending_check as pending_check,
    render_verification_table as render_verification_table,
)

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
    "compare_trajectory_to_analytical",
    # Reporting
    "VerificationCheck",
    "VerificationThresholds",
    "default_thresholds",
    "evaluate_check",
    "pending_check",
    "render_verification_table",
]
