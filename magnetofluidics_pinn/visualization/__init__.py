"""Rendering utilities for `magnetofluidics_pinn`.

Exposes plotting functions for the flow field, particle trajectories,
training-loss convergence, and (NEW) the quantitative verification metrics
in `verification.metrics` - each a pure function returning a `matplotlib`
figure rather than mutating global plotting state.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.visualization.plotting import (
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

__all__ = [
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
    # NEW: verification-metrics plots
    "plot_velocity_profile_comparison",
    "plot_profile_error",
    "plot_radial_leakage",
    "plot_flow_rate_deviation",
    "plot_pressure_gradient_fit",
    "plot_residual_heatmap",
]