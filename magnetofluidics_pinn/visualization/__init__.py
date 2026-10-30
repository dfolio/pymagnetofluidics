"""Rendering utilities for `magnetofluidics_pinn`.

Exposes plotting functions for the flow field, particle trajectories,
training-loss convergence, and (NEW) the quantitative verification metrics
in `verification.metrics` - each a pure function returning a `matplotlib`
figure rather than mutating global plotting state.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.visualization.plotting import (
    plot_flow_rate_deviation as plot_flow_rate_deviation,
    plot_particle_flow_animation as plot_particle_flow_animation,
    plot_particle_flow_field as plot_particle_flow_field,
    plot_particle_residual_heatmap as plot_particle_residual_heatmap,
    plot_pressure_gradient_fit as plot_pressure_gradient_fit,
    plot_pressure_through_particle as plot_pressure_through_particle,
    plot_profile_error as plot_profile_error,
    plot_radial_leakage as plot_radial_leakage,
    plot_residual_by_region as plot_residual_by_region,
    plot_residual_heatmap as plot_residual_heatmap,
    plot_resistance_pair_comparison as plot_resistance_pair_comparison,
    plot_resistance_position_sweep as plot_resistance_position_sweep,
    plot_station_profile_comparison as plot_station_profile_comparison,
    plot_stream_function_field as plot_stream_function_field,
    plot_stream_loss_history as plot_stream_loss_history,
    plot_streamlines as plot_streamlines,
    plot_surface_independence as plot_surface_independence,
    plot_traction_uniformity as plot_traction_uniformity,
    plot_training_history as plot_training_history,
    plot_trajectories as plot_trajectories,
    plot_two_way_history as plot_two_way_history,
    plot_velocity_profile_comparison as plot_velocity_profile_comparison,
    plot_flow_rate_reconnection as plot_flow_rate_reconnection,
    plot_quadrature_convergence as plot_quadrature_convergence,
)

__all__ = [
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
    "plot_two_way_history",
    "plot_velocity_profile_comparison",
    "plot_profile_error",
    "plot_radial_leakage",
    "plot_flow_rate_deviation",
    "plot_pressure_gradient_fit",
    "plot_residual_heatmap",
    "plot_particle_flow_field",
    "plot_particle_residual_heatmap",
    "plot_traction_uniformity",
    "plot_resistance_pair_comparison",
    "plot_resistance_position_sweep",
    "plot_residual_by_region",
    "plot_surface_independence",
    "plot_stream_function_field",
    "plot_particle_flow_animation",
    "plot_stream_loss_history",
    "plot_station_profile_comparison",
    "plot_pressure_through_particle",
    "plot_flow_rate_reconnection",
    "plot_quadrature_convergence",
]
