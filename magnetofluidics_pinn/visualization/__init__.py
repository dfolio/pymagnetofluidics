"""Rendering utilities for `magnetofluidics_pinn`.

Exposes plotting functions for the flow field and particle trajectories,
each a pure function returning a `matplotlib` figure rather than mutating
global plotting state.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.visualization.plotting import (
    plot_streamlines as plot_streamlines,
    plot_trajectories as plot_trajectories,
    plot_training_history as plot_training_history,
)

__all__ = [
    "plot_streamlines",
    "plot_trajectories",
    "plot_training_history",
]
