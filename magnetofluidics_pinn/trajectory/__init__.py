"""Particle-trajectory integration for `magnetofluidics_pinn`.

Exposes the function integrating a magnetic particle's motion through a
trained velocity field and a prescribed magnetic force, for both a single
particle and a swarm.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.trajectory.integrator import (
    integrate_trajectory as integrate_trajectory,
)

__all__ = [
    "integrate_trajectory",
]
