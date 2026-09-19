"""Particle-trajectory integration for `magnetofluidics_pinn`.

Exposes the function integrating a magnetic particle's motion through a
trained velocity field and a prescribed magnetic force, for both a single
particle and a swarm.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.trajectory.integrator import (
    integrate_trajectory as integrate_trajectory,
)
from magnetofluidics_pinn.trajectory.two_way_coupling import (
    solve_force_balanced_velocity as solve_force_balanced_velocity,
    advance_particle_two_way as advance_particle_two_way,
)

__all__ = [
    "integrate_trajectory",
    "solve_force_balanced_velocity",
    "advance_particle_two_way",
]
