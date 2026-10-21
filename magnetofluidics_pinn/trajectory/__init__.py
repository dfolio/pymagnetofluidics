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
    make_dipole_applied_force_fn as make_dipole_applied_force_fn,
    affine_force_balance as affine_force_balance,
    ResistanceResult as ResistanceResult,
    solve_resistance_problems as solve_resistance_problems,
    solve_resistance_problems_stream as solve_resistance_problems_stream,  # NEW
)
from magnetofluidics_pinn.trajectory.mobility import (  # NEW
    ResistancePair as ResistancePair,
    advance_particle_resistance as advance_particle_resistance,
    force_balanced_velocity as force_balanced_velocity,
    solve_resistance_pair as solve_resistance_pair,
    resistance_position_sweep as resistance_position_sweep,
    position_independence as position_independence,
)
__all__ = [
    "integrate_trajectory",
    "solve_force_balanced_velocity",
    "advance_particle_two_way",
    "make_dipole_applied_force_fn",
    "affine_force_balance",
    "ResistanceResult",
    "solve_resistance_problems",
    "ResistancePair",
    "advance_particle_resistance",
    "solve_resistance_pair",
    "solve_resistance_problems_stream",
    "resistance_position_sweep",
    "position_independence",
]
