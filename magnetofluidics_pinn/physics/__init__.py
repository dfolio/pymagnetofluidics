"""PDE residuals and magnetic forcing for `magnetofluidics_pinn`.

This subpackage exposes the flow PDE residuals (Stokes and Navier-Stokes)
evaluated through automatic differentiation of the network's output, and the
magnetic dipole force computed from a prescribed field.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.physics.fluid_residuals import (
    navier_stokes_residual as navier_stokes_residual,
    stokes_residual as stokes_residual,
)
from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force as dipole_force

__all__ = [
    "stokes_residual",
    "navier_stokes_residual",
    "dipole_force",
]
