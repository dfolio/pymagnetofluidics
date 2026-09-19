"""PDE residuals and magnetic forcing for `magnetofluidics_pinn`.

This subpackage exposes the flow PDE residuals (Stokes and Navier-Stokes)
evaluated through automatic differentiation of the network's output, and the
magnetic dipole force computed from a prescribed field.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.physics.conservation import (axial_flow_rate as axial_flow_rate,
                                                       poiseuille_reference_flow_rate as poiseuille_reference_flow_rate)
from magnetofluidics_pinn.physics.fluid_residuals import (
    axisymmetric_vector_laplacian as axisymmetric_vector_laplacian,
    compute_flow_derivatives as compute_flow_derivatives,
    navier_stokes_residual as navier_stokes_residual,
    stokes_residual as stokes_residual,
)
from magnetofluidics_pinn.physics.hydrodynamic_drag import (
    faxen_corrected_velocity as faxen_corrected_velocity,
    surface_traction_force as surface_traction_force,
)
from magnetofluidics_pinn.physics.magnetic_forcing import dipole_force as dipole_force

__all__ = [
    "axisymmetric_vector_laplacian",
    "compute_flow_derivatives",
    "stokes_residual",
    "navier_stokes_residual",
    "dipole_force",
    "faxen_corrected_velocity",
    "surface_traction_force",
    "axial_flow_rate",
    "poiseuille_reference_flow_rate"
]
