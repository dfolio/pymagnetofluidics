"""PDE residuals and magnetic forcing for `magnetofluidics_pinn`.

This subpackage exposes the flow PDE residuals (Stokes and Navier-Stokes)
evaluated through automatic differentiation of the network's output, and the
magnetic dipole force computed from a prescribed field.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.physics.conservation import (annular_flow_rate as annular_flow_rate,
                                                       axial_flow_rate as axial_flow_rate,
                                                       poiseuille_reference_flow_rate as poiseuille_reference_flow_rate)
from magnetofluidics_pinn.physics.fluid_residuals import (
    axisymmetric_vector_laplacian as axisymmetric_vector_laplacian,
    compute_flow_derivatives as compute_flow_derivatives,
    navier_stokes_residual as navier_stokes_residual,
    stokes_residual as stokes_residual,
)
from magnetofluidics_pinn.physics.hydrodynamic_drag import (
    brenner_poiseuille_wall_factor as brenner_poiseuille_wall_factor,
    faxen_corrected_velocity as faxen_corrected_velocity,
    surface_traction_force as surface_traction_force,
)
from magnetofluidics_pinn.physics.stokes_sphere import (  # NEW
    haberman_sayre_wall_factor as haberman_sayre_wall_factor,
    stokes_drag_reference as stokes_drag_reference,
    stokes_sphere_field as stokes_sphere_field,
    tube_resistance_reference as tube_resistance_reference,
    stokes_reduced_stream_function as stokes_reduced_stream_function,
    stokes_sphere_pressure as stokes_sphere_pressure,
    oseen_drag_correction as oseen_drag_correction,
)
from magnetofluidics_pinn.physics.magnetic_forcing import (
    dipole_force as dipole_force,
    axial_dipole_force_si as axial_dipole_force_si,
    force_scale as force_scale,
    make_dimensionless_dipole_force_fn as make_dimensionless_dipole_force_fn,
)

__all__ = [
    "annular_flow_rate",
    "axisymmetric_vector_laplacian",
    "compute_flow_derivatives",
    "stokes_residual",
    "navier_stokes_residual",
    "dipole_force",
    "axial_dipole_force_si",
    "force_scale",
    "make_dimensionless_dipole_force_fn",
    "faxen_corrected_velocity",
    "surface_traction_force",
    "axial_flow_rate",
    "poiseuille_reference_flow_rate",
    "haberman_sayre_wall_factor",
    "stokes_drag_reference",
    "stokes_sphere_field",
    "tube_resistance_reference",
    "stokes_reduced_stream_function",
    "stokes_sphere_pressure",
    "oseen_drag_correction",
]
