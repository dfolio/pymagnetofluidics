"""Boundary and source conditions for `magnetofluidics_pinn`.

This subpackage exposes the flow boundary conditions (inlet/outlet
pressure or velocity, no-slip walls) and the prescribed magnetic field,
which is treated as an input to the solver rather than a solved unknown.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.boundary_conditions.flow_bc import (
    inlet_velocity_condition as inlet_velocity_condition,
    no_slip_condition as no_slip_condition,
    outlet_pressure_condition as outlet_pressure_condition,
)
from magnetofluidics_pinn.boundary_conditions.magnetic_field_bc import (
    biot_savart_field as biot_savart_field,
    gradient_field as gradient_field,
    uniform_field as uniform_field,
)

__all__ = [
    "inlet_velocity_condition",
    "outlet_pressure_condition",
    "no_slip_condition",
    "uniform_field",
    "gradient_field",
    "biot_savart_field",
]
