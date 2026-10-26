"""Neural network construction for `magnetofluidics_pinn`.

Exposes a pure builder function returning a CUDA-aware multilayer
perceptron, with no training logic: construction and optimization are kept
strictly separate.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.networks.constraints import (
    apply_hard_wall_constraint as apply_hard_wall_constraint,
    apply_hard_particle_constraint as apply_hard_particle_constraint,
)
from magnetofluidics_pinn.networks.stream_function import apply_stream_function_constraint as apply_stream_function_constraint  # NEW
from magnetofluidics_pinn.networks.mlp import build_mlp as build_mlp
from magnetofluidics_pinn.networks.warm_start import (transfer_raw_weights as transfer_raw_weights,
                                                      rebuild_for_geometry as rebuild_for_geometry)


__all__ = [
    "build_mlp",
    "apply_hard_wall_constraint",
    "apply_hard_particle_constraint",
    "apply_stream_function_constraint",
    "transfer_raw_weights",
    "rebuild_for_geometry",
]
