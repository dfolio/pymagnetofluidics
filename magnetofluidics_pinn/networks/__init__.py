"""Neural network construction for `magnetofluidics_pinn`.

Exposes a pure builder function returning a CUDA-aware multilayer
perceptron, with no training logic: construction and optimization are kept
strictly separate.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.networks.constraints import (
    apply_hard_wall_constraint as apply_hard_wall_constraint,
)
from magnetofluidics_pinn.networks.mlp import build_mlp as build_mlp

__all__ = [
    "build_mlp",
    "apply_hard_wall_constraint",
]
