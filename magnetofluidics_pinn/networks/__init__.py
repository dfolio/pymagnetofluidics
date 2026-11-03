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

from magnetofluidics_pinn.networks.lifting import (
    LiftingGeometry as LiftingGeometry,
    LiftingMode as LiftingMode,
    build_lifting_geometry as build_lifting_geometry,
    disturbance_pressure as disturbance_pressure,
    disturbance_reduced as disturbance_reduced,
    fields_from_reduced_stream as fields_from_reduced_stream,
    lifting_pressure as lifting_pressure,
    poiseuille_pressure as poiseuille_pressure,
    poiseuille_reduced as poiseuille_reduced,
    quadratic_shear_pressure as quadratic_shear_pressure,
    quadratic_shear_reduced as quadratic_shear_reduced,
    reduced_lifting as reduced_lifting,
    smooth_unit_step as smooth_unit_step,
    squared_distance as squared_distance,
    taper_weight as taper_weight,
    translating_sphere_pressure as translating_sphere_pressure,
    translating_sphere_reduced as translating_sphere_reduced,
    wall_correction_reduced as wall_correction_reduced,
)
from magnetofluidics_pinn.networks.stream_function import (
    apply_stream_function_constraint as apply_stream_function_constraint,
    StreamFunctionSphereFlow as StreamFunctionSphereFlow,
)
from magnetofluidics_pinn.networks.mlp import build_mlp as build_mlp
from magnetofluidics_pinn.networks.warm_start import (
    transfer_raw_weights as transfer_raw_weights,
    rebuild_for_geometry as rebuild_for_geometry,
    scaled_training_config as scaled_training_config,
)


__all__ = [
    "build_mlp",
    "StreamFunctionSphereFlow",
    "apply_hard_wall_constraint",
    "apply_hard_particle_constraint",
    "apply_stream_function_constraint",
    "transfer_raw_weights",
    "rebuild_for_geometry",
    "scaled_training_config",
    "LiftingMode",
    "LiftingGeometry",
    "build_lifting_geometry",
    "smooth_unit_step",
    "squared_distance",
    "poiseuille_reduced",
    "poiseuille_pressure",
    "translating_sphere_reduced",
    "translating_sphere_pressure",
    "quadratic_shear_reduced",
    "quadratic_shear_pressure",
    "disturbance_reduced",
    "disturbance_pressure",
    "taper_weight",
    "wall_correction_reduced",
    "reduced_lifting",
    "lifting_pressure",
    "fields_from_reduced_stream",
]
