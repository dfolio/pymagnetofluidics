"""Collocation-point sampling for `magnetofluidics_pinn`.

Exposes pure, seed-driven functions generating the interior, boundary, and
initial points at which physics and boundary-condition residuals are
evaluated during training.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.sampling.collocation import (
    sample_collocation_points as sample_collocation_points,
    sample_collocation_points_with_obstacle as sample_collocation_points_with_obstacle,
)

__all__ = [
    "sample_collocation_points",
    "sample_collocation_points_with_obstacle",
]

