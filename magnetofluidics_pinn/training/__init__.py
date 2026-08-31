"""Loss composition and training loop for `magnetofluidics_pinn`.

Exposes a functional loss-composition utility and the training loop that
consumes it, keeping the loss definition independent of the optimization
procedure.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484); see package __init__.py.
from magnetofluidics_pinn.training.losses import compose_loss as compose_loss
from magnetofluidics_pinn.training.trainer import (
    train as train,
    LossHistory as LossHistory,
    TrainingHistory as TrainingHistory,
)

__all__ = [
    "compose_loss",
    "train",
    "LossHistory",
    "TrainingHistory"
]
