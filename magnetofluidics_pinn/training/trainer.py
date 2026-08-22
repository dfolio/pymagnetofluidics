"""Training loop.

Runs the optimization procedure given a network, a domain, and a training
configuration, returning a new, trained network instance rather than
mutating the one passed in.
"""

from __future__ import annotations

from torch import nn

from magnetofluidics_pinn.config import FieldConfig, FluidConfig, TrainingConfig
from magnetofluidics_pinn.types import Domain


def train(
    network: nn.Module,
    domain: Domain,
    fluid_config: FluidConfig,
    field_config: FieldConfig,
    training_config: TrainingConfig,
) -> nn.Module:
    """Train a network to satisfy the flow PDE and boundary conditions.

    Args:
    - `network`: Network instance to train, e.g., built by
      [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp].
    - `domain`: Vessel geometry the network is trained on.
    - `fluid_config`: Fluid configuration selecting the flow regime.
    - `field_config`: Prescribed magnetic field configuration (used only if
      the training loop also tracks particle trajectories during training).
    - `training_config`: Training hyperparameters (epochs, learning rate,
      sampling sizes, device, random seed).

    Returns:
    - A trained `torch.nn.Module` instance.

    Raises:
    - `ValueError`: If `training_config.n_epochs` is not strictly positive.
    """
    if training_config.n_epochs <= 0:
        raise ValueError("training_config.n_epochs must be strictly positive.")
    raise NotImplementedError("Implementation scheduled for Step 2.")
