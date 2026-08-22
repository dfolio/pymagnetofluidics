"""Checkpointing and configuration (de)serialization.

Provides pure, side-effect-isolated helpers for saving and loading trained
networks and their associated configuration, so that a run can be resumed or
audited later. File I/O is confined to this module.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from magnetofluidics_pinn.config import (
    DomainConfig,
    FieldConfig,
    FluidConfig,
    TrainingConfig,
)


def save_checkpoint(
    network: nn.Module,
    checkpoint_path: Path,
    domain_config: DomainConfig,
    fluid_config: FluidConfig,
    field_config: FieldConfig,
    training_config: TrainingConfig,
) -> None:
    """Save a trained network alongside the configuration that produced it.

    Args:
    - `network`: Trained network to persist.
    - `checkpoint_path`: Destination path for the checkpoint file.
    - `domain_config`: Domain configuration used for training.
    - `fluid_config`: Fluid configuration used for training.
    - `field_config`: Magnetic field configuration used for training.
    - `training_config`: Training configuration used for training.

    Raises:
    - `OSError`: If `checkpoint_path`'s parent directory does not exist and
      cannot be created.
    """
    raise NotImplementedError("Implementation scheduled for Step 2.")


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[nn.Module, DomainConfig, FluidConfig, FieldConfig, TrainingConfig]:
    """Load a trained network and its associated configuration.

    Args:
    - `checkpoint_path`: Path to a checkpoint produced by
      [`save_checkpoint`][magnetofluidics_pinn.io_utils.save_checkpoint].

    Returns:
    - A tuple `(network, domain_config, fluid_config, field_config,
      training_config)`.

    Raises:
    - `FileNotFoundError`: If `checkpoint_path` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path!s}.")
    raise NotImplementedError("Implementation scheduled for Step 2.")
