"""Checkpointing and configuration (de)serialization.

Provides pure, side-effect-isolated helpers for saving and loading trained
networks and their associated configuration, so that a run can be resumed or
audited later. File I/O is confined to this module.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch import nn

from magnetofluidics_pinn.config import (
    DomainConfig,
    FieldConfig,
    FluidConfig,
    ParticleConfig,
    TrainingConfig,
)
from magnetofluidics_pinn.device_utils import resolve_device


_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"network", "domain_config", "fluid_config", "field_config", "training_config", "particle_config"}
)


def save_checkpoint(
        network: nn.Module,
        checkpoint_path: Path,
        domain_config: DomainConfig,
        fluid_config: FluidConfig,
        field_config: FieldConfig,
        training_config: TrainingConfig,
        particle_config: ParticleConfig,
) -> None:
    """Save a trained network alongside the configuration that produced it.

    Args:
    - `network`: Trained network to persist.
    - `checkpoint_path`: Destination path for the checkpoint file.
    - `domain_config`: Domain configuration used for training.
    - `fluid_config`: Fluid configuration used for training.
    - `field_config`: Magnetic field configuration used for training.
    - `training_config`: Training configuration used for training.
    - `particle_config`: Particle configuration used for training.

    Raises:
    - `OSError`: If `checkpoint_path`'s parent directory does not exist and
      cannot be created.
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Could not create the checkpoint directory {checkpoint_path.parent!s}."
        ) from error
    
    payload = {
        "network"        : network,
        "domain_config"  : domain_config,
        "fluid_config"   : fluid_config,
        "field_config"   : field_config,
        "particle_config": particle_config,
        "training_config": training_config,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, DomainConfig, FluidConfig, FieldConfig, ParticleConfig, TrainingConfig]:
    """Load a trained network and its associated configuration.

    Args:
    - `checkpoint_path`: Path to a checkpoint produced by
      [`save_checkpoint`][magnetofluidics_pinn.io_utils.save_checkpoint].
    - `device`: Device to load the network onto, e.g. `"cpu"`, `"cuda"`,
      `"cuda:0"`; resolved automatically when `None`. # NEW: previously
      `torch.load` used its default behavior of deserializing tensors onto
      the device they were saved from, which raises if that checkpoint was
      saved from a CUDA-resident network and is being loaded on a machine
      without a CUDA device (or without that specific GPU index). Every
      tensor is now explicitly placed on `device` via `map_location`,
      independent of whichever device the checkpoint happened to be saved
      from.

    Returns:
    - A tuple `(network, domain_config, fluid_config, field_config, particle_config, training_config)`, with `network` on the resolved device.
      training_config)`, with `network` on the resolved device.

    Raises:
    - `FileNotFoundError`: If `checkpoint_path` does not exist.
    - `RuntimeError`: If the file exists but cannot be unpickled, or is
      missing one of the expected checkpoint entries.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path!s}.")

    resolved_device = resolve_device(device)
    try:
        # `weights_only=False` is required because the checkpoint stores
        # full module and configuration objects, not just tensors. Only
        # load checkpoints produced by `save_checkpoint` from a trusted
        # source: unpickling arbitrary data is not safe in general.
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except (pickle.UnpicklingError, RuntimeError, EOFError, OSError) as error:
        raise RuntimeError(f"Failed to load checkpoint at {checkpoint_path!s}: {error}") from error

    missing_keys = _REQUIRED_CHECKPOINT_KEYS - payload.keys()
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path!s} is missing keys: {sorted(missing_keys)}."
        )

    # Belt-and-suspenders: `map_location` reliably relocates the tensors
    # `torch.load` deserializes, but this checkpoint stores a full `nn.Module`
    # object rather than a plain state_dict, so an explicit `.to()` guards
    # against any buffer `map_location` did not catch, across torch versions.
    network = payload["network"].to(resolved_device)

    return (
        network,
        payload["domain_config"],
        payload["fluid_config"],
        payload["field_config"],
        payload["particle_config"],
        payload["training_config"],
    )
