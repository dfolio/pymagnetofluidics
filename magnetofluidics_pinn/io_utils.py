"""Checkpointing and configuration (de)serialization.

Provides pure, side-effect-isolated helpers for saving and loading trained
networks and their associated configuration, so that a run can be resumed or
audited later. File I/O is confined to this module.

**Why `state_dict`, not the whole module.** Pickling an `nn.Module` object
directly - as an earlier version of this module did - embeds a reference to
the exact Python class (and its exact import path) used at save time.
Load that checkpoint after refactoring `networks.mlp` or upgrading `torch`,
and unpickling can fail in ways that have nothing to do with the model
itself. It is also unsafe for a checkpoint from an untrusted source:
unpickling an arbitrary object graph can execute arbitrary code. This
module instead persists `network.state_dict()` (parameter tensors only)
alongside a small [`NetworkArchitecture`][magnetofluidics_pinn.io_utils.NetworkArchitecture]
descriptor that says how to rebuild the *un-trained* module before loading
those parameters back in, and loads with `weights_only=True` - PyTorch's
restricted unpickler, extended here to allow only this package's own
configuration dataclasses in addition to its own default safe list of
tensors and Python primitives.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from sympy.physics.biomechanics import activation
from torch import nn

from magnetofluidics_pinn.config import (
    DomainConfig,
    MagneticFieldConfig,
    FluidConfig,
    ParticleConfig,
    TrainingConfig,
)
from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.networks.constraints import apply_hard_wall_constraint
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.types import Domain

# Bumped whenever the *shape* of a checkpoint's payload changes (new/renamed
# keys, a different meaning for an existing key) - not for ordinary model
# retraining. `load_checkpoint` refuses to read a mismatched version rather
# than guessing at a possibly-incompatible layout.
_CHECKPOINT_SCHEMA_VERSION = 2

_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "architecture",
        "state_dict",
        "domain_config",
        "fluid_config",
        "field_config",
        "particle_config",
        "training_config",
    }
)

# Every config type `load_checkpoint` may need to unpickle beyond torch's
# own default-safe primitives (tensors, numbers, strings, containers).
# `NetworkArchitecture` itself is added to this list right after its own
# definition below, since it can't be referenced before it exists.
_SAFE_CHECKPOINT_GLOBALS: list[type] = [
    DomainConfig,
    FluidConfig,
    MagneticFieldConfig,
    ParticleConfig,
    TrainingConfig,
]


@dataclass(frozen=True)
class NetworkArchitecture:
    """Describes how to rebuild a flow network's architecture from scratch.

    Captures exactly the information
    [`save_checkpoint`][magnetofluidics_pinn.io_utils.save_checkpoint] and
    [`load_checkpoint`][magnetofluidics_pinn.io_utils.load_checkpoint] need
    to reconstruct an *untrained* network with the right shape before
    loading a saved `state_dict` into it - deliberately independent of any
    particular trained parameter values.

    Args:
    - `n_inputs`, `n_outputs`, `hidden_layers`, `activation`: Passed
      directly to [`build_mlp`][magnetofluidics_pinn.networks.mlp.build_mlp].
    - `hard_wall_constraint_radius`: If not `None`, the network was wrapped
      with
      [`apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint]
      using a domain of this radius before training; `None` means the raw
      `build_mlp` output was used unwrapped. Only the radius is needed to
      reconstruct the wrapper, since it depends on nothing else about the
      domain.
    """

    n_inputs: int
    n_outputs: int
    hidden_layers: tuple[int, ...]
    activation: str = "tanh"
    hard_wall_constraint_radius: float | None = None

    def build(self) -> nn.Module:
        """Construct a fresh, untrained network matching this architecture.

        Returns:
        - A `torch.nn.Module` with randomly-initialized parameters, on the
          CPU; callers move it to the desired device (as
          [`load_checkpoint`][magnetofluidics_pinn.io_utils.load_checkpoint]
          does) after loading a `state_dict` into it.
        """
        network = build_mlp(self.n_inputs, self.n_outputs, self.hidden_layers,
                            training_config=TrainingConfig(device="cpu"),
                            activation=self.activation)
        if self.hard_wall_constraint_radius is not None:
            # `apply_hard_wall_constraint` only reads `domain.radius`; the
            # other `Domain` fields are placeholders with no effect on the
            # reconstructed architecture.
            placeholder_domain = Domain(kind="channel", length=1.0, radius=self.hard_wall_constraint_radius)
            network = apply_hard_wall_constraint(network, placeholder_domain)
        return network


_SAFE_CHECKPOINT_GLOBALS.append(NetworkArchitecture)


def save_checkpoint(
    network: nn.Module,
    checkpoint_path: Path,
    architecture: NetworkArchitecture,
    domain_config: DomainConfig,
    fluid_config: FluidConfig,
    field_config: MagneticFieldConfig,
    particle_config: ParticleConfig,
    training_config: TrainingConfig,
) -> None:
    """Save a trained network's parameters alongside the configuration that produced it.

    Args:
    - `network`: Trained network to persist. Its `state_dict()` - parameter
      tensors only - is what is actually written, not `network` itself.
    - `checkpoint_path`: Destination path for the checkpoint file.
    - `architecture`: Describes how to rebuild `network`'s architecture; see
      [`NetworkArchitecture`][magnetofluidics_pinn.io_utils.NetworkArchitecture].
      Must match `network`'s actual architecture, or the `state_dict` saved
      here will not load back into the network
      [`load_checkpoint`][magnetofluidics_pinn.io_utils.load_checkpoint]
      reconstructs from it.
    - `domain_config`: Domain configuration used for training.
    - `fluid_config`: Fluid configuration used for training.
    - `field_config`: Magnetic field configuration used for training.
    - `particle_config`: Particle configuration used for training.
    - `training_config`: Training configuration used for training.

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
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "architecture": architecture,
        "state_dict": network.state_dict(),
        "domain_config": domain_config,
        "fluid_config": fluid_config,
        "field_config": field_config,
        "particle_config": particle_config,
        "training_config": training_config,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, DomainConfig, FluidConfig, MagneticFieldConfig, ParticleConfig, TrainingConfig]:
    """Load a trained network and its associated configuration.

    Args:
    - `checkpoint_path`: Path to a checkpoint produced by
      [`save_checkpoint`][magnetofluidics_pinn.io_utils.save_checkpoint].
    - `device`: Device to load the network onto, e.g. `"cpu"`, `"cuda"`,
      `"cuda:0"`; resolved automatically when `None`. Every tensor is
      placed on `device` via `map_location`, independent of whichever
      device the checkpoint happened to be saved from.

    Returns:
    - A tuple `(network, domain_config, fluid_config, field_config, particle_config, training_config)`, with `network` rebuilt from the checkpoint's
      training_config)`, with `network` rebuilt from the checkpoint's
      [`NetworkArchitecture`][magnetofluidics_pinn.io_utils.NetworkArchitecture]
      and its saved `state_dict` loaded in, on the resolved device.

    Raises:
    - `FileNotFoundError`: If `checkpoint_path` does not exist.
    - `RuntimeError`: If the file exists but cannot be unpickled, is missing
      one of the expected checkpoint entries, or was written with an
      incompatible checkpoint schema version.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path!s}.")

    resolved_device = resolve_device(device)
    torch.serialization.add_safe_globals(_SAFE_CHECKPOINT_GLOBALS)
    try:
        # `weights_only=True` restricts unpickling to tensors, Python
        # primitives, and the explicitly-registered types above: loading a
        # tampered-with or otherwise untrusted checkpoint cannot execute
        # arbitrary code as a side effect of deserializing it.
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, OSError) as error:
        raise RuntimeError(f"Failed to load checkpoint at {checkpoint_path!s}: {error}") from error

    missing_keys = _REQUIRED_CHECKPOINT_KEYS - payload.keys()
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path!s} is missing keys: {sorted(missing_keys)}."
        )
    if payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path!s} uses schema version "
            f"{payload['schema_version']!r}, but this version of "
            f"magnetofluidics_pinn reads schema {_CHECKPOINT_SCHEMA_VERSION!r}."
        )

    network = payload["architecture"].build().to(resolved_device)
    network.load_state_dict(payload["state_dict"])

    return (
        network,
        payload["domain_config"],
        payload["fluid_config"],
        payload["field_config"],
        payload["particle_config"],
        payload["training_config"],
    )
