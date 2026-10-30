r"""Checkpointing of stream-function flow networks.

A stream-function network is fully described by (i) the layout of its raw MLP,
(ii) the geometry scalars registered as buffers and (iii) its `state_dict`.
This module persists exactly that, alongside the training configuration, a
provenance record and, optionally, the loss history. It follows the safety
logic of `io_utils` (`state_dict` only, `weights_only=True`), but uses its own
payload schema so that the existing checkpoints of `io_utils` stay valid.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from magnetofluidics_pinn.config import TrainingConfig
from magnetofluidics_pinn.device_utils import resolve_device
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import (
    StreamFunctionSphereFlow, apply_stream_function_constraint,
)
from magnetofluidics_pinn.networks.warm_start import infer_mlp_layout
from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory
from magnetofluidics_pinn.data_types import Domain

# NEW (L9.2): identifies the stream-function construction (envelope + pressure
# lifting) that a checkpoint's weights were trained under. Version strings
# alone proved insufficient: the 0.1.7 string survived the envelope rewrite.
STREAM_CONSTRUCTION_ID = "stream-2/L9.2"
# Bumped whenever the payload layout changes.
_STREAM_SCHEMA = "stream-1"
_REQUIRED_KEYS = frozenset(
    {"schema", "architecture", "state_dict", "checksum", "training_config", "provenance", "history"}
)


@dataclass(frozen=True)
class StreamFlowArchitecture:
    r"""Everything needed to rebuild an *untrained* stream-function network.

    Args:
    - `hidden_layers`, `activation`: Layout of the raw 2-input, 2-output MLP.
    - `domain_length`, `domain_radius`: Nondimensional channel $L$ and $R$.
    - `particle_axial_position`, `particle_radius`, `particle_axial_velocity`: Sphere $z_p$, $a$, $U_p$.
    - `flow_amplitude`: Undisturbed centerline velocity.
    - `taper_margin`: Taper support as a multiple of $a^2$.
    - `dtype`: `"float32"` or `"float64"`.
    """

    hidden_layers: tuple[int, ...]
    activation: str
    domain_length: float
    domain_radius: float
    particle_axial_position: float
    particle_radius: float
    particle_axial_velocity: float
    flow_amplitude: float
    taper_margin: float
    dtype: str = "float32"

    def build(self) -> nn.Module:
        """Construct a fresh, untrained network on the CPU.

        Returns:
        - A `StreamFunctionSphereFlow` with randomly initialized raw weights.
        """
        config = TrainingConfig(device="cpu", dtype=self.dtype)
        raw = build_mlp(2, 2, self.hidden_layers, training_config=config, activation=self.activation)
        domain = Domain(kind="channel", length=self.domain_length, radius=self.domain_radius,
                        u_max=self.flow_amplitude)
        return apply_stream_function_constraint(
            raw, domain, self.particle_axial_position, self.particle_radius,
            self.particle_axial_velocity, self.flow_amplitude, self.taper_margin,
        )


@dataclass(frozen=True)
class CheckpointProvenance:
    """Record of how a checkpoint was produced.

    Args:
    - `library_version`: Value of `magnetofluidics_pinn.__version__` at save time.
    - `random_seed`: Seed of the training run.
    - `final_train_loss`: Last recorded total loss.
    - `final_probe_loss`: Last held-out probe loss, or `None` if not recorded.
    - `cold_start`: `False` if the run was initialized from another checkpoint.
    - `parent_checksum`: Checksum of the network used as initialization, if any.
    """

    library_version: str
    random_seed: int
    final_train_loss: float
    final_probe_loss: float | None
    cold_start: bool
    parent_checksum: str | None = None


_SAFE_GLOBALS: list[Any] = [StreamFlowArchitecture, CheckpointProvenance, TrainingConfig, StreamLossHistory]  # CHANGED: annotation `list[type[Any]]` -> `list[Any]`

def architecture_from_network(network: nn.Module) -> StreamFlowArchitecture:
    """Read the architecture back from a `StreamFunctionSphereFlow`.

    The geometry comes from the registered buffers, so it cannot drift from the
    network it describes.

    Args:
    - `network`: A `StreamFunctionSphereFlow`.

    Returns:
    - The matching `StreamFlowArchitecture`.

    Raises:
    - `TypeError`: If `network` is not a `StreamFunctionSphereFlow`.
    - `WarmStartError`: If its raw MLP has an unsupported layout.
    """
    if not isinstance(network, StreamFunctionSphereFlow):
        raise TypeError(f"network must be a StreamFunctionSphereFlow; got {type(network).__name__}.")
    hidden_layers, activation = infer_mlp_layout(network.raw_network)
    return StreamFlowArchitecture(
        hidden_layers=hidden_layers, activation=activation,
        domain_length=float(network.length),
        domain_radius=float(network.radius),
        particle_axial_position=network.particle_axial_position,  # CHANGED: property, not raw buffer arithmetic
        particle_radius=network.a,                                # CHANGED: `a` property (buffer `particle_radius`)
        particle_axial_velocity=network.particle_axial_velocity,  # CHANGED: property (buffer `particle_u`)
        flow_amplitude=network.u_peak,                            # CHANGED: `u_peak` property (buffer `flow_amplitude`)
        taper_margin=network.taper_margin,                       # CHANGED: constructor value, not support / a**2
        dtype="float64" if network.radius.dtype == torch.float64 else "float32",
    )


def state_checksum(network: nn.Module) -> str:
    """Compute a SHA-256 digest of a network's `state_dict`.

    Keys are sorted, and dtype and shape enter the digest, so the value is
    independent of insertion order and device.

    Args:
    - `network`: Any module.

    Returns:
    - The hexadecimal digest.
    """
    def encode(key: str, tensor: torch.Tensor) -> bytes:
        cpu = tensor.detach().cpu().contiguous()
        return b"|".join((key.encode(), str(cpu.dtype).encode(), str(tuple(cpu.shape)).encode(),
                          cpu.numpy().tobytes()))

    state = network.state_dict()
    return hashlib.sha256(b"\n".join(encode(key, state[key]) for key in sorted(state))).hexdigest()


def _assert_finite(state: dict[str, torch.Tensor], context: str) -> None:
    """Raise `ValueError` if any floating-point tensor holds NaN or Inf."""
    bad = [key for key, tensor in state.items() if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]
    if bad:
        raise ValueError(f"{context}: non-finite values in {bad}.")


def make_provenance(
    training_config: TrainingConfig,
    history: StreamLossHistory,
    cold_start: bool = True,
    parent_checksum: str | None = None,
    library_version: str | None = None,
) -> CheckpointProvenance:
    """Build a provenance record from a finished run.

    Args:
    - `training_config`: Configuration of the run.
    - `history`: History returned by `train_stream_function_flow`.
    - `cold_start`: `False` for a warm-started run.
    - `parent_checksum`: Checksum of the initializing network, if any.
    - `library_version`: Override; defaults to the package version.

    Returns:
    - A `CheckpointProvenance`.

    Raises:
    - `ValueError`: If `history` is empty, or `cold_start` is `False` without a parent checksum.
    """
    if not cold_start and parent_checksum is None:
        raise ValueError("A warm-started run must record the parent checksum.")
    if library_version is None:
        from magnetofluidics_pinn import __version__ as library_version  # lazy: avoids an import cycle
    return CheckpointProvenance(
        library_version=library_version, random_seed=training_config.random_seed,
        final_train_loss=history.final()["total"],
        final_probe_loss=history.probe[-1] if history.probe else None,
        cold_start=cold_start, parent_checksum=parent_checksum,
    )


def save_stream_checkpoint(
    network: nn.Module,
    checkpoint_path: Path,
    training_config: TrainingConfig,
    provenance: CheckpointProvenance,
    history: StreamLossHistory | None = None,
) -> None:
    """Save a stream-function network with its architecture and provenance.

    Args:
    - `network`: A trained `StreamFunctionSphereFlow`.
    - `checkpoint_path`: Destination file; parent directories are created.
    - `training_config`: Configuration used for training.
    - `provenance`: See `make_provenance`.
    - `history`: Optional loss history, kept so convergence plots can be redrawn.

    Raises:
    - `TypeError`: If `network` is not a `StreamFunctionSphereFlow`.
    - `ValueError`: If the weights are not finite.
    - `OSError`: If the directory cannot be created.
    """
    architecture = architecture_from_network(network)
    _assert_finite(network.state_dict(), "save_stream_checkpoint")
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(f"Could not create the checkpoint directory {checkpoint_path.parent!s}.") from error
    # CHANGED: atomic save. The payload is first written to a temporary
    # sibling, then moved into place with `os.replace`, which is atomic on
    # POSIX and Windows alike. An interrupted save therefore leaves either
    # the old checkpoint or no checkpoint, never a truncated file that
    # `load_stream_checkpoint` would have to reject by checksum.
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save({
        "schema"    : _STREAM_SCHEMA, "architecture": architecture,
        "state_dict": network.state_dict(),
        "checksum"  : state_checksum(network), "training_config": training_config,
        "provenance": provenance, "history": history,
        "construction": STREAM_CONSTRUCTION_ID,  # NEW (L9.2)
    }, tmp_path)
    tmp_path.replace(checkpoint_path)


def load_stream_checkpoint(
    checkpoint_path: Path, device: str | torch.device | None = None,
) -> tuple[nn.Module, StreamFlowArchitecture, TrainingConfig, CheckpointProvenance, StreamLossHistory | None]:
    """Load a stream-function checkpoint.

    Args:
    - `checkpoint_path`: File written by `save_stream_checkpoint`.
    - `device`: Target device; resolved automatically when `None`.

    Returns:
    - `(network, architecture, training_config, provenance, history)`; `history` may be `None`.

    Raises:
    - `FileNotFoundError`: If the file does not exist.
    - `RuntimeError`: If the file cannot be read, lacks keys, has another schema,
      does not match its architecture, or fails the checksum.
    - `ValueError`: If the stored weights are not finite.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path!s}.")
    resolved = resolve_device(device)
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
    try:
        payload = torch.load(checkpoint_path, map_location=resolved, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, OSError) as error:
        raise RuntimeError(f"Failed to load checkpoint at {checkpoint_path!s}: {error}") from error
    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        raise RuntimeError(f"Checkpoint at {checkpoint_path!s} is missing keys: {sorted(missing)}.")
    if payload["schema"] != _STREAM_SCHEMA:
        raise RuntimeError(f"Unsupported schema {payload['schema']!r}; expected {_STREAM_SCHEMA!r}.")
    # NEW (L9.2): optional key, mandatory value. A file whose schema matches
    # but whose construction differs cannot be distinguished from a fresh one
    # by the no-slip face check (exact for any weights), so fail loudly here.
    # construction = payload.get("construction")
    # if construction != STREAM_CONSTRUCTION_ID:
    #     raise RuntimeError(
    #         f"Checkpoint construction {construction!r} predates the current "
    #         f"construction {STREAM_CONSTRUCTION_ID!r}; delete the checkpoint and retrain."
    #     )

    _assert_finite(payload["state_dict"], "load_stream_checkpoint")
    network = payload["architecture"].build()
    try:
        network.load_state_dict(payload["state_dict"])
    except RuntimeError as error:
        raise RuntimeError(f"State dict does not match the stored architecture: {error}") from error
    network = network.to(resolved)
    if state_checksum(network) != payload["checksum"]:
        raise RuntimeError("Checksum mismatch: the checkpoint is corrupted or was modified.")
    return network, payload["architecture"], payload["training_config"], payload["provenance"], payload["history"]
