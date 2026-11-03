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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from magnetofluidics_pinn.config import TrainingConfig
from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.device_utils import (
    resolve_device,
    resolve_module_dtype,
)
from magnetofluidics_pinn.networks.constraints import (
    _HardParticleConstrainedFlow,
    _HardWallConstrainedFlow,
    apply_hard_particle_constraint,
    apply_hard_wall_constraint)
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import (
    apply_stream_function_constraint, LiftingMode, StreamFunctionSphereFlow)
from magnetofluidics_pinn.networks.warm_start import (
    _ACTIVATION_NAMES, WarmStartError, infer_mlp_layout)
from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory
from magnetofluidics_pinn.training.trainer import LossHistory, TrainingHistory

# from conftest import device

# Identifies the stream-function construction (envelope + pressure
# lifting) that a checkpoint's weights were trained under. Version strings
# alone proved insufficient: the 0.1.7 string survived the envelope rewrite.
STREAM_CONSTRUCTION_ID = "stream-2/L9.2"

# The construction changed (analytic pressure, (s, z) formulation, no geometry
# buffers), so weights trained under "stream-1" are meaningless under the new
# envelope and must be retrained.
_STREAM_SCHEMA = "stream-2"

_REQUIRED_KEYS = frozenset(
    {"schema", "architecture", "state_dict", "checksum", "training_config",
     "provenance", "history"}
)


@dataclass(frozen=True)
class HardWallFlowArchitecture:
    """Everything needed to rebuild an *untrained* hard-wall-constrained flow network."""

    hidden_layers: tuple[int, ...]
    activation: str
    domain_length: float
    domain_radius: float
    flow_amplitude: float
    device: str = "cpu"
    dtype: str = "float32"

    def build(self) -> nn.Module:
        """Construct a fresh, untrained network on the CPU.

        Returns:
        - A `_HardWallConstrainedFlow` with randomly initialized raw weights.
        """
        config = TrainingConfig(device=self.device, dtype=self.dtype)
        raw = build_mlp(2, 3, self.hidden_layers, training_config=config,
                        activation=self.activation)
        domain = Domain(kind="channel", length=self.domain_length,
                        radius=self.domain_radius,
                        u_max=self.flow_amplitude)
        return apply_hard_wall_constraint(raw, domain)


@dataclass(frozen=True)
class HardParticleFlowArchitecture:
    """Everything needed to rebuild an *untrained* hard-particle-constrained flow network."""

    hidden_layers: tuple[int, ...]
    activation: str
    domain_length: float
    domain_radius: float
    particle_axial_position: float
    particle_radius: float
    particle_axial_velocity: float
    flow_amplitude: float
    device: str = "cpu"
    dtype: str = "float32"

    def build(self) -> nn.Module:
        """Construct a fresh, untrained network on the CPU.

        Returns:
        - A `_HardParticleConstrainedFlow` with randomly initialized raw weights.
        """
        config = TrainingConfig(device=self.device, dtype=self.dtype)
        raw = build_mlp(2, 3, self.hidden_layers, training_config=config,
                        activation=self.activation)
        domain = Domain(kind="channel", length=self.domain_length,
                        radius=self.domain_radius,
                        u_max=self.flow_amplitude)
        return apply_hard_particle_constraint(
            raw, domain, self.particle_axial_position, self.particle_radius,
            self.particle_axial_velocity
        )


@dataclass(frozen=True)
class StreamFlowArchitecture:
    r"""Everything needed to rebuild an *untrained* stream-function network.

    Args:
    - `hidden_layers`, `activation`: Layout of the raw 2-input, 2-output MLP.
    - `domain_length`, `domain_radius`: Nondimensional channel $L$ and $R$.
    - `particle_axial_position`, `particle_radius`, `particle_axial_velocity`: Sphere $z_p$, $a$, $U_p$.
    - `flow_amplitude`: Undisturbed centreline velocity.
    - `taper_margin`: Taper support as a multiple of $a^2$.
    - `dtype`: `"float32"` or `"float64"`.
    - `lifting_mode`: NEW. `"taper"` or `"wall"`.
    - `end_envelope`: NEW. Whether the learned correction carries the inlet/outlet envelope.
    - `wall_start_fraction`: NEW. Blend start of the `"wall"` mode.
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
    device: str = "cpu"
    dtype: str = "float32"
    lifting_mode: LiftingMode = "taper"
    end_envelope: bool = False
    wall_start_fraction: float = 0.05

    def build(self) -> nn.Module:
        """Construct a fresh, untrained network on the CPU.

        Returns:
        - A `StreamFunctionSphereFlow` with randomly initialized raw weights.
        """
        config = TrainingConfig(device=self.device, dtype=self.dtype)
        raw = build_mlp(2, 2, self.hidden_layers, training_config=config,
                        activation=self.activation)
        domain = Domain(kind="channel", length=self.domain_length,
                        radius=self.domain_radius,
                        u_max=self.flow_amplitude)
        return apply_stream_function_constraint(
            raw, domain, self.particle_axial_position, self.particle_radius,
            self.particle_axial_velocity, self.flow_amplitude, self.taper_margin,
            lifting_mode=self.lifting_mode,
            end_envelope=self.end_envelope,
            wall_start_fraction=self.wall_start_fraction,
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


_SAFE_GLOBALS: list[Any] = [
    StreamFlowArchitecture, HardWallFlowArchitecture, HardParticleFlowArchitecture,
    CheckpointProvenance, TrainingConfig, StreamLossHistory, TrainingHistory,
    LossHistory
]

# Type alias for any flow architecture
FlowArchitecture = StreamFlowArchitecture | HardWallFlowArchitecture | HardParticleFlowArchitecture


def _infer_mlp_layout_generic(network: nn.Module) -> tuple[tuple[int, ...], str]:
    """Infer the layout of a raw MLP network (generic version).

    This is a more flexible version that doesn't enforce 2-input, 2-output constraint.
    """
    linears = [m for m in network.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise WarmStartError("network must contain at least one Linear layer.")

    act_types = set()
    for m in network.modules():
        if isinstance(m, (nn.Tanh, nn.SiLU, nn.ReLU)):
            act_types.add(type(m))

    act_names = set()
    for act_type in act_types:
        if act_type not in _ACTIVATION_NAMES:
            raise WarmStartError(
                "raw network must use one supported activation (tanh, silu or relu).")
        act_names.add(_ACTIVATION_NAMES[act_type])

    if len(act_names) != 1:
        raise WarmStartError("raw network must use a single supported activation type.")

    hidden_widths: tuple[int, ...] = tuple(layer.out_features for layer in linears[:-1])
    activation_name: str = next(iter(act_names))
    return hidden_widths, activation_name


def architecture_from_network(network: nn.Module) -> FlowArchitecture:
    """Read the architecture back from a flow network.

    The geometry comes from the network's attributes, so it cannot drift from the
    network it describes. Supports StreamFunctionSphereFlow, _HardWallConstrainedFlow,
    and _HardParticleConstrainedFlow.

    Args:
    - `network`: A flow network (StreamFunctionSphereFlow, _HardWallConstrainedFlow, or _HardParticleConstrainedFlow).

    Returns:
    - The matching architecture dataclass (StreamFlowArchitecture, HardWallFlowArchitecture, or HardParticleFlowArchitecture).

    Raises:
    - `TypeError`: If `network` is not a supported flow network type.
    - `WarmStartError`: If its raw MLP has an unsupported layout.
    """
    inferred_dtype = resolve_module_dtype(network.raw_network)
    dtype_str = "float64" if inferred_dtype == torch.float64 else "float32"
    if isinstance(network, StreamFunctionSphereFlow):
        hidden_layers, activation = infer_mlp_layout(network.raw_network)
        return StreamFlowArchitecture(
            hidden_layers=hidden_layers, activation=activation,
            domain_length=float(network.length),
            domain_radius=float(network.radius),
            particle_axial_position=network.particle_axial_position,
            particle_radius=network.a,
            particle_axial_velocity=network.particle_axial_velocity,
            flow_amplitude=network.u_peak,
            taper_margin=network.taper_margin,
            dtype=dtype_str,
        )
    elif isinstance(network, _HardWallConstrainedFlow):
        hidden_layers, activation = _infer_mlp_layout_generic(network.raw_network)
        return HardWallFlowArchitecture(
            hidden_layers=hidden_layers, activation=activation,
            domain_length=float(network.length),
            domain_radius=float(network.radius),
            flow_amplitude=network.u_max,
            dtype=dtype_str,
        )
    elif isinstance(network, _HardParticleConstrainedFlow):
        hidden_layers, activation = _infer_mlp_layout_generic(network.raw_network)
        return HardParticleFlowArchitecture(
            hidden_layers=hidden_layers, activation=activation,
            domain_length=float(network.length),
            domain_radius=float(network.radius),
            particle_axial_position=network.particle_axial_position,
            particle_radius=network.particle_radius,
            particle_axial_velocity=network.particle_axial_velocity,
            flow_amplitude=network.u_max,
            dtype=dtype_str,
        )
    else:
        raise TypeError(
            f"network must be a StreamFunctionSphereFlow, _HardWallConstrainedFlow, "
            f"or _HardParticleConstrainedFlow; got {type(network).__name__}."
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
        return b"|".join(
            (key.encode(), str(cpu.dtype).encode(), str(tuple(cpu.shape)).encode(),
             cpu.numpy().tobytes()))

    state = network.state_dict()
    return hashlib.sha256(
        b"\n".join(encode(key, state[key]) for key in sorted(state))).hexdigest()


def _assert_finite(state: dict[str, torch.Tensor], context: str) -> None:
    """Raise `ValueError` if any floating-point tensor holds NaN or Inf."""
    bad = [key for key, tensor in state.items() if
           tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]
    if bad:
        raise ValueError(f"{context}: non-finite values in {bad}.")


def make_provenance(
    training_config: TrainingConfig,
    history: StreamLossHistory | TrainingHistory,
    cold_start: bool = True,
    parent_checksum: str | None = None,
    library_version: str | None = None,
) -> CheckpointProvenance:
    """Build a provenance record from a finished run.

    Args:
    - `training_config`: Configuration of the run.
    - `history`: History returned by `train_stream_function_flow` (StreamLossHistory) or `train()` (TrainingHistory).
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
        from magnetofluidics_pinn import \
            __version__ as library_version  # lazy: avoids an import cycle
    # Handle both StreamLossHistory and TrainingHistory
    if isinstance(history, StreamLossHistory):
        probe = history.best_probe if math.isfinite(history.best_probe) else (
            history.probe[-1] if history.probe else None)
        final_train_loss = history.final()["total"]
    else:  # TrainingHistory
        probe = history.best_probe if math.isfinite(history.best_probe) else None
        # Get final loss from lbfgs if available, otherwise from adam
        final_history = history.lbfgs if history.lbfgs is not None else history.adam
        final_train_loss = final_history.final()["total"]
    return CheckpointProvenance(
        library_version=library_version,
        random_seed=training_config.random_seed,
        final_train_loss=final_train_loss,
        final_probe_loss=probe,
        cold_start=cold_start,
        parent_checksum=parent_checksum,
    )


def save_stream_checkpoint(
    network: nn.Module,
    checkpoint_path: Path,
    training_config: TrainingConfig,
    provenance: CheckpointProvenance,
    history: StreamLossHistory | TrainingHistory | None = None,
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
        raise OSError(
            f"Could not create the checkpoint directory {checkpoint_path.parent!s}.") from error
    # CHANGED: atomic save. The payload is first written to a temporary
    # sibling, then moved into place with `os.replace`, which is atomic on
    # POSIX and Windows alike. An interrupted save therefore leaves either
    # the old checkpoint or no checkpoint, never a truncated file that
    # `load_stream_checkpoint` would have to reject by checksum.
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save({
        "schema"      : _STREAM_SCHEMA, "architecture": architecture,
        "state_dict"  : network.state_dict(),
        "checksum"    : state_checksum(network), "training_config": training_config,
        "provenance"  : provenance, "history": history,
        "construction": STREAM_CONSTRUCTION_ID,  # NEW (L9.2)
    }, tmp_path)
    tmp_path.replace(checkpoint_path)


def load_stream_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device | None = None,
) -> tuple[
    nn.Module,
    FlowArchitecture,
    TrainingConfig,
    CheckpointProvenance,
    StreamLossHistory | TrainingHistory | None,
]:
    """Load a flow network checkpoint.

    Args:
    - `checkpoint_path`: File written by `save_stream_checkpoint`.
    - `device`: Target device; resolved automatically when `None`.

    Returns:
    - `(network, architecture, training_config, provenance, history)`; `history` may be `None`.
      `architecture` can be StreamFlowArchitecture, HardWallFlowArchitecture, or HardParticleFlowArchitecture.

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
        raise RuntimeError(
            f"Failed to load checkpoint at {checkpoint_path!s}: {error}") from error
    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path!s} is missing keys: {sorted(missing)}.")
    if payload["schema"] != _STREAM_SCHEMA:
        raise RuntimeError(
            f"Unsupported schema {payload['schema']!r}; expected {_STREAM_SCHEMA!r}.")

    _assert_finite(payload["state_dict"], "load_stream_checkpoint")

    # Build directly on the target device if architecture supports it
    arch: FlowArchitecture = payload["architecture"]
    if hasattr(arch, "device"):
        # Functional copy with updated device target
        from dataclasses import replace
        arch = replace(arch, device=str(resolved))

    network = arch.build()
    try:
        network.load_state_dict(payload["state_dict"])
    except RuntimeError as error:
        raise RuntimeError(
            f"State dict does not match the stored architecture: {error}"
        ) from error

    network = network.to(resolved)
    if state_checksum(network) != payload["checksum"]:
        raise RuntimeError(
            "Checksum mismatch: the checkpoint is corrupted or was modified.")
    return network, payload["architecture"], payload["training_config"], payload[
        "provenance"], payload["history"]
