r"""Weight transfer between stream-function flow networks.

Only the raw MLP of a
[`StreamFunctionSphereFlow`][magnetofluidics_pinn.networks.stream_function.StreamFunctionSphereFlow]
carries transferable knowledge: the analytic lifting ($F_P$, $T$, $F_S$,
$F_\text{shear}$) is rebuilt from the geometry on every construction. A warm
start therefore copies the parameters of `raw_network` into a freshly built
wrapper and leaves that wrapper's own geometry buffers untouched.

A warm start changes the *cost* of a run, not the claim it verifies. Keep a
cold-started run as the audit reference.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import torch
from torch import nn

from magnetofluidics_pinn.config import TrainingConfig
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import apply_stream_function_constraint
from magnetofluidics_pinn.data_types import Domain

# Reverse of `networks.mlp._ACTIVATION_LAYERS`, used to read a layout back from a module.
_ACTIVATION_NAMES: dict[type[nn.Module], str] = {nn.Tanh: "tanh", nn.SiLU: "silu", nn.ReLU: "relu"}


class WarmStartError(ValueError):
    """Raised when a weight transfer is impossible or would corrupt the target."""


def infer_mlp_layout(raw_network: nn.Module) -> tuple[tuple[int, ...], str]:
    """Read the hidden widths and the activation name of a `build_mlp(2, 2, ...)` network.

    Args:
    - `raw_network`: The raw MLP wrapped by a stream-function network.

    Returns:
    - The tuple `(hidden_layers, activation)`.

    Raises:
    - `WarmStartError`: If `raw_network` is not an alternating
      Linear/activation `nn.Sequential` mapping 2 inputs to 2 outputs with a
      single supported activation.
    """
    if not isinstance(raw_network, nn.Sequential):
        raise WarmStartError(f"raw network must be an nn.Sequential; got {type(raw_network).__name__}.")
    layers = list(raw_network)
    linears, activations = layers[0::2], layers[1::2]
    if len(layers) < 3 or len(layers) % 2 == 0 or not all(isinstance(m, nn.Linear) for m in linears):
        raise WarmStartError("raw network must alternate Linear and activation layers, ending on a Linear layer.")
    names = {_ACTIVATION_NAMES.get(type(m)) for m in activations}
    if len(names) != 1 or None in names:
        raise WarmStartError("raw network must use one supported activation (tanh, silu or relu).")
    if linears[0].in_features != 2 or linears[-1].out_features != 2:
        raise WarmStartError("raw network must map 2 inputs to 2 outputs.")
    return tuple(m.out_features for m in linears[:-1]), names.pop()


def is_transferable(source: nn.Module, target: nn.Module) -> tuple[bool, str]:
    """Check whether the raw weights of `source` can initialize `target`.

    Args:
    - `source`, `target`: Stream-function networks exposing a `raw_network`.

    Returns:
    - `(True, "")` if the layouts match, `(False, reason)` otherwise. Differences
      in device or dtype are allowed; they are resolved by the transfer.
    """
    try:
        layouts = tuple(infer_mlp_layout(module.raw_network) for module in (source, target))
    except AttributeError:
        return False, "both networks must expose a `raw_network` attribute."
    except WarmStartError as error:
        return False, str(error)
    if layouts[0] != layouts[1]:
        return False, f"layout mismatch: source {layouts[0]!r} vs target {layouts[1]!r}."
    return True, ""


def transfer_raw_weights(source: nn.Module, target: nn.Module) -> nn.Module:
    """Return a copy of `target` whose raw MLP carries the weights of `source`.

    Neither argument is modified. The geometry buffers of `target` (sphere
    position, radius, velocity, amplitude, taper support) are preserved.

    Args:
    - `source`: Trained stream-function network.
    - `target`: Freshly built stream-function network with a matching layout.

    Returns:
    - A new module with `target`'s buffers and `source`'s raw parameters, cast to
      `target`'s device and dtype.

    Raises:
    - `WarmStartError`: If the layouts differ or a transferred value is not finite.
    """
    transferable, reason = is_transferable(source, target)
    if not transferable:
        raise WarmStartError(reason)
    reference = next(target.raw_network.parameters())
    state = {
        key: value.detach().clone().to(device=reference.device, dtype=reference.dtype)
        for key, value in source.raw_network.state_dict().items()
    }
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise WarmStartError("source weights contain NaN or Inf values.")
    result = copy.deepcopy(target)
    result.raw_network.load_state_dict(state)
    return result


def rebuild_for_geometry(
    source: nn.Module,
    domain: Domain,
    particle_axial_position: float,
    particle_radius: float,
    particle_axial_velocity: float,
    flow_amplitude: float,
    taper_margin: float,
    training_config: TrainingConfig,
) -> nn.Module:
    """Build a network for a new geometry, initialized from `source`.

    Args:
    - `source`: Trained stream-function network providing the raw weights.
    - `domain`: Nondimensional channel of the new problem.
    - `particle_axial_position`, `particle_radius`, `particle_axial_velocity`,
      `flow_amplitude`, `taper_margin`: Parameters of the new problem; see
      `apply_stream_function_constraint`.
    - `training_config`: Supplies the device and dtype of the new network.

    Returns:
    - A wrapped network with the new geometry and the transferred weights.

    Raises:
    - `WarmStartError`: If `source` has an unsupported layout.
    """
    hidden_layers, activation = infer_mlp_layout(source.raw_network)
    raw = build_mlp(2, 2, hidden_layers, training_config=training_config, activation=activation)
    fresh = apply_stream_function_constraint(
        raw, domain, particle_axial_position, particle_radius,
        particle_axial_velocity, flow_amplitude, taper_margin,
    )
    return transfer_raw_weights(source, fresh)


def scaled_training_config(
    config: TrainingConfig, learning_rate_scale: float = 0.1, epoch_fraction: float = 0.25,
) -> TrainingConfig:
    """Derive a reduced budget for a warm-started run.

    A smaller Adam learning rate avoids destroying the transferred solution in the
    first epochs, and fewer epochs reflect the better starting point.

    Args:
    - `config`: The cold-start configuration; never modified.
    - `learning_rate_scale`: Positive factor applied to `learning_rate`.
    - `epoch_fraction`: Fraction in $(0, 1]$ of `n_epochs` to keep (at least 1 epoch).

    Returns:
    - A new `TrainingConfig`.

    Raises:
    - `ValueError`: If a factor is out of range.
    """
    if not (math.isfinite(learning_rate_scale) and learning_rate_scale > 0.0):
        raise ValueError(f"learning_rate_scale must be finite and positive; got {learning_rate_scale!r}.")
    if not (math.isfinite(epoch_fraction) and 0.0 < epoch_fraction <= 1.0):
        raise ValueError(f"epoch_fraction must lie in (0, 1]; got {epoch_fraction!r}.")
    return dataclasses.replace(
        config,
        learning_rate=config.learning_rate * learning_rate_scale,
        n_epochs=max(1, round(config.n_epochs * epoch_fraction)),
    )
