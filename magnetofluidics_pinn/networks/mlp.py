"""Multilayer perceptron builder.

Builds the feed-forward network approximating the velocity and pressure
fields. Construction is a pure function of its hyperparameters: no global
state, no implicit device selection beyond what is explicitly requested.
"""

from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    n_inputs: int,
    n_outputs: int,
    hidden_layers: tuple[int, ...],
    activation: str = "tanh",
    device: str | None = None,
) -> nn.Module:
    """Build a fully connected feed-forward network.

    Args:
    - `n_inputs`: Number of input coordinates (e.g., 2 for `(r, z)`, 3 with
      time for unsteady problems).
    - `n_outputs`: Number of output quantities (e.g., 3 for `(u_r, u_z, p)`).
    - `hidden_layers`: Number of neurons in each hidden layer, e.g.,
      `(64, 64, 64)`.
    - `activation`: Activation function name; one of `"tanh"`, `"silu"`,
      `"relu"`.
    - `device`: Target device, `"cuda"` or `"cpu"`. Resolved automatically to
      `"cuda"` when available, `"cpu"` otherwise, if left as `None`.

    Returns:
    - A `torch.nn.Module` instance placed on the resolved device.

    Raises:
    - `ValueError`: If `hidden_layers` is empty or `activation` is
      unsupported.
    """
    if len(hidden_layers) == 0:
        raise ValueError("hidden_layers must contain at least one layer size.")
    if activation not in {"tanh", "silu", "relu"}:
        raise ValueError(f"Unsupported activation: {activation!r}.")
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raise NotImplementedError(
        f"Implementation scheduled for Step 2 (target device: {resolved_device})."
    )
