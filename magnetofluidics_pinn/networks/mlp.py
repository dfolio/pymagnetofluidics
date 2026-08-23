"""Multilayer perceptron builder.

Builds the feed-forward network approximating the velocity and pressure
fields. Construction is a pure function of its hyperparameters: no global
state, no implicit device selection beyond what is explicitly requested.
"""

from __future__ import annotations

import torch
from torch import nn

# Maps each supported activation name to its `torch.nn` layer factory,
# keeping `build_mlp` a simple lookup instead of a branching if/elif chain.
_ACTIVATION_LAYERS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


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
    - `ValueError`: If `hidden_layers` is empty, contains a non-positive
      width, if `activation` is unsupported, or if `device` is neither
      `"cuda"`, `"cpu"`, nor `None`.
    - `RuntimeError`: If `device="cuda"` is requested but no CUDA device is
      available on this machine.
    """
    if len(hidden_layers) == 0:
        raise ValueError("hidden_layers must contain at least one layer size.")
    if any(width <= 0 for width in hidden_layers):
        raise ValueError(f"Every entry of hidden_layers must be strictly positive; got {hidden_layers!r}.")
    if activation not in _ACTIVATION_LAYERS:
        raise ValueError(f"Unsupported activation: {activation!r}.")
    if device is not None and device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be 'cuda' or 'cpu'; got {device!r}.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but no CUDA device is available.")
    
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    activation_cls = _ACTIVATION_LAYERS[activation]
    
    layers: list[nn.Module] = []
    previous_width = n_inputs
    for width in hidden_layers:
        layers.append(nn.Linear(previous_width, width))
        layers.append(activation_cls())
        previous_width = width
    layers.append(nn.Linear(previous_width, n_outputs))
    network = nn.Sequential(*layers)
    
    # Xavier/Glorot initialization is the standard choice for PINNs with
    # saturating activations such as tanh (Raissi et al., 2019); it also
    # gives ReLU/SiLU networks a reasonable starting point.
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    return network.to(resolved_device)
