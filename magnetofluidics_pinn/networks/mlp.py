"""Multilayer perceptron builder.

Builds the feed-forward network approximating the velocity and pressure
fields. Construction is a pure function of its hyperparameters: no global
state, no implicit device selection beyond what is explicitly requested.
"""

from __future__ import annotations

from torch import nn

from magnetofluidics_pinn.device_utils import resolve_device

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
    - `device`: Target device, e.g. `"cuda"`, `"cuda:0"`, or `"cpu"`.
      Resolved automatically to `"cuda"` when available, `"cpu"` otherwise,
      if left as `None`. # CHANGED: any device string accepted by
      `torch.device` now works (previously only exactly "cuda"/"cpu"),
      resolved through the shared `device_utils.resolve_device`.

    Returns:
    - A `torch.nn.Module` instance placed on the resolved device.

    Raises:
    - `ValueError`: If `hidden_layers` is empty, contains a non-positive
      width, if `activation` is unsupported, or if `device` is not a valid
      device string.
    - `RuntimeError`: If `device` (or the auto-selected default) resolves
      to a CUDA device but no CUDA device is available.
    """
    if len(hidden_layers) == 0:
        raise ValueError("hidden_layers must contain at least one layer size.")
    if any(width <= 0 for width in hidden_layers):
        raise ValueError(f"Every entry of hidden_layers must be strictly positive; got {hidden_layers!r}.")
    if activation not in _ACTIVATION_LAYERS:
        raise ValueError(f"Unsupported activation: {activation!r}.")
    
    resolved_device = resolve_device(device)
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
