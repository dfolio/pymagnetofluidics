"""Centralized `torch.device` resolution for `magnetofluidics_pinn`.

Every other module in this package that needs a device follows one rule:
**resolve the device in exactly one place, at the orchestration boundary,
and trust it everywhere downstream.**

Concretely:

- Pure physics/math functions (`physics.*`, `boundary_conditions.*`) never
  call `resolve_device`. They operate on whatever device their input
  tensors already live on (via `torch.zeros_like`, `.device`-aware
  construction, etc.), so a single collocation batch, boundary batch, or
  particle state is guaranteed to be self-consistent regardless of which
  device the caller chose.
- Orchestration functions that *create* new tensors or move existing
  modules across devices (`networks.build_mlp`, `training.train`,
  `sampling.sample_collocation_points`, `io_utils.load_checkpoint`,
  `trajectory.integrate_trajectory`) call
  [`resolve_device`][magnetofluidics_pinn.device_utils.resolve_device] or
  [`resolve_module_device`][magnetofluidics_pinn.device_utils.resolve_module_device]
  exactly once, at the top of the function, and place every tensor they
  subsequently create on that single resolved device.

This avoids failure modes that otherwise recur throughout a project like
this one: (1) duplicated, slightly-inconsistent
`device or ("cuda" if torch.cuda.is_available() else "cpu")` boilerplate
scattered across modules, (2) silent cross-device tensor mismatches (e.g.,
a trained network living on `cuda:0` being called with CPU-resident
coordinates), and (3) silent dtype mismatches (e.g., a network moved to
`torch.float64` being called with the default-`float32` tensors a fresh
`torch.linspace(...)` or `ParticleState` produces) - all of which `torch`
reports only at the point of failure, often far from the actual root cause.
"""

from __future__ import annotations

import torch
from torch import nn


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a requested device to a concrete, available `torch.device`.

    Args:
    - `device`: `None` to auto-select (`"cuda"` if available, else
      `"cpu"`), a device string accepted by `torch.device` (e.g. `"cpu"`,
      `"cuda"`, `"cuda:1"`), or an already-constructed `torch.device`.

    Returns:
    - A `torch.device` guaranteed to be constructible and, if its type is
      `"cuda"`, backed by an available CUDA installation.

    Raises:
    - `TypeError`: If `device` is neither `None`, a `str`, nor a
      `torch.device`.
    - `ValueError`: If `device` is a `str` that `torch.device` cannot parse
      (e.g. a typo such as `"gpu"`).
    - `RuntimeError`: If the resolved device type is `"cuda"` but no CUDA
      device is available on this machine.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, str):
        try:
            resolved = torch.device(device)
        except RuntimeError as error:
            raise ValueError(
                "device must be a valid torch device string (e.g. 'cpu', "
                f"'cuda', 'cuda:0'); got {device!r}."
            ) from error
    else:
        raise TypeError(
            f"device must be None, a str, or a torch.device; got {type(device).__name__}."
        )

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requests a CUDA device, but no CUDA device is available."
        )
    return resolved


def resolve_module_device(module: nn.Module) -> torch.device:
    """Infer the device a module's parameters currently live on.

    Used whenever a function receives an already-constructed module (e.g. a
    trained flow network) and needs to place *new* tensors — grid points,
    particle positions — on the same device as that module, rather than
    re-deriving a device independently and risking a mismatch.

    Args:
    - `module`: Any `torch.nn.Module` instance.

    Returns:
    - The `torch.device` of the module's first parameter.

    Raises:
    - `ValueError`: If `module` has no parameters, so its device cannot be
      inferred.
    """
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise ValueError(
            "module has no parameters; its device cannot be inferred."
        ) from error

def resolve_module_dtype(module: nn.Module) -> torch.dtype:
    """Infer the floating-point dtype a module's parameters currently use.

    Used alongside
    [`resolve_module_device`][magnetofluidics_pinn.device_utils.resolve_module_device]
    whenever a function needs to feed a new tensor into an already-
    constructed module: matching only the device is not enough if that
    module was ever moved to a non-default dtype (e.g. via `.double()`),
    since `torch` raises just as readily on a dtype mismatch as on a
    device mismatch.

    Args:
    - `module`: Any `torch.nn.Module` instance.

    Returns:
    - The `torch.dtype` of the module's first parameter.

    Raises:
    - `ValueError`: If `module` has no parameters, so its dtype cannot be
      inferred.
    """
    try:
        return next(module.parameters()).dtype
    except StopIteration as error:
        raise ValueError(
            "module has no parameters; its dtype cannot be inferred."
        ) from error
