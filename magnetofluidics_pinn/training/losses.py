"""Functional composition of the training loss.

The total loss is expressed as a weighted reduction over independent loss
terms (PDE residual, boundary conditions, optional data term), each supplied
as a callable, so that new terms can be added without modifying this module.
"""

from __future__ import annotations

from functools import reduce
from typing import Callable

import torch

LossTerm = Callable[[], torch.Tensor]


class LossHistoryBase:
    """Base class for loss history objects providing common functionality."""

    def final(self) -> dict[str, float]:
        """Return the last recorded value of every loss.

        Returns:
        - Mapping from loss name to value.

        Raises:
        - `ValueError`: If nothing was recorded.
        - `NotImplementedError`: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement final()")

    def print_summary(self) -> None:
        """Print a formatted summary of the training history.

        Raises:
        - `NotImplementedError`: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement print_summary()")


def compose_loss(
    terms: dict[str, LossTerm], weights: dict[str, float] | None = None
) -> torch.Tensor:
    """Compose a weighted sum of independent loss terms.

    Args:
    - `terms`: Mapping from a loss-term name (e.g., `"pde"`, `"boundary"`) to
      a zero-argument callable returning a scalar tensor for that term.
    - `weights`: Optional mapping from a loss-term name to its scalar weight.
      Terms without an explicit entry default to a weight of `1.0`.

    Returns:
    - A scalar tensor equal to the weighted sum of all term values.

    Raises:
    - `ValueError`: If `terms` is empty, or if `weights` contains a key not
      present in `terms`.
    """
    if not terms:
        raise ValueError("terms must contain at least one loss term.")
    if weights is not None and not set(weights).issubset(terms):
        raise ValueError("weights contains a key not present in terms.")

    resolved_weights = {name: (weights or {}).get(name, 1.0) for name in terms}
    weighted_values = (resolved_weights[name] * term() for name, term in terms.items())
    return reduce(torch.add, weighted_values)
