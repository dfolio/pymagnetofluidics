"""Straight-channel vessel geometry.

Builds the simplest supported domain: an axisymmetric, straight channel of
constant radius, used for the first phase of the roadmap (single particle,
uniform magnetic field, Stokes flow).
"""

from __future__ import annotations

from magnetofluidics_pinn.config import DomainConfig
from magnetofluidics_pinn.types import Domain


def build_channel_domain(config: DomainConfig) -> Domain:
    """Build a straight-channel domain from a domain configuration.

    The returned `Domain` holds physical (SI, meter) values, matching
    `config`. Call
    [`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]
    before passing it to `sampling` or `physics`. # CHANGED: documented the
    physical-to-dimensionless handoff.

    Args:
    - `config`: Domain configuration with `kind == "channel"`.

    Returns:
    - A [`Domain`][magnetofluidics_pinn.types.Domain] instance describing the
      straight channel geometry, in physical (SI) units.

    Raises:
    - `ValueError`: If `config.kind` is not `"channel"`.
    """
    if config.kind != "channel":
        raise ValueError(
            f"Expected a channel domain configuration, got kind={config.kind!r}."
        )
    raise NotImplementedError("Implementation scheduled for Step 2.")
