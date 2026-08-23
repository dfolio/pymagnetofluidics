"""Straight-channel vessel geometry.

Builds the simplest supported domain: an axisymmetric, straight channel of
constant radius.
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
    - `ValueError`: If `config.kind` is not `"channel"`, or if `config.length`
      or `config.radius` is not strictly positive (a degenerate geometry has
      no valid interior to solve on).
    """
    if config.kind != "channel":
        raise ValueError(
            f"Expected a channel domain configuration, got kind={config.kind!r}."
        )
    if config.length <= 0.0:
        raise ValueError(f"config.length must be strictly positive; got {config.length!r}.")
    if config.radius <= 0.0:
        raise ValueError(f"config.radius must be strictly positive; got {config.radius!r}.")

    # A straight channel is fully described by its axial extent and radius;
    # it carries no branch angle, unlike a bifurcated vessel.
    return Domain(
        kind="channel",
        length=config.length,
        radius=config.radius,
        branch_angle=None,
    )
