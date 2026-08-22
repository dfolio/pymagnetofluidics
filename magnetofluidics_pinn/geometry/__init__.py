"""Domain-construction functions for `magnetofluidics_pinn`.

This subpackage exposes pure functions that turn a
[`DomainConfig`][magnetofluidics_pinn.config.DomainConfig] into a
[`Domain`][magnetofluidics_pinn.types.Domain] description, for the two
supported vessel shapes: a straight channel and a bifurcated vessel.
"""

# NOTE: explicit `as <same_name>` re-export (PEP 484) so PyCharm/mypy treat
# these as intentional public re-exports instead of unused imports.
from magnetofluidics_pinn.geometry.bifurcation import (
    build_bifurcation_domain as build_bifurcation_domain,
)
from magnetofluidics_pinn.geometry.channel import build_channel_domain as build_channel_domain

__all__ = [
    "build_channel_domain",
    "build_bifurcation_domain",
]
