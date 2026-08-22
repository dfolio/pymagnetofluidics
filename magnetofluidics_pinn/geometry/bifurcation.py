"""Bifurcated vessel geometry.

Builds a branched domain (a main channel splitting into two daughter
branches), used in the later roadmap phase once the straight-channel case is
validated.
"""

from __future__ import annotations

from magnetofluidics_pinn.config import DomainConfig
from magnetofluidics_pinn.types import Domain


def build_bifurcation_domain(config: DomainConfig) -> Domain:
    """Build a bifurcated-vessel domain from a domain configuration.

    The returned `Domain` holds physical (SI, meter) values, matching
    `config`. Call
    [`scaling.nondimensionalize_domain`][magnetofluidics_pinn.scaling.nondimensionalize_domain]
    before passing it to `sampling` or `physics`. # CHANGED: documented the
    physical-to-dimensionless handoff.

    Args:
    - `config`: Domain configuration with `kind == "bifurcation"` and a
      defined `branch_angle`.

    Returns:
    - A [`Domain`][magnetofluidics_pinn.types.Domain] instance describing the
      bifurcated geometry, in physical (SI) units.

    Raises:
    - `ValueError`: If `config.kind` is not `"bifurcation"` or if
      `config.branch_angle` is `None`.
    """
    if config.kind != "bifurcation":
        raise ValueError(
            f"Expected a bifurcation domain configuration, got kind={config.kind!r}."
        )
    if config.branch_angle is None:
        raise ValueError("A bifurcation domain requires a `branch_angle`.")
    raise NotImplementedError("Implementation scheduled for a later roadmap phase.")
