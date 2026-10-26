"""Tests for `magnetofluidics_pinn.geometry`.

Every other test module gets its domain from the `domain` fixture, which
constructs an already-dimensionless `Domain` by hand. That is convenient, but
it means `build_channel_domain` (the function every real user calls first) was
never executed by any test. This module calls the builders directly.

`DomainConfig.__post_init__` already rejects a non-positive length or radius,
so the builders' own guards are only reachable by bypassing that validation
(`object.__setattr__` on the frozen dataclass), which is what the "defence in
depth" tests below do on purpose.
"""

from __future__ import annotations

import dataclasses

import pytest

import magnetofluidics_pinn as mfp


def _bypass_validation(config: mfp.DomainConfig, **overrides) -> mfp.DomainConfig:
    """Return a copy of `config` with fields overwritten *after* `__post_init__` ran."""
    clone = dataclasses.replace(config)
    for name, value in overrides.items():
        object.__setattr__(clone, name, value)
    return clone


class TestBuildChannelDomain:
    def test_returns_a_channel_domain_with_the_configured_extents(self) -> None:
        config = mfp.DomainConfig(kind="channel", length=3.0e-3, radius=7.5e-4)
        domain = mfp.build_channel_domain(config)

        assert isinstance(domain, mfp.Domain)
        assert domain.kind == "channel"
        assert domain.length == pytest.approx(3.0e-3)
        assert domain.radius == pytest.approx(7.5e-4)
        assert domain.branch_angle is None

    def test_values_stay_in_physical_si_units(self) -> None:
        """No scaling happens here: that is `nondimensionalize_domain`'s job."""
        config = mfp.DomainConfig(length=2.0e-3, radius=5.0e-4)
        domain = mfp.build_channel_domain(config)
        assert (domain.length, domain.radius) == (config.length, config.radius)

    def test_domain_is_immutable(self) -> None:
        domain = mfp.build_channel_domain(mfp.DomainConfig())
        with pytest.raises(AttributeError):
            domain.radius = 1.0  # type: ignore[misc]

    def test_rejects_a_bifurcation_configuration(self) -> None:
        config = mfp.DomainConfig(kind="bifurcation", branch_angle=0.4)
        with pytest.raises(ValueError, match="Expected a channel domain configuration"):
            mfp.build_channel_domain(config)

    @pytest.mark.parametrize("bad_length", [0.0, -1.0e-3])
    def test_rejects_non_positive_length(self, bad_length: float) -> None:
        config = _bypass_validation(mfp.DomainConfig(), length=bad_length)
        with pytest.raises(ValueError, match="config.length must be strictly positive"):
            mfp.build_channel_domain(config)

    @pytest.mark.parametrize("bad_radius", [0.0, -1.0e-3])
    def test_rejects_non_positive_radius(self, bad_radius: float) -> None:
        config = _bypass_validation(mfp.DomainConfig(), radius=bad_radius)
        with pytest.raises(ValueError, match="config.radius must be strictly positive"):
            mfp.build_channel_domain(config)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "build_channel_domain drops DomainConfig.u_max (Domain.u_max stays at its 0.0 default), "
            "although nondimensionalize_domain/redimensionalize_domain and the hard-wall constraint "
            "all consume Domain.u_max. Flagged, not fixed: forwarding it alone would activate "
            "_HardWallConstrainedFlow's 2*u_max base profile, whose peak (2*u_max) disagrees with the "
            "trainer's unit inlet peak - see test_networks_constraints.py's matching xfail. Decide the "
            "intended u_max convention (peak vs mean velocity) first."
        ),
    )
    def test_forwards_the_configured_maximum_velocity(self) -> None:
        config = mfp.DomainConfig(u_max=2.0e-3)
        assert mfp.build_channel_domain(config).u_max == pytest.approx(2.0e-3)


class TestBuildBifurcationDomain:
    """A documented Phase-3 stub: validate the request, then fail loudly."""

    def test_valid_configuration_raises_not_implemented(self) -> None:
        config = mfp.DomainConfig(kind="bifurcation", branch_angle=0.4)
        with pytest.raises(NotImplementedError, match="later roadmap phase"):
            mfp.build_bifurcation_domain(config)

    def test_rejects_a_channel_configuration(self) -> None:
        with pytest.raises(ValueError, match="Expected a bifurcation domain configuration"):
            mfp.build_bifurcation_domain(mfp.DomainConfig(kind="channel"))

    def test_rejects_a_missing_branch_angle(self) -> None:
        config = _bypass_validation(mfp.DomainConfig(kind="bifurcation", branch_angle=0.4), branch_angle=None)
        with pytest.raises(ValueError, match="requires a `branch_angle`"):
            mfp.build_bifurcation_domain(config)
