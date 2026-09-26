"""Unit tests for the validation added to `trajectory.integrate_trajectory`.

Covers the new guard: a non-zero `particle_config.radius` (Faxén
correction active) combined with an initial position on or across the
symmetry axis must raise immediately, with every offending particle index
named; the pre-existing point-particle path (radius=0) must be unaffected.
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp


def _uniform_field_fn(coordinates: torch.Tensor) -> mfp.MagneticFieldSample:  # FIXED: was mfp.FieldSample
    return mfp.uniform_field(coordinates, magnitude=1.0, orientation=(1.0, 0.0))


class TestOnAxisFaxenValidation:
    def test_point_particle_on_axis_does_not_raise(self, constrained_network, domain) -> None:
        particle_config = mfp.ParticleConfig(radius=0.0)  # FIXED: ParticleConfig has no `position` field (see ParticleState)
        state = mfp.ParticleState(
            position=torch.tensor([0.0, 0.1]), velocity=torch.zeros(2), time=0.0,
        )
        scales = mfp.Scales(length=1.0, velocity=1.0, time=1.0, pressure=1.0, magnetic_field=1.0)
        mobility = torch.eye(2).unsqueeze(0)

        trajectories = mfp.integrate_trajectory(
            constrained_network, _uniform_field_fn, domain, scales, particle_config,
            [state], mobility, time_span=(0.0, 0.01),
        )
        assert len(trajectories) == 1
        assert len(trajectories[0]) >= 2

    def test_finite_radius_particle_on_axis_raises(self, constrained_network, domain) -> None:
        particle_config = mfp.ParticleConfig(radius=1.0e-3)  # FIXED: ParticleConfig has no `position` field (see ParticleState)
        state = mfp.ParticleState(
            position=torch.tensor([0.0, 0.1]), velocity=torch.zeros(2), time=0.0,
        )
        scales = mfp.Scales(length=1.0, velocity=1.0, time=1.0, pressure=1.0, magnetic_field=1.0)
        mobility = torch.eye(2).unsqueeze(0)

        with pytest.raises(ValueError, match="starting on or across the symmetry axis"):
            mfp.integrate_trajectory(
                constrained_network, _uniform_field_fn, domain, scales, particle_config,
                [state], mobility, time_span=(0.0, 0.01),
            )

    def test_finite_radius_particle_off_axis_does_not_raise(self, constrained_network, domain) -> None:
        particle_config = mfp.ParticleConfig(radius=1.0e-3)  # FIXED: position lives on ParticleState, not ParticleConfig
        state = mfp.ParticleState(
            position=torch.tensor([0.2, 0.1]), velocity=torch.zeros(2), time=0.0,
        )
        scales = mfp.Scales(length=1.0, velocity=1.0, time=1.0, pressure=1.0, magnetic_field=1.0)
        mobility = torch.eye(2).unsqueeze(0)

        trajectories = mfp.integrate_trajectory(
            constrained_network, _uniform_field_fn, domain, scales, particle_config,
            [state], mobility, time_span=(0.0, 0.005),
        )
        assert len(trajectories) == 1

    def test_error_names_only_the_offending_particle_indices(self, constrained_network, domain) -> None:
        particle_config = mfp.ParticleConfig(radius=1.0e-3)  # FIXED: ParticleConfig has no `position` field (see ParticleState)
        on_axis_state = mfp.ParticleState(position=torch.tensor([0.0, 0.1]), velocity=torch.zeros(2), time=0.0)
        off_axis_state = mfp.ParticleState(position=torch.tensor([0.3, 0.1]), velocity=torch.zeros(2), time=0.0)
        scales = mfp.Scales(length=1.0, velocity=1.0, time=1.0, pressure=1.0, magnetic_field=1.0)
        mobility = torch.eye(2).unsqueeze(0).repeat(3, 1, 1)

        with pytest.raises(ValueError, match=r"index \[0, 2\]"):
            mfp.integrate_trajectory(
                constrained_network, _uniform_field_fn, domain, scales, particle_config,
                [on_axis_state, off_axis_state, on_axis_state], mobility, time_span=(0.0, 0.01),
            )
