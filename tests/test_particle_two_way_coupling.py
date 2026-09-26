"""Regression coverage for the two-way particle-flow coupling path.

`sampling.collocation.sample_collocation_points_with_particle` and the
`particle_config` slot in a saved checkpoint (`io_utils.save_checkpoint` /
`load_checkpoint`) had no test coverage at all before this module: exactly
where the `SphericalObstacle -> ParticleConfig`/`ParticleState` refactor
left a boolean-mask cast broken (`sample_collocation_points_with_particle`
raised on every call with more than one candidate point) and a checkpoint
schema mismatch (a missing `particle_config` key skipped the intended
`RuntimeError` and fell through to a bare `KeyError`). `ParticleConfig`'s
own `radius=0.0` "point-particle limit" - which is also the field's own
default - is covered here too, since it previously made even
`ParticleConfig()` raise.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn import io_utils


class TestParticleConfigRadius:
    """`ParticleConfig.radius` must accept its own documented point-particle limit."""

    def test_default_radius_is_legal(self) -> None:
        """The field's own default, radius=0.0, must not raise."""
        config = mfp.ParticleConfig()
        assert config.radius == 0.0

    def test_zero_radius_is_legal(self) -> None:
        config = mfp.ParticleConfig(radius=0.0)
        assert config.radius == 0.0

    @pytest.mark.parametrize("bad_radius", [-1.0e-3, float("nan"), float("inf"), float("-inf")])
    def test_invalid_radius_raises(self, bad_radius: float) -> None:
        with pytest.raises(ValueError, match=f"radius must be finite and non-negative; got {bad_radius!r}."):
            mfp.ParticleConfig(radius=bad_radius)


class TestSampleCollocationPointsWithParticle:
    """`sampling.collocation.sample_collocation_points_with_particle`."""

    def test_returns_requested_point_counts(self, domain: mfp.Domain, particle_config: mfp.ParticleConfig,
                                             particle_state: mfp.ParticleState) -> None:
        collocation = mfp.sample_collocation_points_with_particle(
            domain=domain, particle_state=particle_state, particle_config=particle_config,
            n_interior=200, n_boundary=9, n_surface_points=16, random_seed=0,
        )
        assert collocation.interior.shape == (200, 2)
        assert collocation.boundary.shape == (9, 2)
        assert collocation.particle_surface.shape == (16, 2)

    def test_interior_points_stay_outside_the_particle(
            self, domain: mfp.Domain, particle_config: mfp.ParticleConfig, particle_state: mfp.ParticleState
    ) -> None:
        """Regression test for the `int(...)`-cast bug: every surviving interior point
        must lie strictly outside the excluded particle disk, not merely have been
        drawn (which the broken cast would have crashed on before even reaching here).
        """
        collocation = mfp.sample_collocation_points_with_particle(
            domain=domain, particle_state=particle_state, particle_config=particle_config,
            n_interior=500, n_boundary=9, n_surface_points=8, random_seed=1,
        )
        radial, axial = collocation.interior[:, 0], collocation.interior[:, 1]
        squared_distance_to_particle = (axial - particle_state.axial_position) ** 2 + radial**2
        assert torch.all(squared_distance_to_particle >= particle_config.radius**2)

    def test_particle_surface_points_lie_on_the_sphere(
            self, domain: mfp.Domain, particle_config: mfp.ParticleConfig, particle_state: mfp.ParticleState
    ) -> None:
        collocation = mfp.sample_collocation_points_with_particle(
            domain=domain, particle_state=particle_state, particle_config=particle_config,
            n_interior=30, n_boundary=9, n_surface_points=64, random_seed=2,
        )
        radial, axial = collocation.particle_surface[:, 0], collocation.particle_surface[:, 1]
        distance_to_center = torch.sqrt(radial.square() + (axial - particle_state.axial_position).square())
        assert torch.allclose(distance_to_center, torch.full_like(distance_to_center, particle_config.radius), atol=1e-5)

    def test_raises_when_particle_touches_the_wall(self, domain: mfp.Domain, particle_state: mfp.ParticleState) -> None:
        oversized_particle = mfp.ParticleConfig(radius=domain.radius)
        with pytest.raises(ValueError, match="must be strictly less than domain.radius"):
            mfp.sample_collocation_points_with_particle(
                domain=domain, particle_state=particle_state, particle_config=oversized_particle,
                n_interior=20, n_boundary=9, n_surface_points=8, random_seed=0,
            )

    def test_raises_when_particle_crosses_the_outlet(self, domain: mfp.Domain, particle_config: mfp.ParticleConfig) -> None:
        near_outlet_state = mfp.ParticleState(
            position=torch.tensor([0.0, domain.length - 0.01]), velocity=torch.zeros(2), time=0.0,
        )
        with pytest.raises(ValueError, match="fit strictly inside the channel's axial extent"):
            mfp.sample_collocation_points_with_particle(
                domain=domain, particle_state=near_outlet_state, particle_config=particle_config,
                n_interior=20, n_boundary=9, n_surface_points=8, random_seed=0,
            )

    def test_raises_on_insufficient_oversampling(self, domain: mfp.Domain, particle_state: mfp.ParticleState) -> None:
        """A particle occupying a large fraction of the cross-section, combined with
        an `oversampling_factor` barely above 1.0, cannot survive enough rejection
        draws; this must raise RuntimeError rather than silently return a short batch.
        """
        large_particle = mfp.ParticleConfig(radius=0.9 * domain.radius)
        with pytest.raises(RuntimeError, match="fewer than the requested n_interior"):
            mfp.sample_collocation_points_with_particle(
                domain=domain, particle_state=particle_state, particle_config=large_particle,
                n_interior=5_000, n_boundary=9, n_surface_points=8, random_seed=0,
                oversampling_factor=1.01,
            )


class TestCheckpointWithParticleConfig:
    """`io_utils.save_checkpoint` / `load_checkpoint` round-tripping `particle_config`."""

    def _build(self, domain: mfp.Domain, training_config: mfp.TrainingConfig) -> torch.nn.Module:
        raw = mfp.build_mlp(n_inputs=2, n_outputs=3, hidden_layers=(4, 4), training_config=training_config)
        return mfp.apply_hard_wall_constraint(raw, domain)

    def test_round_trip_preserves_particle_config(
            self, domain: mfp.Domain, domain_config: mfp.FluidConfig, field_config: mfp.MagneticFieldConfig,
            particle_config: mfp.ParticleConfig,
    ) -> None:
        training_config = mfp.TrainingConfig(device="cpu")
        network = self._build(domain, training_config)
        domain_config = mfp.DomainConfig(kind="channel", length=domain.length, radius=domain.radius, fluid=domain_config)
        architecture = io_utils.NetworkArchitecture(
            n_inputs=2, n_outputs=3, hidden_layers=(4, 4), hard_wall_constraint_radius=domain.radius,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.pt"
            io_utils.save_checkpoint(
                network, checkpoint_path, architecture, domain_config, domain_config, field_config,
                particle_config, training_config,
            )
            loaded_network, _, _, _, loaded_particle_config, _ = io_utils.load_checkpoint(
                checkpoint_path, device="cpu"
            )

        assert loaded_particle_config == particle_config
        for key, value in network.state_dict().items():
            assert torch.equal(value, loaded_network.state_dict()[key])

    def test_missing_particle_config_key_raises_runtime_error(
            self, domain: mfp.Domain, domain_config: mfp.FluidConfig, field_config: mfp.MagneticFieldConfig,
            particle_config: mfp.ParticleConfig,
    ) -> None:
        """A checkpoint written without `particle_config` must fail loudly and early
        (`RuntimeError`, naming the missing key), not with a bare `KeyError` deep
        inside `load_checkpoint`'s return statement.
        """
        training_config = mfp.TrainingConfig(device="cpu")
        network = self._build(domain, training_config)
        domain_config = mfp.DomainConfig(kind="channel", length=domain.length, radius=domain.radius, fluid=domain_config)
        architecture = io_utils.NetworkArchitecture(
            n_inputs=2, n_outputs=3, hidden_layers=(4, 4), hard_wall_constraint_radius=domain.radius,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.pt"
            io_utils.save_checkpoint(
                network, checkpoint_path, architecture, domain_config, domain_config, field_config,
                particle_config, training_config,
            )
            # Simulate a checkpoint written before `particle_config` existed in the
            # schema, by loading the payload back and stripping the key out.
            torch.serialization.add_safe_globals(io_utils._SAFE_CHECKPOINT_GLOBALS)
            payload = torch.load(checkpoint_path, weights_only=True)
            del payload["particle_config"]
            torch.save(payload, checkpoint_path)

            with pytest.raises(RuntimeError, match=r"missing keys:.*particle_config"):
                io_utils.load_checkpoint(checkpoint_path, device="cpu")
