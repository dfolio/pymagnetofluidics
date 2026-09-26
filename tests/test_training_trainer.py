"""Unit tests for `magnetofluidics_pinn.training.trainer`.

Covers, in order: the decomposed interior-residual helper
(`_stokes_residual_losses`), the two new loss terms (`_positivity_loss`,
`_conservation_loss`), the `_LossComponents` -> weighted-total wiring
(`_evaluate_loss_components`, `compose_loss`), the plain-float bookkeeping
(`_record_loss_components`, `_format_progress_line`), and finally `train()`
itself end-to-end, both its nominal path and every `ValueError` its
docstring commits to.

Private functions are imported directly from `training.trainer`: they are
exactly the units this change set introduced or altered, so testing them
in isolation - not only through the public `train()` - is what lets a
failure point at *which* term broke instead of just "training changed".
"""

from __future__ import annotations

import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.training import trainer as trainer_module
from .conftest import make_poiseuille_network


# ---------------------------------------------------------------------------
# _stokes_residual_losses
# ---------------------------------------------------------------------------

class TestStokesResidualLosses:
    def test_matches_manual_per_column_mean(self, constrained_network, domain, domain_config) -> None:
        coordinates = torch.rand(64, 2)
        coordinates[:, 0] = coordinates[:, 0] * (domain.radius - 0.05) + 0.05  # keep off-axis
        coordinates[:, 1] = coordinates[:, 1] * domain.length
        coordinates = coordinates.clone().requires_grad_(True)
        
        residual = stokes_residual(constrained_network, coordinates, domain_config)
        expected = (
            torch.mean(residual[:, 0:1].square()),
            torch.mean(residual[:, 1:2].square()),
            torch.mean(residual[:, 2:3].square()),
        )
        
        # Recompute independently through the function under test, on a
        # fresh (but identically-seeded) coordinate tensor, since
        # `stokes_residual` consumes the autograd graph of `coordinates`.
        coordinates2 = coordinates.detach().clone().requires_grad_(True)
        momentum_r, momentum_z, continuity = trainer_module._stokes_residual_losses(
            constrained_network, coordinates2, domain_config
        )
        
        assert momentum_r.item() == pytest.approx(expected[0].item(), rel=1e-5)
        assert momentum_z.item() == pytest.approx(expected[1].item(), rel=1e-5)
        assert continuity.item() == pytest.approx(expected[2].item(), rel=1e-5)
    
    def test_zero_for_the_exact_analytic_solution(self, domain, domain_config) -> None:
        """The analytic Poiseuille field exactly satisfies Stokes flow: all three residuals vanish.

        Unlike `physics.conservation`'s tests (which only look at u_z and
        can safely ignore pressure), the momentum equations *do* depend on
        pressure: z-momentum needs the matching linear pressure gradient
        dp/dz = -4 * peak_velocity / R**2 that balances the parabolic
        profile's viscous term, or the residual is a nonzero constant, not
        the profile itself being wrong.
        """
        radius, peak_velocity = domain.radius, 1.0
        pressure_gradient = -4.0 * peak_velocity / radius ** 2
        
        def exact_stokes_network(coordinates: torch.Tensor) -> torch.Tensor:
            radial, axial = coordinates[:, 0:1], coordinates[:, 1:2]
            velocity_r = torch.zeros_like(radial)
            velocity_z = peak_velocity * (1.0 - (radial / radius).square())
            pressure = pressure_gradient * axial
            return torch.cat([velocity_r, velocity_z, pressure], dim=1)
        
        coordinates = torch.rand(64, 2)
        coordinates[:, 0] = coordinates[:, 0] * (domain.radius - 0.05) + 0.05
        coordinates[:, 1] = coordinates[:, 1] * domain.length
        coordinates = coordinates.requires_grad_(True)
        
        momentum_r, momentum_z, continuity = trainer_module._stokes_residual_losses(
            exact_stokes_network, coordinates, domain_config
        )
        assert momentum_r.item() == pytest.approx(0.0, abs=1e-10)
        assert momentum_z.item() == pytest.approx(0.0, abs=1e-10)
        assert continuity.item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# _positivity_loss
# ---------------------------------------------------------------------------

class TestPositivityLoss:
    def test_zero_when_axial_velocity_is_non_negative(self) -> None:
        def all_positive_network(coordinates: torch.Tensor) -> torch.Tensor:
            n = coordinates.shape[0]
            return torch.cat([torch.zeros(n, 1), torch.ones(n, 1), torch.zeros(n, 1)], dim=1)
        
        coordinates = torch.rand(32, 2)
        loss = trainer_module._positivity_loss(all_positive_network, coordinates)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)
    
    def test_matches_closed_form_for_a_known_negative_field(self) -> None:
        """E[ReLU(-u_z)^2] for a constant u_z = -c is exactly c^2."""
        c = 0.37
        
        def constant_negative_network(coordinates: torch.Tensor) -> torch.Tensor:
            n = coordinates.shape[0]
            return torch.cat([torch.zeros(n, 1), torch.full((n, 1), -c), torch.zeros(n, 1)], dim=1)
        
        coordinates = torch.rand(10, 2)
        loss = trainer_module._positivity_loss(constant_negative_network, coordinates)
        assert loss.item() == pytest.approx(c ** 2, rel=1e-6)
    
    def test_only_penalizes_the_negative_part(self) -> None:
        """Mixed positive/negative predictions: only the negative entries contribute."""
        
        def mixed_network(coordinates: torch.Tensor) -> torch.Tensor:
            n = coordinates.shape[0]
            u_z = torch.tensor([1.0, -2.0, 0.5, -0.5]).reshape(n, 1)
            return torch.cat([torch.zeros(n, 1), u_z, torch.zeros(n, 1)], dim=1)
        
        coordinates = torch.rand(4, 2)
        loss = trainer_module._positivity_loss(mixed_network, coordinates)
        expected = (0.0 ** 2 + 2.0 ** 2 + 0.0 ** 2 + 0.5 ** 2) / 4.0
        assert loss.item() == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# _conservation_loss
# ---------------------------------------------------------------------------

class TestConservationLoss:
    def test_zero_for_the_exact_analytic_solution(self, domain) -> None:
        network = make_poiseuille_network(domain.radius, 1.0)
        loss = trainer_module._conservation_loss(
            network, domain, peak_velocity=1.0, n_stations=6, n_quadrature_points=256, device=torch.device("cpu"),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_positive_for_a_non_conserving_field(self, domain) -> None:
        """A field whose flow rate drifts with z must score a strictly positive loss."""
        
        def drifting_network(coordinates: torch.Tensor) -> torch.Tensor:
            radial, axial = coordinates[:, 0:1], coordinates[:, 1:2]
            velocity_z = (1.0 - (radial / domain.radius).square()) * (1.0 - 0.1 * axial)
            return torch.cat([torch.zeros_like(radial), velocity_z, torch.zeros_like(radial)], dim=1)
        
        loss = trainer_module._conservation_loss(
            drifting_network, domain, peak_velocity=1.0, n_stations=6, n_quadrature_points=128,
            device=torch.device("cpu"),
        )
        assert loss.item() > 1e-4
    
    def test_differentiable_through_network_parameters(self, domain) -> None:
        # FIXED: build_mlp now takes an explicit training_config instead of device=.
        training_config = mfp.TrainingConfig(device="cpu")
        network = mfp.build_mlp(2, 3, (8, 8), training_config=training_config)
        loss = trainer_module._conservation_loss(
            network, domain, peak_velocity=1.0, n_stations=4, n_quadrature_points=16, device=torch.device("cpu"),
        )
        loss.backward()
        gradients = [p.grad for p in network.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)


# ---------------------------------------------------------------------------
# _evaluate_loss_components / _record_loss_components
# ---------------------------------------------------------------------------

class TestEvaluateLossComponents:
    def _tiny_batch(self, domain):
        # FIXED: _build_training_batch now takes a required `coupling_config`
        # positional argument (right after `domain`); `None` reproduces the
        # particle-free behavior every call in this file relies on.
        return trainer_module._build_training_batch(domain, None, n_interior=40, n_boundary=9,
                                                    random_seed=0, device=torch.device("cpu"),
                                                    axis_clearance_fraction=None)
    
    def test_returns_all_eight_components_plus_total(self, constrained_network, domain, domain_config) -> None:
        training_config = mfp.TrainingConfig(device="cpu")
        batch = self._tiny_batch(domain)
        # FIXED: _evaluate_loss_components now takes a required `coupling_config`
        # trailing argument; `None` selects the particle-free loss composition.
        components = trainer_module._evaluate_loss_components(
            constrained_network, batch, domain, domain_config, training_config, None
        )
        for name in trainer_module.LOSS_COMPONENT_NAMES:
            assert torch.is_tensor(getattr(components, name))
        assert torch.is_tensor(components.total)
    
    def test_total_equals_manually_reweighted_sum(self, constrained_network, domain, domain_config) -> None:
        training_config = mfp.TrainingConfig(
            device="cpu", momentum_loss_weight=2.0, continuity_loss_weight=3.0,
            positivity_loss_weight=0.5, conservation_loss_weight=1.5,
        )
        batch = self._tiny_batch(domain)
        components = trainer_module._evaluate_loss_components(
            constrained_network, batch, domain, domain_config, training_config, None
        )
        expected_total = (
                2.0 * components.momentum_r + 2.0 * components.momentum_z + 3.0 * components.continuity
                + trainer_module._BOUNDARY_LOSS_WEIGHT * (components.wall + components.inlet + components.outlet)
                + 0.5 * components.positivity + 1.5 * components.conservation
        )
        assert components.total.item() == pytest.approx(expected_total.item(), rel=1e-5)
    
    def test_zero_weight_removes_a_term_from_the_total(self, constrained_network, domain, domain_config) -> None:
        """Setting a weight to zero must make the total insensitive to that term's value."""
        training_config_off = mfp.TrainingConfig(device="cpu", positivity_loss_weight=0.0)
        training_config_on = mfp.TrainingConfig(device="cpu", positivity_loss_weight=1.0)
        batch = self._tiny_batch(domain)
        
        components_off = trainer_module._evaluate_loss_components(
            constrained_network, batch, domain, domain_config, training_config_off, None
        )
        components_on = trainer_module._evaluate_loss_components(
            constrained_network, batch, domain, domain_config, training_config_on, None
        )
        # Everything but the positivity weight is identical, so the totals
        # must differ by exactly the (weighted) positivity term - up to the
        # float32 precision floor for subtracting two ~O(100) totals
        # (~total magnitude * 2**-23), not a tight relative tolerance.
        difference = components_on.total.item() - components_off.total.item()
        assert difference == pytest.approx(components_on.positivity.item(), abs=1e-4)


class TestRecordLossComponents:
    def test_produces_plain_floats_for_every_tracked_name(self, constrained_network, domain, domain_config) -> None:
        training_config = mfp.TrainingConfig(device="cpu")
        batch = trainer_module._build_training_batch(domain, None, 20, 9, 0,
                                                     axis_clearance_fraction=training_config.axis_clearance_fraction,
                                                     device=torch.device("cpu"))
        components = trainer_module._evaluate_loss_components(
            constrained_network, batch, domain, domain_config, training_config, None
        )
        recorded = trainer_module._record_loss_components(components)
        
        assert set(recorded) == set(trainer_module.LOSS_COMPONENT_NAMES) | {"total"}
        assert all(isinstance(value, float) for value in recorded.values())
        assert recorded["total"] == pytest.approx(components.total.item())


# ---------------------------------------------------------------------------
# _format_progress_line
# ---------------------------------------------------------------------------

def test_format_progress_line_renders_every_entry_in_order() -> None:
    line = trainer_module._format_progress_line("Epoch     3", {"total": 1.5, "momentum_r": 0.25})
    assert line.startswith("Epoch     3 | ")
    assert "Total: 1.50e+00" in line
    assert "Momentum R: 2.50e-01" in line
    assert line.index("Total") < line.index("Momentum R")


# ---------------------------------------------------------------------------
# train() end-to-end
# ---------------------------------------------------------------------------

class TestTrainEndToEnd:
    def test_nominal_run_returns_trained_network_and_full_history(
            self, constrained_network, domain, domain_config, field_config
    ) -> None:
        training_config = mfp.TrainingConfig(
            n_interior_points=30, n_boundary_points=9, n_epochs=3, device="cpu",
            use_lbfgs_refinement=True, lbfgs_n_interior_points=15, lbfgs_n_boundary_points=6,
            lbfgs_rounds=1, lbfgs_iterations_per_round=2,
            n_conservation_stations=3, n_conservation_quadrature_points=8,
        )
        trained_network, history = mfp.train(
            constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config
        )
        
        assert trained_network is not constrained_network  # a private copy was trained
        assert len(history.adam.step) == 3
        assert len(history.lbfgs.step) >= 1
        for name in trainer_module.LOSS_COMPONENT_NAMES + ("total",):
            assert len(getattr(history.adam, name)) == 3
    
    def test_original_network_is_not_mutated(self, constrained_network, domain, domain_config, field_config) -> None:
        original_state = {k: v.clone() for k, v in constrained_network.state_dict().items()}
        training_config = mfp.TrainingConfig(
            n_interior_points=20, n_boundary_points=9, n_epochs=3, device="cpu", use_lbfgs_refinement=False,
        )
        mfp.train(constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config)
        
        for key, value in constrained_network.state_dict().items():
            assert torch.equal(value, original_state[key])
    
    def test_wall_loss_is_near_zero_throughout_after_hard_constraint(
            self, constrained_network, domain, domain_config, field_config
    ) -> None:
        """Both u_r and u_z are now hard-constrained at the wall, so the soft wall loss is vestigial."""
        training_config = mfp.TrainingConfig(
            n_interior_points=30, n_boundary_points=9, n_epochs=3, device="cpu", use_lbfgs_refinement=False,
        )
        _, history = mfp.train(constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config)
        assert max(history.adam.wall) < 1e-8
    
    def test_disabled_lbfgs_yields_empty_lbfgs_history(
            self, constrained_network, domain, domain_config, field_config
    ) -> None:
        training_config = mfp.TrainingConfig(
            n_interior_points=20, n_boundary_points=9, n_epochs=2, device="cpu", use_lbfgs_refinement=False,
        )
        _, history = mfp.train(constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config)
        for name in trainer_module.LOSS_COMPONENT_NAMES + ("step", "total"):
            assert getattr(history.lbfgs, name) == ()
    
    def test_print_summary_runs_without_error(self, constrained_network, domain, domain_config, field_config) -> None:
        training_config = mfp.TrainingConfig(
            n_interior_points=20, n_boundary_points=9, n_epochs=2, device="cpu", use_lbfgs_refinement=False,
        )
        _, history = mfp.train(constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config)
        history.print_summary()  # must not raise
    
    def test_raises_on_non_positive_n_epochs(self, constrained_network, domain, domain_config, field_config) -> None:
        training_config = mfp.TrainingConfig(n_interior_points=10, n_boundary_points=9, n_epochs=1, device="cpu")
        object.__setattr__(training_config, "n_epochs", 0)  # bypass __post_init__ to hit train()'s own check
        with pytest.raises(ValueError, match="n_epochs must be strictly positive"):
            mfp.train(constrained_network, domain, fluid_config=domain_config, field_config=field_config, training_config=training_config)


# ---------------------------------------------------------------------------
# train() with a TwoWayCouplingConfig (regression coverage: this path had no
# test coverage at all, and was broken by three independent bugs introduced
# by the SphericalObstacle -> ParticleConfig/ParticleState refactor:
# `_component_names` keying its accumulator dict under "obstacle" while
# `_record_loss_components` produced "particle" (KeyError on the first
# step), `_evaluate_loss_components` reading a `coupling_config.
# particle_loss_weight` that `TwoWayCouplingConfig` defined as
# `object_loss_weight` (AttributeError), and `TwoWayCouplingConfig.
# __post_init__` rejecting its own zero-velocity default (ValueError).
# ---------------------------------------------------------------------------

class TestTrainWithCoupling:
    def test_nominal_run_returns_particle_loss_history(
            self, constrained_network, domain, domain_config, field_config, particle_config
    ) -> None:
        training_config = mfp.TrainingConfig(
            n_interior_points=24, n_boundary_points=9, n_epochs=2, device="cpu",
            use_lbfgs_refinement=True, lbfgs_n_interior_points=12, lbfgs_n_boundary_points=6,
            lbfgs_rounds=1, lbfgs_iterations_per_round=2,
        )
        coupling_config = mfp.TwoWayCouplingConfig(
            particle_config=particle_config, particle_position=(0.0, domain.length / 2.0), n_surface_points=10,
        )
        trained_network, history = mfp.train(
            constrained_network, domain, fluid_config=domain_config, field_config=field_config,
            training_config=training_config, coupling_config=coupling_config,
        )

        assert trained_network is not constrained_network
        assert history.adam.particle is not None
        assert len(history.adam.particle) == 2
        assert history.lbfgs.particle is not None
        assert len(history.lbfgs.particle) >= 1
        assert all(torch.isfinite(torch.tensor(value)) for value in history.adam.particle)

    def test_default_zero_velocity_does_not_raise(self, particle_config) -> None:
        """The documented `(0.0, 0.0)` default ("a fixed, anchored particle") must be legal."""
        coupling_config = mfp.TwoWayCouplingConfig(particle_config=particle_config)
        assert coupling_config.particle_velocity == (0.0, 0.0)
        assert coupling_config.particle_loss_weight == mfp.DEFAULT_BOUNDARY_LOSS_WEIGHT

    def test_non_finite_velocity_raises(self, particle_config) -> None:
        with pytest.raises(ValueError, match="particle_velocity components must be finite"):
            mfp.TwoWayCouplingConfig(particle_config=particle_config, particle_velocity=(0.0, float("nan")))
