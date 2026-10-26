"""Tests for `magnetofluidics_pinn.verification.metrics`.

This module had no dedicated test file at all before this pass, and it is
exactly where a signature change to `physics.fluid_residuals.stokes_residual`
(the third argument became a `DomainConfig`, not a `FluidConfig`) went
unnoticed in two of its three call sites: `evaluate_held_out_residual` and
`evaluate_residual_grid` both raised `AttributeError` on every call. Both are
fixed alongside this file; `TestEvaluateHeldOutResidual` and
`TestEvaluateResidualGrid` are the regression tests for that fix.

Most numeric checks below use `analytical_flow_network` (see `conftest.py`):
a network stand-in that returns the exact closed-form Hagen-Poiseuille field,
so every metric's *value* can be checked against a known answer (zero error,
for an exactly-correct network), not merely checked for "does not raise".
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn.verification.metrics import ProfileEvaluation


class TestSummarizeResidual:
    def test_max_and_rms_match_hand_computation(self) -> None:
        residual = torch.tensor([[1.0, -2.0, 0.0], [3.0, 0.0, -4.0]])
        summary = mfp.summarize_residual(residual, ("a", "b", "c"))

        assert summary.loc["a", "max_abs"] == pytest.approx(3.0)
        assert summary.loc["b", "max_abs"] == pytest.approx(2.0)
        assert summary.loc["c", "max_abs"] == pytest.approx(4.0)
        assert summary.loc["a", "rms"] == pytest.approx((1.0**2 + 3.0**2) ** 0.5 / (2**0.5))
        assert summary.loc["total", "max_abs"] == pytest.approx(4.0)
        assert summary.loc["total", "rms"] == pytest.approx(
            (1.0**2 + 2.0**2 + 0.0**2 + 3.0**2 + 0.0**2 + 4.0**2) ** 0.5 / (6**0.5)
        )

    def test_wrong_column_count_raises(self) -> None:
        with pytest.raises(ValueError, match=r"residual must have shape \(n_points, 2\)"):
            mfp.summarize_residual(torch.zeros(5, 3), ("a", "b"))

    def test_non_2d_residual_raises(self) -> None:
        with pytest.raises(ValueError, match=r"residual must have shape"):
            mfp.summarize_residual(torch.zeros(5), ("a",))


class TestEvaluateStructuralConstraints:
    def test_untrained_hard_wall_network_satisfies_every_probe(
            self, constrained_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        """The hard-wall wrapper's guarantee holds for any parameter value,
        so an untrained network must already pass every probe near-exactly.
        """
        metrics = mfp.evaluate_structural_constraints(constrained_network, domain, n_probe_points=200, random_seed=0)
        assert set(metrics) == {
            "max_abs_ur_axis", "max_abs_ur_wall", "max_abs_uz_wall", "max_abs_duz_dr_axis", "max_abs_dp_dr_axis",
        }
        for name, value in metrics.items():
            assert value < 1e-4, f"{name} = {value!r} is not structurally zero"

    def test_non_positive_n_probe_points_raises(self, constrained_network: torch.nn.Module, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="n_probe_points must be strictly positive"):
            mfp.evaluate_structural_constraints(constrained_network, domain, n_probe_points=0, random_seed=0)


class TestProfileEvaluation:
    def _valid_kwargs(self) -> dict:
        return {
            "z_stations": (0.5, 1.0), "radial_grid": torch.linspace(0.0, 1.0, 4),
            "predicted_uz": torch.zeros(2, 4), "predicted_ur": torch.zeros(2, 4), "analytical_uz": torch.zeros(4),
        }

    def test_valid_construction_does_not_raise(self) -> None:
        ProfileEvaluation(**self._valid_kwargs())

    def test_wrong_predicted_uz_shape_raises(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["predicted_uz"] = torch.zeros(2, 3)
        with pytest.raises(ValueError, match="predicted_uz must have shape"):
            ProfileEvaluation(**kwargs)

    def test_wrong_predicted_ur_shape_raises(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["predicted_ur"] = torch.zeros(3, 4)
        with pytest.raises(ValueError, match="predicted_ur must have shape"):
            ProfileEvaluation(**kwargs)

    def test_wrong_analytical_uz_shape_raises(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["analytical_uz"] = torch.zeros(3)
        with pytest.raises(ValueError, match="analytical_uz must have shape"):
            ProfileEvaluation(**kwargs)


class TestEvaluateVelocityProfiles:
    def test_exact_network_matches_analytical_profile(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        radial_grid = torch.linspace(0.0, domain.radius, 15)
        profile = mfp.evaluate_velocity_profiles(
            analytical_flow_network, domain, z_stations=(0.5 * domain.length, 0.9 * domain.length),
            radial_grid=radial_grid, peak_velocity=1.0,
        )
        assert torch.allclose(profile.predicted_uz[0], profile.analytical_uz, atol=1e-5)
        assert torch.allclose(profile.predicted_uz[1], profile.analytical_uz, atol=1e-5)
        assert torch.allclose(profile.predicted_ur, torch.zeros_like(profile.predicted_ur), atol=1e-5)

    def test_empty_z_stations_raises(self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="at least one axial station"):
            mfp.evaluate_velocity_profiles(analytical_flow_network, domain, (), torch.linspace(0.0, 1.0, 5), 1.0)

    def test_non_1d_radial_grid_raises(self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="radial_grid must be a 1-D tensor"):
            mfp.evaluate_velocity_profiles(analytical_flow_network, domain, (1.0,), torch.zeros(2, 2), 1.0)


class TestComputeGlobalAndMaxPointwiseError:
    def test_exact_network_has_near_zero_error(self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain) -> None:
        radial_grid = torch.linspace(0.0, domain.radius, 10)
        axial_grid = torch.linspace(0.0, domain.length, 10)
        assert mfp.compute_global_l2_error(analytical_flow_network, domain, radial_grid, axial_grid, 1.0) < 1e-5
        assert mfp.compute_max_pointwise_error(analytical_flow_network, domain, radial_grid, axial_grid, 1.0) < 1e-5

    def test_zero_output_network_has_full_relative_error(self, domain: mfp.Domain) -> None:
        class ZeroNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.zeros(coordinates.shape[0], 3)

        radial_grid = torch.linspace(0.0, domain.radius, 10)
        axial_grid = torch.linspace(0.0, domain.length, 10)
        # A network predicting exactly zero everywhere has a relative L2 error of
        # exactly 1.0 (its output contributes nothing to the numerator's difference
        # beyond the analytical field itself), and a max pointwise error equal to
        # the analytical field's own peak, u_peak.
        assert mfp.compute_global_l2_error(ZeroNetwork(), domain, radial_grid, axial_grid, 1.0) == pytest.approx(1.0)
        assert mfp.compute_max_pointwise_error(ZeroNetwork(), domain, radial_grid, axial_grid, 1.0) == pytest.approx(1.0)

    def test_degenerate_zero_peak_velocity_raises(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        radial_grid = torch.linspace(0.0, domain.radius, 5)
        axial_grid = torch.linspace(0.0, domain.length, 5)
        with pytest.raises(ValueError, match="Analytical velocity field norm is zero"):
            mfp.compute_global_l2_error(analytical_flow_network, domain, radial_grid, axial_grid, 0.0)


class TestComputeProfileInvariance:
    def test_identical_profiles_have_zero_error_and_zero_deviation(self) -> None:
        radial_grid = torch.linspace(0.0, 1.3, 5)
        analytical_uz = 1.0 - (radial_grid / 1.3) ** 2
        profile = ProfileEvaluation(
            z_stations=(0.5, 1.0, 2.0), radial_grid=radial_grid,
            predicted_uz=analytical_uz.unsqueeze(0).expand(3, -1).clone(),
            predicted_ur=torch.zeros(3, 5), analytical_uz=analytical_uz,
        )
        table = mfp.compute_profile_invariance(profile)
        assert (table["max_abs_error_vs_analytical"] < 1e-6).all()
        assert (table["max_abs_deviation_from_mean_profile"] < 1e-6).all()

    def test_a_shifted_station_is_flagged_by_both_columns(self) -> None:
        radial_grid = torch.linspace(0.0, 1.3, 5)
        analytical_uz = 1.0 - (radial_grid / 1.3) ** 2
        predicted = analytical_uz.unsqueeze(0).expand(3, -1).clone()
        predicted[1] += 0.2  # one station's profile is uniformly offset
        profile = ProfileEvaluation(
            z_stations=(0.5, 1.0, 2.0), radial_grid=radial_grid,
            predicted_uz=predicted, predicted_ur=torch.zeros(3, 5), analytical_uz=analytical_uz,
        )
        table = mfp.compute_profile_invariance(profile)
        assert table.loc[1, "max_abs_error_vs_analytical"] == pytest.approx(0.2, abs=1e-5)
        assert table.loc[1, "max_abs_deviation_from_mean_profile"] > table.loc[0, "max_abs_deviation_from_mean_profile"]


class TestComputeRadialLeakage:
    def test_zero_radial_velocity_has_zero_leakage(self) -> None:
        profile = ProfileEvaluation(
            z_stations=(1.0,), radial_grid=torch.linspace(0.0, 1.3, 5),
            predicted_uz=torch.zeros(1, 5), predicted_ur=torch.zeros(1, 5), analytical_uz=torch.zeros(5),
        )
        leakage = mfp.compute_radial_leakage(profile, peak_velocity=1.0)
        assert leakage["max_abs_radial_leakage"] == pytest.approx(0.0)
        assert leakage["normalized_radial_leakage"] == pytest.approx(0.0)

    def test_leakage_matches_hand_computation(self) -> None:
        predicted_ur = torch.tensor([[0.0, 0.05, -0.2, 0.1]])
        profile = ProfileEvaluation(
            z_stations=(1.0,), radial_grid=torch.linspace(0.0, 1.3, 4),
            predicted_uz=torch.zeros(1, 4), predicted_ur=predicted_ur, analytical_uz=torch.zeros(4),
        )
        leakage = mfp.compute_radial_leakage(profile, peak_velocity=2.0)
        assert leakage["max_abs_radial_leakage"] == pytest.approx(0.2)
        assert leakage["normalized_radial_leakage"] == pytest.approx(0.1)

    def test_non_positive_peak_velocity_raises(self) -> None:
        profile = ProfileEvaluation(
            z_stations=(1.0,), radial_grid=torch.linspace(0.0, 1.3, 4),
            predicted_uz=torch.zeros(1, 4), predicted_ur=torch.zeros(1, 4), analytical_uz=torch.zeros(4),
        )
        with pytest.raises(ValueError, match="peak_velocity must be finite and strictly positive"):
            mfp.compute_radial_leakage(profile, peak_velocity=0.0)


class TestComputeFlowRateCurveAndErrors:
    def test_exact_network_reproduces_reference_flow_rate(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        axial_grid = torch.linspace(0.0, domain.length, 6)
        curve = mfp.compute_flow_rate_curve(analytical_flow_network, domain, axial_grid, n_quadrature_points=128)
        reference_flow_rate = mfp.poiseuille_reference_flow_rate(domain.radius, peak_velocity=1.0)
        assert torch.allclose(curve, torch.full_like(curve, reference_flow_rate), atol=1e-3)

    def test_flow_rate_errors_match_hand_computation(self) -> None:
        curve = torch.tensor([1.0, 1.1, 0.9, 1.0])
        errors = mfp.compute_flow_rate_errors(curve, reference_flow_rate=1.0)
        assert errors["max_abs_flow_rate_error"] == pytest.approx(0.1)
        assert errors["max_relative_flow_rate_error"] == pytest.approx(0.1)
        assert errors["flow_rate_span"] == pytest.approx(0.2)
        assert errors["flow_rate_std"] == pytest.approx(curve.std(unbiased=False).item())

    def test_non_positive_reference_flow_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="reference_flow_rate must be finite and strictly positive"):
            mfp.compute_flow_rate_errors(torch.tensor([1.0, 1.0]), reference_flow_rate=0.0)

    def test_non_1d_axial_grid_raises(self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="axial_grid must be a 1-D tensor"):
            mfp.compute_flow_rate_curve(analytical_flow_network, domain, torch.zeros(2, 2))


class TestAxisPressureProfileAndGradient:
    def test_exact_network_matches_analytical_pressure(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        axial_positions, predicted_pressure = mfp.evaluate_axis_pressure_profile(
            analytical_flow_network, domain, n_axial_points=20
        )
        expected = mfp.hagen_poiseuille_pressure(
            torch.as_tensor(axial_positions), domain.radius, peak_velocity=1.0, domain_length=domain.length,
        ).numpy()
        assert predicted_pressure == pytest.approx(expected, abs=1e-4)

    def test_pressure_gradient_error_is_near_zero_for_exact_network(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        result = mfp.compute_pressure_gradient_error(analytical_flow_network, domain, peak_velocity=1.0)
        assert result["relative_error"] < 1e-4
        assert result["predicted_pressure_gradient"] == pytest.approx(result["analytical_pressure_gradient"], rel=1e-3)

    def test_too_few_axial_points_raises(self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="n_axial_points must be at least 2"):
            mfp.evaluate_axis_pressure_profile(analytical_flow_network, domain, n_axial_points=1)


class TestEvaluateHeldOutResidual:
    """Regression coverage: `evaluate_held_out_residual` was broken by a signature
    change (`stokes_residual`'s third argument became a `DomainConfig`) that no
    test caught, because this module had no test file at all.
    """

    def test_exact_network_has_near_machine_precision_residual(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain, domain_config: mfp.DomainConfig
    ) -> None:
        summary = mfp.evaluate_held_out_residual(
            analytical_flow_network, domain, domain_config, random_seed=0, n_points=500,
        )
        assert isinstance(summary, pd.DataFrame)
        # float32 machine-precision noise, not a genuine equation violation - the
        # same tolerance the shipped verification notebook uses for this exact check.
        assert summary.loc["total", "max_abs"] < 1e-2

    def test_non_positive_n_points_raises(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain, domain_config: mfp.DomainConfig
    ) -> None:
        with pytest.raises(ValueError, match="n_points must be strictly positive"):
            mfp.evaluate_held_out_residual(analytical_flow_network, domain, domain_config, random_seed=0, n_points=0)


class TestEvaluateResidualGrid:
    """Regression coverage: same fix as `TestEvaluateHeldOutResidual`, above."""

    def test_exact_network_has_near_machine_precision_residual(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain, domain_config: mfp.DomainConfig
    ) -> None:
        radial_grid = torch.linspace(1.0e-3, domain.radius, 10)
        axial_grid = torch.linspace(0.0, domain.length, 10)
        grids = mfp.evaluate_residual_grid(analytical_flow_network, domain_config, radial_grid, axial_grid)

        assert set(grids) == {"momentum_r", "momentum_z", "continuity"}
        for name, grid in grids.items():
            assert grid.shape == (10, 10)
            assert grid.max().item() < 1e-2, f"{name} residual is not near machine precision"

    def test_axis_touching_radial_grid_raises_under_standard_form(
            self, analytical_flow_network: torch.nn.Module, domain: mfp.Domain, domain_config: mfp.DomainConfig
    ) -> None:
        radial_grid = torch.linspace(0.0, domain.radius, 10)  # touches r=0
        axial_grid = torch.linspace(0.0, domain.length, 10)
        with pytest.raises(ValueError, match="touching or crossing the symmetry"):
            mfp.evaluate_residual_grid(
                analytical_flow_network, domain_config, radial_grid, axial_grid, residual_form="standard",
            )


class TestCompareTrajectoryToAnalytical:
    def test_exact_trajectory_has_near_zero_error(self, domain: mfp.Domain) -> None:
        peak_velocity, initial_radius = 1.0, 0.3
        time_grid = torch.tensor([0.0, 1.0, 2.0, 3.0])
        axial_velocity = peak_velocity * (1.0 - (initial_radius / domain.radius) ** 2)
        trajectory = [
            mfp.ParticleState(
                position=torch.tensor([initial_radius, 0.2 + axial_velocity * t.item()]),
                velocity=torch.tensor([0.0, axial_velocity]), time=t.item(),
            )
            for t in time_grid
        ]
        comparison = mfp.compare_trajectory_to_analytical(
            trajectory, initial_position=(initial_radius, 0.2), peak_velocity=peak_velocity, radius=domain.radius,
        )
        assert (comparison["radial_error"] < 1e-6).all()
        assert (comparison["axial_error"] < 1e-6).all()

    def test_empty_trajectory_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one ParticleState"):
            mfp.compare_trajectory_to_analytical([], initial_position=(0.0, 0.0), peak_velocity=1.0, radius=1.3)
