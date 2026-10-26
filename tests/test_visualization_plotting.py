"""Smoke tests for `magnetofluidics_pinn.visualization.plotting`.

Every plot function is a pure renderer returning a `matplotlib.figure.Figure`
(none touches `pyplot`), so there is no global figure state to leak between
tests. Each test therefore checks three things, in increasing strictness:

1. a `Figure` comes back;
2. it actually *renders* (saving to an in-memory PNG runs `tight_layout`,
   legend placement and colorbar drawing, which construction alone does not);
3. a few structural facts that would change if the wrong data were plotted
   (line counts, axis limits, labels), plus each documented rejection.

The Agg backend is forced so the suite runs on a machine with no display.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # must precede the imports below that pull in matplotlib

import matplotlib.figure
import numpy as np
import pytest
import torch

import magnetofluidics_pinn as mfp
from magnetofluidics_pinn.verification.metrics import ProfileEvaluation


def _assert_renders(figure: matplotlib.figure.Figure) -> None:
    """Assert `figure` is a `Figure` that survives a full draw to PNG bytes."""
    assert isinstance(figure, matplotlib.figure.Figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 0


def _profile_evaluation(n_stations: int = 2, n_radial: int = 6) -> ProfileEvaluation:
    radial_grid = torch.linspace(0.0, 1.3, n_radial)
    analytical_uz = 1.0 - (radial_grid / 1.3) ** 2
    offsets = 0.01 * torch.arange(n_stations, dtype=torch.float32).unsqueeze(1)
    return ProfileEvaluation(
        z_stations=tuple(float(index + 1) for index in range(n_stations)),
        radial_grid=radial_grid,
        predicted_uz=analytical_uz.unsqueeze(0).expand(n_stations, -1).clone() + offsets,
        predicted_ur=0.005 * torch.ones(n_stations, n_radial),
        analytical_uz=analytical_uz,
    )


def _loss_history(n_steps: int) -> mfp.LossHistory:
    names = ("step",) + mfp.LOSS_COMPONENT_NAMES + ("total",)
    fields = {name: tuple(1.0 / (1 + index) for index in range(n_steps)) for name in names}
    fields["step"] = tuple(range(n_steps))
    return mfp.LossHistory(**fields)


class TestPlotStreamlines:
    def test_returns_a_rendering_figure_with_domain_limits(
            self, constrained_network: torch.nn.Module, domain: mfp.Domain
    ) -> None:
        figure = mfp.plot_streamlines(constrained_network, domain, resolution=12)
        _assert_renders(figure)
        axes = figure.axes[0]
        assert axes.get_xlim() == pytest.approx((0.0, domain.length))
        assert axes.get_ylim() == pytest.approx((0.0, domain.radius))
        assert axes.get_xlabel() == "Axial position $z^*$"

    @pytest.mark.parametrize("bad_resolution", [0, -5])
    def test_non_positive_resolution_raises(
            self, constrained_network: torch.nn.Module, domain: mfp.Domain, bad_resolution: int
    ) -> None:
        with pytest.raises(ValueError, match="resolution must be strictly positive"):
            mfp.plot_streamlines(constrained_network, domain, resolution=bad_resolution)

    def test_non_channel_domain_raises(self, constrained_network: torch.nn.Module) -> None:
        bifurcation = mfp.Domain(kind="bifurcation", length=4.0, radius=1.3, branch_angle=0.3)
        with pytest.raises(NotImplementedError, match="only supports domain.kind == 'channel'"):
            mfp.plot_streamlines(constrained_network, bifurcation, resolution=8)


class TestPlotTrajectories:
    @staticmethod
    def _trajectory(n_steps: int, radial: float) -> list[mfp.ParticleState]:
        return [
            mfp.ParticleState(
                position=torch.tensor([radial, 0.1 * step]), velocity=torch.zeros(2), time=float(step),
            )
            for step in range(n_steps)
        ]

    def test_draws_one_path_per_particle_plus_the_wall_and_axis_guides(self, domain: mfp.Domain) -> None:
        trajectories = [self._trajectory(5, 0.2), self._trajectory(5, 0.6), self._trajectory(3, 0.9)]
        figure = mfp.plot_trajectories(trajectories, domain)
        _assert_renders(figure)
        # Three particle paths, plus `axhline` for the wall (r = R) and the axis (r = 0).
        assert len(figure.axes[0].lines) == 3 + 2

    def test_empty_trajectory_list_raises(self, domain: mfp.Domain) -> None:
        with pytest.raises(ValueError, match="at least one particle path"):
            mfp.plot_trajectories([], domain)

    def test_non_channel_domain_raises(self) -> None:
        bifurcation = mfp.Domain(kind="bifurcation", length=4.0, radius=1.3, branch_angle=0.3)
        with pytest.raises(NotImplementedError, match="only supports domain.kind == 'channel'"):
            mfp.plot_trajectories([self._trajectory(3, 0.2)], bifurcation)


class TestPlotTrainingHistory:
    def test_adam_only_history_renders(self) -> None:
        history = mfp.TrainingHistory(adam=_loss_history(6), lbfgs=_loss_history(0))
        _assert_renders(mfp.plot_training_history(history))

    def test_adam_plus_lbfgs_history_renders_with_a_stage_boundary(self) -> None:
        history = mfp.TrainingHistory(adam=_loss_history(6), lbfgs=_loss_history(4))
        figure = mfp.plot_training_history(history)
        _assert_renders(figure)
        # One panel per component plus "total", plus the all-losses overlay.
        used_panels = [axes for axes in figure.axes if axes.axison]
        assert len(used_panels) == len(mfp.LOSS_COMPONENT_NAMES) + 2

    def test_linear_scale_option_renders(self) -> None:
        history = mfp.TrainingHistory(adam=_loss_history(4), lbfgs=_loss_history(0))
        figure = mfp.plot_training_history(history, log_scale=False)
        _assert_renders(figure)
        assert figure.axes[0].get_yscale() == "linear"

    def test_default_scale_is_logarithmic(self) -> None:
        history = mfp.TrainingHistory(adam=_loss_history(4), lbfgs=_loss_history(0))
        assert mfp.plot_training_history(history).axes[0].get_yscale() == "log"

    def test_empty_history_raises(self) -> None:
        history = mfp.TrainingHistory(adam=_loss_history(0), lbfgs=_loss_history(0))
        with pytest.raises(ValueError, match="both phases contain zero steps"):
            mfp.plot_training_history(history)


class TestProfilePlots:
    def test_velocity_profile_comparison_overlays_every_station_and_the_reference(self) -> None:
        figure = mfp.plot_velocity_profile_comparison(_profile_evaluation(n_stations=3))
        _assert_renders(figure)
        assert len(figure.axes[0].lines) == 3 + 1  # three predicted curves + the analytical one

    def test_profile_error_draws_one_curve_per_station_plus_the_zero_line(self) -> None:
        figure = mfp.plot_profile_error(_profile_evaluation(n_stations=2))
        _assert_renders(figure)
        assert len(figure.axes[0].lines) == 2 + 1

    def test_radial_leakage_draws_one_curve_per_station_plus_the_zero_line(self) -> None:
        figure = mfp.plot_radial_leakage(_profile_evaluation(n_stations=4))
        _assert_renders(figure)
        assert len(figure.axes[0].lines) == 4 + 1

    @pytest.mark.parametrize(
        "plot_function",
        [mfp.plot_velocity_profile_comparison, mfp.plot_profile_error, mfp.plot_radial_leakage],
    )
    def test_profile_plots_reject_an_evaluation_with_no_stations(self, plot_function) -> None:
        empty = ProfileEvaluation(
            z_stations=(), radial_grid=torch.linspace(0.0, 1.3, 4),
            predicted_uz=torch.zeros(0, 4), predicted_ur=torch.zeros(0, 4), analytical_uz=torch.zeros(4),
        )
        with pytest.raises(ValueError, match="at least one z station"):
            plot_function(empty)


class TestPlotFlowRateDeviation:
    def test_renders_the_deviation_curve_and_zero_line(self) -> None:
        axial_positions = torch.linspace(0.0, 4.0, 8)
        curve = torch.full((8,), 1.5) + 0.01 * torch.arange(8)
        figure = mfp.plot_flow_rate_deviation(axial_positions, curve, reference_flow_rate=1.5)
        _assert_renders(figure)
        axes = figure.axes[0]
        assert len(axes.lines) == 2
        assert np.allclose(axes.lines[0].get_ydata(), 0.01 * np.arange(8), atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must have the same shape"):
            mfp.plot_flow_rate_deviation(torch.zeros(5), torch.zeros(4), reference_flow_rate=1.0)

    @pytest.mark.parametrize("bad_reference", [0.0, -1.0])
    def test_non_positive_reference_raises(self, bad_reference: float) -> None:
        with pytest.raises(ValueError, match="reference_flow_rate must be strictly positive"):
            mfp.plot_flow_rate_deviation(torch.zeros(4), torch.ones(4), reference_flow_rate=bad_reference)


class TestPlotPressureGradientFit:
    def test_renders_samples_fit_and_analytical_line(self) -> None:
        axial_positions = np.linspace(0.0, 4.0, 10)
        predicted = -4.0 * (axial_positions - 4.0)
        figure = mfp.plot_pressure_gradient_fit(
            axial_positions, predicted, fitted_slope=-4.0, fitted_intercept=16.0,
            analytical_slope=-4.0, reference_pressure_at_outlet=0.0, domain_length=4.0,
        )
        _assert_renders(figure)
        assert len(figure.axes[0].lines) == 2  # fitted line + analytical line (the samples are a scatter)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must have the same shape"):
            mfp.plot_pressure_gradient_fit(
                np.zeros(5), np.zeros(4), fitted_slope=0.0, fitted_intercept=0.0,
                analytical_slope=0.0, reference_pressure_at_outlet=0.0, domain_length=1.0,
            )


class TestPlotResidualHeatmap:
    def test_renders_with_a_colorbar(self) -> None:
        radial_grid, axial_grid = torch.linspace(0.1, 1.3, 6), torch.linspace(0.0, 4.0, 9)
        figure = mfp.plot_residual_heatmap(radial_grid, axial_grid, torch.rand(6, 9), "momentum_z")
        _assert_renders(figure)
        assert len(figure.axes) == 2  # the heatmap axes and its colorbar axes
        assert "momentum_z" in figure.axes[0].get_title()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"residual_magnitude must have shape \(6, 9\)"):
            mfp.plot_residual_heatmap(torch.zeros(6), torch.zeros(9), torch.zeros(9, 6), "continuity")
