"""Unit tests for `magnetofluidics_pinn.physics.conservation`.

Covers both the nominal path (quadrature convergence against the
closed-form Hagen-Poiseuille flow rate, differentiability through a real
network's parameters) and the edge cases each function's docstring commits
to raising on.
"""

from __future__ import annotations

import math

import pytest
import torch

import magnetofluidics_pinn as mfp

from .conftest import make_poiseuille_network


class TestAxialFlowRate:
    """Tests for `physics.conservation.axial_flow_rate`."""

    def test_matches_closed_form_for_exact_poiseuille_field(self) -> None:
        """Quadrature over an exact analytic profile should converge to Q_ref."""
        radius, peak_velocity = 1.3, 2.0
        network = make_poiseuille_network(radius, peak_velocity)
        reference = mfp.poiseuille_reference_flow_rate(radius, peak_velocity)

        axial_positions = torch.linspace(0.0, 5.0, 7)
        flow_rate = mfp.axial_flow_rate(network, radius, axial_positions, n_radial_quadrature_points=512)

        # A quadratic integrand is not exactly integrated by the trapezoidal
        # rule; at this resolution the residual quadrature error must still
        # be small relative to the flow rate itself.
        assert torch.allclose(flow_rate, torch.full_like(flow_rate, reference), atol=2.0e-4)

    def test_quadrature_error_shrinks_with_resolution(self) -> None:
        """Trapezoidal quadrature error should decrease as nodes increase (O(1/n^2))."""
        radius, peak_velocity = 1.0, 1.0
        network = make_poiseuille_network(radius, peak_velocity)
        reference = mfp.poiseuille_reference_flow_rate(radius, peak_velocity)
        axial_positions = torch.linspace(0.0, 1.0, 3)

        errors = []
        for n_points in (16, 64, 256):
            flow_rate = mfp.axial_flow_rate(network, radius, axial_positions, n_radial_quadrature_points=n_points)
            errors.append((flow_rate - reference).abs().max().item())

        assert errors[0] > errors[1] > errors[2]
        # Quadrupling the resolution should shrink the error by roughly 16x
        # (allowing generous slack for a coarse, non-asymptotic first step).
        assert errors[0] / errors[1] > 4.0
        assert errors[1] / errors[2] > 4.0

    def test_output_shape_matches_axial_positions(self) -> None:
        network = make_poiseuille_network(1.0, 1.0)
        axial_positions = torch.linspace(0.0, 2.0, 11)
        flow_rate = mfp.axial_flow_rate(network, 1.0, axial_positions, n_radial_quadrature_points=8)
        assert flow_rate.shape == (11,)

    def test_differentiable_through_network_parameters(self) -> None:
        """The flow rate must be usable inside a training loss (real gradients, no NaNs)."""
        # FIXED: build_mlp now takes an explicit training_config instead of device=.
        training_config = mfp.TrainingConfig(device="cpu")
        network = mfp.build_mlp(2, 3, (8, 8), training_config=training_config)
        axial_positions = torch.linspace(0.0, 1.0, 5)
        loss = torch.mean((mfp.axial_flow_rate(network, 1.0, axial_positions, 32) - 1.0) ** 2)
        loss.backward()

        gradients = [p.grad for p in network.parameters()]
        assert all(g is not None for g in gradients)
        assert any(g.norm().item() > 0.0 for g in gradients)
        assert all(torch.isfinite(g).all() for g in gradients)

    def test_raises_on_non_positive_radius(self) -> None:
        network = make_poiseuille_network(1.0, 1.0)
        for bad_radius in (0.0, -1.0):
            with pytest.raises(ValueError, match="radius must be strictly positive"):
                mfp.axial_flow_rate(network, bad_radius, torch.linspace(0.0, 1.0, 3), 8)

    def test_raises_on_non_1d_axial_positions(self) -> None:
        network = make_poiseuille_network(1.0, 1.0)
        with pytest.raises(ValueError, match="1-D tensor"):
            mfp.axial_flow_rate(network, 1.0, torch.zeros(3, 2), 8)

    def test_raises_on_too_few_quadrature_points(self) -> None:
        network = make_poiseuille_network(1.0, 1.0)
        with pytest.raises(ValueError, match="at least 2"):
            mfp.axial_flow_rate(network, 1.0, torch.linspace(0.0, 1.0, 3), 1)


class TestPoiseuilleReferenceFlowRate:
    """Tests for `physics.conservation.poiseuille_reference_flow_rate`."""

    def test_matches_closed_form(self) -> None:
        radius, peak_velocity = 2.0, 3.0
        expected = math.pi * radius**2 * peak_velocity / 2.0
        assert mfp.poiseuille_reference_flow_rate(radius, peak_velocity) == pytest.approx(expected)

    def test_scales_quadratically_with_radius(self) -> None:
        base = mfp.poiseuille_reference_flow_rate(1.0, 1.0)
        doubled = mfp.poiseuille_reference_flow_rate(2.0, 1.0)
        assert doubled == pytest.approx(4.0 * base)

    def test_scales_linearly_with_peak_velocity(self) -> None:
        base = mfp.poiseuille_reference_flow_rate(1.0, 1.0)
        doubled = mfp.poiseuille_reference_flow_rate(1.0, 2.0)
        assert doubled == pytest.approx(2.0 * base)

    @pytest.mark.parametrize("bad_radius", [0.0, -1.0])
    def test_raises_on_non_positive_radius(self, bad_radius: float) -> None:
        with pytest.raises(ValueError, match="radius must be strictly positive"):
            mfp.poiseuille_reference_flow_rate(bad_radius, 1.0)

    @pytest.mark.parametrize("bad_peak_velocity", [0.0, -1.0])
    def test_raises_on_non_positive_peak_velocity(self, bad_peak_velocity: float) -> None:
        with pytest.raises(ValueError, match="peak_velocity must be strictly positive"):
            mfp.poiseuille_reference_flow_rate(1.0, bad_peak_velocity)
