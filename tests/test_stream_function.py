import pytest
import torch
from torch import nn

from magnetofluidics_pinn.config import FluidConfig, TrainingConfig
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import apply_stream_function_constraint, bump_function
from magnetofluidics_pinn.physics.fluid_residuals import stokes_residual
from magnetofluidics_pinn.sampling.exterior import sample_exterior_points, sample_inlet_outlet_points
from magnetofluidics_pinn.training.stream_trainer import train_stream_function_flow
from magnetofluidics_pinn.types import Domain

DOM = Domain(kind="channel", length=8.0, radius=1.0, u_max=1.0)
A, ZP, UP = 0.25, 4.0, 0.7
CFG = TrainingConfig(device="cpu", dtype="float64", n_epochs=2, n_interior_points=64,
                     n_boundary_points=8, use_lbfgs_refinement=False, residual_form="r_weighted")


def _grid(n: int = 400, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = 0.02 + 0.97 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    z = 8.0 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    return torch.cat([r, z], dim=1)


def _network(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    raw = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 2)).double()
    return apply_stream_function_constraint(raw, DOM, 4.0, 0.25, 0.3, 1.0, 8.0)


def _points(n: int, r_min: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = r_min + (0.99 - r_min) * torch.rand(n, 1, generator=g, dtype=torch.float64)
    z = 8.0 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    return torch.cat([r, z], dim=1)


def make(amplitude=1.0):
    raw = build_mlp(2, 2, (16, 16), training_config=CFG).double()
    return apply_stream_function_constraint(raw, DOM, ZP, A, UP, amplitude)


def test_bump_function_properties():
    q = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    out = bump_function(q)
    assert out[0] == 1.0 and out[2] == 0.0 and out[3] == 0.0 and 0.0 < out[1] < 1.0


def test_sphere_no_slip_is_exact_for_random_weights():
    net = make()
    theta = torch.linspace(0.0, torch.pi, 40, dtype=torch.float64)
    pts = torch.stack([A * torch.sin(theta), ZP + A * torch.cos(theta)], dim=1)
    out = net(pts)
    assert out[:, 0].abs().max() < 1e-10
    assert (out[:, 1] - UP).abs().max() < 2e-8


def test_wall_and_axis_are_exact():
    net = make()
    z = torch.linspace(0.1, 7.9, 30, dtype=torch.float64)
    wall = net(torch.stack([torch.full_like(z, 1.0), z], dim=1))
    axis = net(torch.stack([torch.zeros_like(z), z], dim=1))
    assert wall[:, :2].abs().max() < 1e-10
    assert axis[:, 0].abs().max() < 1e-12


def test_continuity_holds_identically():
    net = make()
    pts = torch.tensor([[0.4, 4.4], [0.7, 3.0], [0.3, 6.0]], dtype=torch.float64).requires_grad_(True)
    res = stokes_residual(net, pts, FluidConfig(), residual_form="standard")
    assert res[:, 2].abs().max().item() < 1e-9


def test_no_grad_call_returns_detached_output():
    net = make()
    with torch.no_grad():
        out = net(torch.tensor([[0.5, 2.0]], dtype=torch.float64))
    assert not out.requires_grad


def test_invalid_geometry_and_network_shape():
    raw = build_mlp(2, 2, (8,), training_config=CFG).double()
    with pytest.raises(ValueError):
        apply_stream_function_constraint(raw, DOM, 0.1, A)  # sphere overlaps the inlet
    with pytest.raises(ValueError):
        apply_stream_function_constraint(raw, DOM, ZP, 1.5)  # sphere wider than the tube
    bad = build_mlp(2, 3, (8,), training_config=CFG).double()
    with pytest.raises(ValueError):
        apply_stream_function_constraint(bad, DOM, ZP, A)


def test_exterior_sampling_properties():
    pts = sample_exterior_points(DOM, ZP, A, 500, 0, dtype=torch.float64, device="cpu")
    assert pts.shape == (500, 2)
    assert ((pts[:, 1] - ZP).square() + pts[:, 0].square() >= A ** 2).all()
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= 1.0).all() and (pts[:, 1] > 0).all() and (pts[:, 1] < 8.0).all()
    again = sample_exterior_points(DOM, ZP, A, 500, 0, dtype=torch.float64, device="cpu")
    assert torch.equal(pts, again)
    with pytest.raises(ValueError):
        sample_exterior_points(DOM, ZP, A, 0, 0)
    with pytest.raises(ValueError):
        sample_exterior_points(DOM, ZP, 1.5, 10, 0)
    inlet, outlet = sample_inlet_outlet_points(DOM, 9, 0, device="cpu", dtype=torch.float64)
    assert inlet.shape[0] + outlet.shape[0] == 9 and (inlet[:, 1] == 0).all() and (outlet[:, 1] == 8.0).all()


def test_trainer_returns_copy_and_history():
    net = make()
    before = [p.clone() for p in net.parameters()]
    trained, hist = train_stream_function_flow(net, DOM, FluidConfig(), CFG, ZP, A, 1.0)
    assert all(torch.equal(b, p) for b, p in zip(before, net.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(before, trained.parameters()))
    assert len(hist.adam) == 2 and hist.lbfgs == () and set(hist.final()) == set(hist.names)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_divergence_free_for_random_weights(seed):
    net = _network(seed)
    pts = _grid().requires_grad_(True)
    out = net(pts)
    d_ur = torch.autograd.grad(out[:, 0].sum(), pts, retain_graph=True)[0][:, 0:1]
    d_uz = torch.autograd.grad(out[:, 1].sum(), pts)[0][:, 1:2]
    div = d_ur + out[:, 0:1] / pts[:, 0:1] + d_uz
    assert div.abs().max().item() < 1e-8 * max(1.0, out[:, :2].abs().max().item())


def test_wall_axis_and_sphere_conditions():
    net = _network(3)
    z = torch.linspace(0.0, 8.0, 50, dtype=torch.float64)
    wall = net(torch.stack([torch.full_like(z, 1.0), z], 1))
    axis = net(torch.stack([torch.zeros_like(z), z], 1))
    theta = torch.linspace(0.0, torch.pi, 91, dtype=torch.float64)
    sphere = net(torch.stack([A * torch.sin(theta), ZP + A * torch.cos(theta)], 1))
    assert wall[:, :2].abs().max() < 1e-10
    assert axis[:, 0].abs().max() < 1e-12
    assert sphere[:, 0].abs().max() < 1e-9
    assert (sphere[:, 1] - 0.3).abs().max() < 2e-8


def test_flux_is_constant_outside_taper():
    net = _network(4)
    r = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    for z in (0.3, 7.5):
        out = net(torch.stack([r, torch.full_like(r, z)], 1))
        q = torch.trapezoid(out[:, 1] * 2.0 * torch.pi * r, r)
        assert abs(q.item() - torch.pi / 2.0) < 1e-5


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        _network(5)(torch.zeros(3, 3, dtype=torch.float64))


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_standard_residual_off_axis(seed):
    net, pts = _network(seed), _points(300, 0.05, seed)
    native = net.stokes_residual_sz(pts)
    standard = stokes_residual(net, pts.clone().requires_grad_(True), FluidConfig(), residual_form="standard")
    scale = standard.abs().max().item()
    assert (native[:, :2] - standard[:, :2]).abs().max().item() < 1e-8 * max(scale, 1.0)


def test_divergence_column_is_zero():
    net = _network(2)
    assert net.stokes_residual_sz(_points(300, 0.01))[:, 2].abs().max().item() < 1e-9


def test_finite_and_continuous_on_the_axis():
    net = _network(3)
    z = torch.linspace(0.2, 7.8, 40, dtype=torch.float64).unsqueeze(1)
    axis = net.stokes_residual_sz(torch.cat([torch.zeros_like(z), z], 1))
    near = net.stokes_residual_sz(torch.cat([torch.full_like(z, 1e-5), z], 1))
    assert torch.isfinite(axis).all()
    assert (axis[:, :2] - near[:, :2]).abs().max().item() < 1e-3 * max(1.0, axis.abs().max().item())


def test_gradients_reach_parameters_and_input_validation():
    net = _network(4)
    net.stokes_residual_sz(_points(50, 0.02)).square().mean().backward()
    assert all(p.grad is not None for p in net.raw_network.parameters())
    with pytest.raises(ValueError):
        net.stokes_residual_sz(torch.zeros(3, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        net.stokes_residual_sz(torch.tensor([[-0.1, 1.0]], dtype=torch.float64))
