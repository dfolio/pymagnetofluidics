"""Unit tests for stream-function checkpointing, weight transfer, gates and the warm-started sweep."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from magnetofluidics_pinn.config import FluidConfig, TrainingConfig
from magnetofluidics_pinn.io_stream import (
    architecture_from_network, load_stream_checkpoint, make_provenance,
    save_stream_checkpoint, state_checksum,
)
from magnetofluidics_pinn.networks.mlp import build_mlp
from magnetofluidics_pinn.networks.stream_function import apply_stream_function_constraint
from magnetofluidics_pinn.networks.warm_start import (
    WarmStartError, infer_mlp_layout, is_transferable, rebuild_for_geometry,
    scaled_training_config, transfer_raw_weights,
)
from magnetofluidics_pinn.trajectory import mobility
from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory
from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.verification.warm_start_gates import (
    warm_start_cost_ratio, warm_start_equivalence_gate,
)

DOMAIN = Domain(kind="channel", length=8.0, radius=1.0, u_max=1.0)
CONFIG = TrainingConfig(device="cpu", n_epochs=100)


def make_network(hidden=(8, 8), z=4.0, a=0.25, u=0.0, amplitude=1.0, margin=8.0, dtype="float32", seed=0):
    torch.manual_seed(seed)
    config = TrainingConfig(device="cpu", dtype=dtype)
    raw = build_mlp(2, 2, hidden, training_config=config)
    return apply_stream_function_constraint(raw, DOMAIN, z, a, u, amplitude, margin)


def first_weight(network) -> float:
    return next(network.raw_network.parameters()).flatten()[0].item()


def history() -> StreamLossHistory:
    return StreamLossHistory(("total",), ((2.0,), (1.0,)), ((0.5,),), probe=(0.6,))


# --- layout and architecture -------------------------------------------------------------------

def test_layout_is_inferred():
    assert infer_mlp_layout(make_network((8, 6)).raw_network) == ((8, 6), "tanh")


def test_layout_rejects_wrong_io():
    bad = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    with pytest.raises(WarmStartError):
        infer_mlp_layout(bad)


def test_architecture_round_trip_rebuilds_same_geometry():
    network = make_network()
    rebuilt = architecture_from_network(network).build()
    for name in ("radius", "length", "particle_z", "particle_a", "particle_u", "amplitude", "support"):
        assert float(getattr(rebuilt, name)) == pytest.approx(float(getattr(network, name)), rel=1e-6)


def test_architecture_rejects_plain_module():
    with pytest.raises(TypeError):
        architecture_from_network(torch.nn.Linear(2, 2))


# --- checkpoints -------------------------------------------------------------------------------

def test_checksum_is_stable_and_sensitive():
    network = make_network()
    assert state_checksum(network) == state_checksum(copy.deepcopy(network))
    with torch.no_grad():
        next(network.raw_network.parameters()).add_(1e-3)
    assert state_checksum(network) != state_checksum(make_network())


def test_save_load_reproduces_outputs(tmp_path):
    network = make_network()
    path = tmp_path / "nested" / "ckpt.pt"
    save_stream_checkpoint(network, path, CONFIG, make_provenance(CONFIG, history(), library_version="test"), history())
    loaded, architecture, config, provenance, stored = load_stream_checkpoint(path, device="cpu")
    grid = torch.tensor([[0.3, 3.0], [0.6, 4.4], [0.9, 6.0]])
    assert torch.allclose(network(grid), loaded(grid))
    assert architecture == architecture_from_network(network)
    assert config == CONFIG and provenance.cold_start and provenance.final_probe_loss == 0.6
    assert stored == history()


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stream_checkpoint(tmp_path / "absent.pt")


def test_save_rejects_nan_weights(tmp_path):
    network = make_network()
    with torch.no_grad():
        next(network.raw_network.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError):
        save_stream_checkpoint(network, tmp_path / "x.pt", CONFIG, make_provenance(CONFIG, history(), library_version="t"))


def _tamper(path, mutate):
    payload = torch.load(path, weights_only=False)  # trusted: written by this test
    mutate(payload)
    torch.save(payload, path)


def test_load_detects_corruption_schema_and_missing_keys(tmp_path):
    network = make_network()
    provenance = make_provenance(CONFIG, history(), library_version="t")
    path = tmp_path / "c.pt"

    save_stream_checkpoint(network, path, CONFIG, provenance)
    _tamper(path, lambda p: p["state_dict"][next(iter(p["state_dict"]))].add_(1.0)
            if p["state_dict"][next(iter(p["state_dict"]))].is_floating_point() else None)
    with pytest.raises(RuntimeError, match="Checksum"):
        load_stream_checkpoint(path)

    save_stream_checkpoint(network, path, CONFIG, provenance)
    _tamper(path, lambda p: p.update(schema="stream-0"))
    with pytest.raises(RuntimeError, match="schema"):
        load_stream_checkpoint(path)

    save_stream_checkpoint(network, path, CONFIG, provenance)
    _tamper(path, lambda p: p.pop("checksum"))
    with pytest.raises(RuntimeError, match="missing"):
        load_stream_checkpoint(path)


def test_provenance_requires_parent_for_warm_runs():
    with pytest.raises(ValueError):
        make_provenance(CONFIG, history(), cold_start=False, library_version="t")


# --- weight transfer ---------------------------------------------------------------------------

def test_transfer_copies_weights_keeps_buffers_and_purity():
    source, target = make_network(seed=1, z=4.0), make_network(seed=2, z=3.0)
    source_before, target_before = state_checksum(source), state_checksum(target)
    result = transfer_raw_weights(source, target)
    assert first_weight(result) == first_weight(source)
    assert float(result.particle_z) == 3.0
    assert state_checksum(source) == source_before and state_checksum(target) == target_before


def test_transfer_rejects_mismatches_and_nan():
    with pytest.raises(WarmStartError):
        transfer_raw_weights(make_network((8, 8)), make_network((8, 6)))
    ok, reason = is_transferable(make_network((8, 8)), torch.nn.Linear(2, 2))
    assert not ok and "raw_network" in reason
    source = make_network()
    with torch.no_grad():
        next(source.raw_network.parameters()).fill_(float("inf"))
    with pytest.raises(WarmStartError):
        transfer_raw_weights(source, make_network(seed=3))


def test_transfer_casts_dtype():
    result = transfer_raw_weights(make_network(dtype="float64"), make_network(dtype="float32", seed=5))
    assert next(result.raw_network.parameters()).dtype == torch.float32


def test_rebuild_for_geometry_moves_sphere():
    source = make_network()
    moved = rebuild_for_geometry(source, DOMAIN, 3.0, 0.25, 0.0, 1.0, 8.0, CONFIG)
    assert float(moved.particle_z) == 3.0 and first_weight(moved) == first_weight(source)


def test_scaled_config_and_validation():
    scaled = scaled_training_config(CONFIG, 0.1, 0.25)
    assert scaled.n_epochs == 25 and scaled.learning_rate == pytest.approx(CONFIG.learning_rate * 0.1)
    assert CONFIG.n_epochs == 100
    for kwargs in ({"learning_rate_scale": 0.0}, {"epoch_fraction": 0.0}, {"epoch_fraction": 1.5},
                   {"learning_rate_scale": float("nan")}):
        with pytest.raises(ValueError):
            scaled_training_config(CONFIG, **kwargs)


# --- gates -------------------------------------------------------------------------------------

def test_equivalence_gate_pass_fail_and_errors():
    assert warm_start_equivalence_gate(-8.0, -8.1, 0.05).passed
    assert not warm_start_equivalence_gate(-8.0, -9.0, 0.05).passed
    with pytest.raises(ValueError):
        warm_start_equivalence_gate(0.0, 1.0, 0.05)
    with pytest.raises(ValueError):
        warm_start_equivalence_gate(1.0, float("nan"), 0.05)


def test_cost_ratio():
    cold = StreamLossHistory(("total",), ((1.0,),) * 10, ((1.0,),) * 10)
    warm = StreamLossHistory(("total",), ((1.0,),) * 3, ((1.0,),) * 2)
    assert warm_start_cost_ratio(cold, warm) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        warm_start_cost_ratio(StreamLossHistory(("total",), (), ()), warm)


# --- warm-started sweep (trainer and traction stubbed) -----------------------------------------

@pytest.fixture
def stubbed_solver(monkeypatch):
    calls: list[tuple[int, float]] = []

    def fake_train(network, domain, fluid_config, training_config, *args, **kwargs):
        calls.append((training_config.n_epochs, first_weight(network)))
        trained = copy.deepcopy(network)
        with torch.no_grad():
            next(trained.raw_network.parameters()).add_(1.0)
        return trained, StreamLossHistory(("total",), ((1.0,),), ())

    monkeypatch.setattr(mobility, "train_stream_function_flow", fake_train)
    monkeypatch.setattr(mobility, "sphere_traction_force", lambda *a, **k: -1.0)
    return calls


def test_sweep_warm_start_carries_each_problem_separately(stubbed_solver):
    table = mobility.resistance_position_sweep(
        DOMAIN, FluidConfig(), 0.25, (3.0, 3.5), CONFIG, warm_start="previous_position",
        refinement_config=scaled_training_config(CONFIG), hidden_layers=(8, 8))
    epochs = [c[0] for c in stubbed_solver]
    assert epochs == [100, 100, 25, 25]
    assert stubbed_solver[2][1] == pytest.approx(stubbed_solver[0][1] + 1.0)  # ambient -> ambient
    assert stubbed_solver[3][1] == pytest.approx(stubbed_solver[1][1] + 1.0)  # translation -> translation
    assert list(table["warm_started"]) == [False, True]


def test_sweep_default_is_cold(stubbed_solver):
    mobility.resistance_position_sweep(DOMAIN, FluidConfig(), 0.25, (3.0, 3.5), CONFIG, hidden_layers=(8, 8))
    assert [c[0] for c in stubbed_solver] == [100] * 4


def test_sweep_rejects_bad_mode_and_short_input(stubbed_solver):
    with pytest.raises(ValueError):
        mobility.resistance_position_sweep(DOMAIN, FluidConfig(), 0.25, (3.0, 3.5), CONFIG, warm_start="x")
    with pytest.raises(ValueError):
        mobility.resistance_position_sweep(DOMAIN, FluidConfig(), 0.25, (3.0,), CONFIG)
