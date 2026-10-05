import pytest
import torch

from magnetofluidics_pinn.trajectory.mobility import (
    ResistancePair, advance_particle_resistance, force_balanced_velocity,
)

PAIR = ResistancePair(ambient_force=8.93, translation_force=-9.33)


def test_force_balanced_velocity_is_linear_and_monotone():
    passive = force_balanced_velocity(PAIR, 0.0)
    assert passive == pytest.approx(8.93 / 9.33)
    assert force_balanced_velocity(PAIR, -0.3) < passive < force_balanced_velocity(PAIR, 0.3)
    assert force_balanced_velocity(PAIR, 0.3) - passive == pytest.approx(0.3 / 9.33)
    assert PAIR.mobility == pytest.approx(1.0 / 9.33)


def test_drag_balance_closes_exactly():
    applied = -0.3
    u = force_balanced_velocity(PAIR, applied)
    assert applied + PAIR.ambient_force + u * PAIR.translation_force == pytest.approx(0.0, abs=1e-12)


def test_invalid_pair_and_inputs():
    with pytest.raises(ValueError):
        ResistancePair(1.0, 0.5)
    with pytest.raises(ValueError):
        ResistancePair(float("nan"), -1.0)
    with pytest.raises(ValueError):
        force_balanced_velocity(PAIR, float("inf"))


def test_constant_force_trajectory_is_linear():
    states = advance_particle_resistance(PAIR, lambda z: 0.0, 2.0, 5, 0.1, (0.5, 7.5))
    assert len(states) == 6
    u = PAIR.free_velocity
    assert states[-1].position[1].item() == pytest.approx(2.0 + u * 0.5)
    assert states[-1].time == pytest.approx(0.5)
    assert torch.equal(states[0].position, torch.tensor([0.0, 2.0], dtype=torch.float64))


def test_trajectory_validation_and_range_guard():
    with pytest.raises(ValueError):
        advance_particle_resistance(PAIR, lambda z: 0.0, 2.0, 0, 0.1, (0.5, 7.5))
    with pytest.raises(ValueError):
        advance_particle_resistance(PAIR, lambda z: 0.0, 2.0, 3, -0.1, (0.5, 7.5))
    with pytest.raises(ValueError):
        advance_particle_resistance(PAIR, lambda z: 0.0, 2.0, 3, 0.1, (7.5, 0.5))
    with pytest.raises(RuntimeError):
        advance_particle_resistance(PAIR, lambda z: 0.0, 7.0, 20, 0.5, (0.5, 7.5))