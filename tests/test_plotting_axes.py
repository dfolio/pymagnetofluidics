import numpy as np
import pytest

from magnetofluidics_pinn.visualization.plotting import _uniform_plot_axis


def test_uniform_plot_axis_is_bit_exact_uniform():
    # This is the actual property streamplot demands; allclose is not enough to prove it,
    # since allclose is exactly the kind of near-enough check that let the GPU-origin bug
    # through in the first place.
    axis = _uniform_plot_axis(8.0, 300)
    diffs = np.diff(axis)
    assert np.all(diffs == diffs[0])
    assert axis[0] == 0.0
    assert axis[-1] == 8.0
    assert axis.shape == (300,)


def test_uniform_plot_axis_rejects_too_few_points():
    with pytest.raises(ValueError):
        _uniform_plot_axis(1.0, 1)


@pytest.mark.parametrize("extent,n", [(1.0, 2), (7.5e-4, 500), (100.0, 2001)])
def test_uniform_plot_axis_uniform_across_scales(extent, n):
    axis = _uniform_plot_axis(extent, n)
    diffs = np.diff(axis)
    assert np.all(diffs == diffs[0])
    