r"""Quadrature rules for surface integrals over a sphere on the symmetry axis.

For an axisymmetric integrand the surface element is $\mathrm dA = 2\pi a^2 \sin\theta\,\mathrm d\theta$, so

$$ F = \oint_S t_z\,\mathrm dA = 2\pi a^2 \int_0^\pi t_z(\theta)\sin\theta\,\mathrm d\theta
     = 2\pi a^2 \int_{-1}^{1} t_z(x)\,\mathrm dx, \qquad x = \cos\theta . $$

The integrand is smooth in $x$, so Gauss-Legendre converges geometrically and needs a few dozen nodes, where the
trapezoidal rule in $\theta$ converges like $h^2$ because the odd derivatives do not vanish at the poles. Every rule is
returned as a pair `(theta, weights)` with $F \approx 2\pi a^2 \sum_i w_i\,t_z(\theta_i)$, so the three rules are interchangeable.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

QuadratureRule = Literal["gauss_legendre", "simpson", "trapezoid"]
_RULES: tuple[str, ...] = ("gauss_legendre", "simpson", "trapezoid")


def sphere_quadrature(
    n_nodes: int,
    rule: QuadratureRule = "gauss_legendre",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return polar-angle nodes and weights for $\int_0^\pi f(\theta)\sin\theta\,\mathrm d\theta$.

    Args:
    - `n_nodes`: Number of nodes; at least 1 for Gauss-Legendre, at least 2 for the trapezoidal rule and at least 3 for Simpson.
    - `rule`: `"gauss_legendre"` (default; nodes are interior, so $r = a\sin\theta > 0$), `"simpson"` or `"trapezoid"`
      (both on the uniform grid $\theta_i = i\pi/(n - 1)$ including the poles). For Simpson with an even number of nodes,
      the last interval is integrated with the trapezoidal rule.
    - `device`: Device of the returned tensors.
    - `dtype`: Dtype of the returned tensors.

    Returns:
    - The tuple `(theta, weights)` of tensors `(n_nodes,)`. The weights already contain the factor $\sin\theta$ and the
      measure, so $\int_0^\pi f\sin\theta\,\mathrm d\theta \approx \sum_i w_i f(\theta_i)$.

    Raises:
    - `ValueError`: If `rule` is unknown or `n_nodes` is too small for the rule.
    """
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES!r}; got {rule!r}.")
    minimum = {"gauss_legendre": 1, "trapezoid": 2, "simpson": 3}[rule]
    if n_nodes < minimum:
        raise ValueError(
            f"rule={rule!r} needs at least {minimum} nodes; got {n_nodes!r}.")
    if rule == "gauss_legendre":
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        theta_np, weights_np = np.arccos(x), w
    else:
        theta_np = np.linspace(0.0, math.pi, n_nodes)
        step = math.pi / (n_nodes - 1)
        base = np.full(n_nodes, step)
        base[[0, -1]] = 0.5 * step
        if rule == "simpson":
            base = np.zeros(n_nodes)
            n_even = n_nodes if (
                                        n_nodes - 1) % 2 == 0 else n_nodes - 1  # nodes covered by composite Simpson
            pattern = np.ones(n_even)
            pattern[1:-1:2], pattern[2:-1:2] = 4.0, 2.0
            base[:n_even] = pattern * step / 3.0
            if n_even != n_nodes:  # trapezoid on the last interval
                base[n_even - 1] += 0.5 * step
                base[n_even] += 0.5 * step
        weights_np = base * np.sin(theta_np)
    kwargs = {"device": device, "dtype": dtype}
    return torch.as_tensor(theta_np, **kwargs), torch.as_tensor(weights_np, **kwargs)
