r"""Physical acceptance gates for two-way particle-flow solutions.

Every gate returns a `VerificationCheck` so it plugs into
`render_verification_table`. Thresholds must come from the reference
uncertainty (see `reference_tolerance`), not from arbitrary constants.

For an on-axis sphere in a tube, far upstream and downstream of the sphere the flow is
fully developed Poiseuille flow, and the momentum balance on the fluid between two far
sections reduces to
$$F_z = \pi R^2\,\big[p_\text{up}(z_p) - p_\text{down}(z_p)\big],$$
where $p_\text{up/down}$ are the far-field axis pressures extrapolated to $z_p$ along
the Poiseuille slope $-4U/R^2$.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from magnetofluidics_pinn.device_utils import resolve_module_device, resolve_module_dtype
from magnetofluidics_pinn.data_types import Domain
from magnetofluidics_pinn.verification.force_checks import global_momentum_balance, sphere_traction_force
from magnetofluidics_pinn.verification.reporting import VerificationCheck, evaluate_check


def divergence_probe(
    network: nn.Module, domain: Domain, n_points: int = 2000, random_seed: int = 0,
    axis_clearance_fraction: float = 1.0e-2,
) -> float:
    r"""Measure the relative divergence $\max\abs{\div\mathbf u} / (\max\abs{\mathbf u}/R)$.

    Args:
    - `network`: Flow network mapping `(r, z)` to `(u_r, u_z, p)`.
    - `domain`: Nondimensional channel.
    - `n_points`: Number of random points, strictly positive.
    - `random_seed`: Seed of the device-local generator.
    - `axis_clearance_fraction`: Points keep $r \ge$ this fraction of $R$ (the $u_r/r$ term).

    Returns:
    - The relative divergence; round-off level for a network that conserves mass by construction.

    Raises:
    - `ValueError`: If `n_points <= 0` or the clearance is outside $(0, 1)$.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be strictly positive; got {n_points!r}.")
    if not 0.0 < axis_clearance_fraction < 1.0:
        raise ValueError("axis_clearance_fraction must lie in (0, 1).")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    generator = torch.Generator(device=device).manual_seed(random_seed)
    unit = torch.rand(n_points, 2, generator=generator, device=device, dtype=dtype)
    radial = domain.radius * (axis_clearance_fraction + (1.0 - axis_clearance_fraction) * unit[:, 0:1])
    axial = domain.length * unit[:, 1:2]
    with torch.enable_grad():
        coords = torch.cat([radial, axial], dim=1).requires_grad_(True)
        out = network(coords)
        u_r, u_z = out[:, 0:1], out[:, 1:2]
        d_ur_dr = torch.autograd.grad(u_r, coords, torch.ones_like(u_r), retain_graph=True)[0][:, 0:1]
        d_uz_dz = torch.autograd.grad(u_z, coords, torch.ones_like(u_z))[0][:, 1:2]
        divergence = (d_ur_dr + u_r / coords[:, 0:1] + d_uz_dz).detach()
    scale = max(out[:, :2].detach().abs().max().item(), 1.0e-30) / domain.radius
    return divergence.abs().max().item() / scale


def axis_pressure_jump(
    network: nn.Module, domain: Domain, particle_z: float, ambient_peak_velocity: float,
    offset: float = 2.0, window: float = 1.5, n_points: int = 64,
) -> float:
    r"""Extra pressure drop caused by the sphere, from far-field axis fits.

    Fits the axis pressure on two windows, upstream $[z_p-o-w, z_p-o]$ and downstream
    $[z_p+o, z_p+o+w]$, with the known Poiseuille slope $g=-4U/R^2$ (zero for fluid at rest),
    and returns the difference of the extrapolated values at $z_p$.

    Args:
    - `network`: Flow network.
    - `domain`: Nondimensional channel.
    - `particle_z`: Sphere center.
    - `ambient_peak_velocity`: Undisturbed centerline velocity $U$ (may be zero).
    - `offset`, `window`: Window geometry, both strictly positive.
    - `n_points`: Samples per window, at least 2.

    Returns:
    - $p_\text{up}(z_p) - p_\text{down}(z_p)$.

    Raises:
    - `ValueError`: If an argument is invalid or the windows leave $[0, L]$.
    """
    if offset <= 0.0 or window <= 0.0 or n_points < 2:
        raise ValueError("offset and window must be positive and n_points at least 2.")
    if particle_z - offset - window < -1.0e-9 or particle_z + offset + window > domain.length + 1.0e-9:
        raise ValueError("The far-field windows must lie inside [0, L].")
    device, dtype = resolve_module_device(network), resolve_module_dtype(network)
    slope = -4.0 * ambient_peak_velocity / domain.radius**2

    def intercept(z_start: float, z_end: float) -> torch.Tensor:
        z = torch.linspace(z_start, z_end, n_points, device=device, dtype=dtype)
        coords = torch.stack([torch.zeros_like(z), z], dim=1)
        with torch.no_grad():
            pressure = network(coords)[:, 2]
        return (pressure - slope * z).mean()

    up = intercept(particle_z - offset - window, particle_z - offset)
    down = intercept(particle_z + offset, particle_z + offset + window)
    return (up - down).item()


def pressure_drop_gate(
    network: nn.Module, domain: Domain, particle_z: float, particle_radius: float,
    ambient_peak_velocity: float, tolerance: float, offset: float = 2.0, window: float = 1.5,
) -> VerificationCheck:
    r"""Check $\abs{\pi R^2 \Delta p_\text{extra} - F_z}/\abs{F_z} \le$ `tolerance`.

    Args:
    - `network`, `domain`, `particle_z`, `particle_radius`: Flow and sphere.
    - `ambient_peak_velocity`: Undisturbed centerline velocity.
    - `tolerance`: Relative bound, finite and positive.
    - `offset`, `window`: Far-field windows, see `axis_pressure_jump`.

    Returns:
    - A `VerificationCheck`.
    """
    force = sphere_traction_force(network, particle_z, particle_radius, 361)
    jump = axis_pressure_jump(network, domain, particle_z, ambient_peak_velocity, offset, window)
    metric = abs(math.pi * domain.radius**2 * jump - force) / max(abs(force), 1.0e-12)
    return evaluate_check(
        "Extra pressure drop vs traction force",
        "|pi R^2 dp_extra - F_z| / |F_z| from far-field axis pressure fits",
        metric, tolerance, "<=",
    )


def momentum_balance_gate(
    network: nn.Module, domain: Domain, particle_z: float, particle_radius: float,
    tolerance: float, section_offset: float = 2.0,
) -> VerificationCheck:
    """Check the surface-traction force against the cross-section momentum force.

    Args:
    - `network`, `domain`, `particle_z`, `particle_radius`: Flow and sphere.
    - `tolerance`: Relative bound.
    - `section_offset`: Distance of the two sections from `particle_z`.

    Returns:
    - A `VerificationCheck` on `relative_difference`.
    """
    balance = global_momentum_balance(
        network, particle_z, particle_radius, domain.radius, section_offset)
    return evaluate_check(
        "Global momentum balance", "|F_surface - F_momentum| / max(|F|)",
        balance["relative_difference"], tolerance, "<=",
    )


def free_velocity_bounds_check(free_velocity: float, peak_velocity: float) -> tuple[VerificationCheck, VerificationCheck]:
    r"""Check $0 < U_\text{free} < U_\text{peak}$ for a passive on-axis sphere.

    Args:
    - `free_velocity`: Passive sphere velocity.
    - `peak_velocity`: Undisturbed centerline velocity, strictly positive.

    Returns:
    - The pair of lower- and upper-bound checks.

    Raises:
    - `ValueError`: If `peak_velocity <= 0`.
    """
    if peak_velocity <= 0.0:
        raise ValueError("peak_velocity must be strictly positive.")
    return (
        evaluate_check("Passive velocity > 0", "U_free positive", free_velocity, 0.0, ">"),
        evaluate_check("Passive velocity < peak", "U_free below centerline velocity",
                       free_velocity, peak_velocity, "<"),
    )


def generalization_gate(train_loss: float, probe_loss: float, max_ratio: float = 5.0) -> VerificationCheck:
    """Check that the held-out probe loss stays within `max_ratio` of the training loss.

    Args:
    - `train_loss`: Final loss on the optimized batch.
    - `probe_loss`: Loss on an independent batch.
    - `max_ratio`: Largest accepted ratio, strictly positive.

    Returns:
    - A `VerificationCheck`.
    """
    return evaluate_check(
        "Generalization gap", "probe loss / train loss",
        probe_loss / max(train_loss, 1.0e-30), max_ratio, "<=",
    )


def reference_tolerance(relative_uncertainty: float, reynolds_radius: float) -> float:
    r"""Tolerance for a comparison with a creeping-flow reference.

    The result is $\max(\varepsilon_\text{ref},\ 2\cdot\tfrac38 Re_a)$: the stated reference
    uncertainty, or twice the Oseen estimate of the inertial error.

    Args:
    - `relative_uncertainty`: Reference uncertainty (for example 0.05), non-negative.
    - `reynolds_radius`: Radius-based Reynolds number $Re_a \ge 0$.

    Returns:
    - The tolerance.

    Raises:
    - `ValueError`: If an argument is negative or not finite.
    """
    if not all(math.isfinite(v) and v >= 0.0 for v in (relative_uncertainty, reynolds_radius)):
        raise ValueError("Both arguments must be finite and non-negative.")
    return max(relative_uncertainty, 2.0 * 0.375 * reynolds_radius)