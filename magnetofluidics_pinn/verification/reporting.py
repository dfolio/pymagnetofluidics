r"""Verification-checklist scaffolding shared across `magnetofluidics_pinn` notebooks.

`magnetofluidics_pinn.verification.reporting` defines the small, reusable
vocabulary a verification notebook uses to turn a scattered set of computed
metrics into a single, auditable pass/fail table:
[`VerificationCheck`][magnetofluidics_pinn.verification.reporting.VerificationCheck]
records one named claim (a metric, a threshold, and a comparison), and
[`VerificationThresholds`][magnetofluidics_pinn.verification.reporting.VerificationThresholds]
collects every calibratable tolerance a notebook needs into one place,
rather than scattering bare numeric literals through the analysis cells.
Nothing here is specific to Phase 1: the same scaffolding is intended to be
reused, unchanged, by the Phase 2 and Phase 3 verification notebooks the
project roadmap anticipates.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, fields
from typing import Callable, Literal, NamedTuple

import pandas as pd
import torch

# Maps each supported comparison operator to its `operator` module callable,
# keeping `evaluate_check` a simple lookup instead of a branching if/elif
# chain - mirroring `networks.mlp._ACTIVATION_LAYERS`'s dispatch idiom.
_COMPARISON_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
}


class VerificationCheck(NamedTuple):
    """One named, auditable verification claim.

    An immutable record pairing a measured value against a threshold under
    a stated comparison, together with a human-readable name and
    description - the atomic unit
    [`render_verification_table`][magnetofluidics_pinn.verification.reporting.render_verification_table]
    renders one row from.

    Args:
    - `name`: Short, human-readable identifier for the check.
    - `description`: One-sentence explanation of what is being verified.
    - `metric_value`: The measured value, or `None` for an unpopulated
      preview row (see
      [`pending_check`][magnetofluidics_pinn.verification.reporting.pending_check]).
    - `threshold`: The value `metric_value` is compared against, or `None`
      alongside an unpopulated `metric_value`.
    - `comparison`: One of `"<="`, `"<"`, `">="`, `">"`, read as
      `metric_value <comparison> threshold`.
    - `passed`: Whether the comparison holds, or `None` when unpopulated.
    """

    name: str
    description: str
    metric_value: float | None
    threshold: float | None
    comparison: Literal["<=", "<", ">=", ">"]
    passed: bool | None


def pending_check(
    name: str,
    description: str,
    comparison: Literal["<=", "<", ">=", ">"] = "<=",
) -> VerificationCheck:
    """Build an unpopulated `VerificationCheck` for a pre-execution preview table.

    Used to render the qualitative verification matrix a notebook shows
    before running any (potentially expensive) computation.

    Args:
    - `name`: Short, human-readable identifier for the check.
    - `description`: One-sentence explanation of what will be verified.
    - `comparison`: The comparison this check will eventually use; recorded
      now purely for display consistency between the preview and the final
      table.

    Returns:
    - A `VerificationCheck` with `metric_value`, `threshold`, and `passed`
      all `None`.

    Raises:
    - `ValueError`: If `comparison` is not one of `"<="`, `"<"`, `">="`,
      `">"`.
    """
    if comparison not in _COMPARISON_OPERATORS:
        raise ValueError(f"comparison must be one of {tuple(_COMPARISON_OPERATORS)!r}; got {comparison!r}.")
    return VerificationCheck(
        name=name, description=description, metric_value=None, threshold=None,
        comparison=comparison, passed=None,
    )


def evaluate_check(
    name: str,
    description: str,
    metric_value: float,
    threshold: float,
    comparison: Literal["<=", "<", ">=", ">"] = "<=",
) -> VerificationCheck:
    """Build a populated `VerificationCheck`, evaluating its pass/fail outcome.

    Args:
    - `name`: Short, human-readable identifier for the check.
    - `description`: One-sentence explanation of what is being verified.
    - `metric_value`: The measured value.
    - `threshold`: The value `metric_value` is compared against.
    - `comparison`: One of `"<="`, `"<"`, `">="`, `">"`.

    Returns:
    - A `VerificationCheck` with `passed` computed as
      `comparison(metric_value, threshold)`.

    Raises:
    - `ValueError`: If `comparison` is not one of `"<="`, `"<"`, `">="`,
      `">"`, or if `metric_value` or `threshold` is not finite.
    """
    if comparison not in _COMPARISON_OPERATORS:
        raise ValueError(f"comparison must be one of {tuple(_COMPARISON_OPERATORS)!r}; got {comparison!r}.")
    if not math.isfinite(metric_value):
        raise ValueError(f"metric_value must be finite; got {metric_value!r}.")
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite; got {threshold!r}.")
    passed = _COMPARISON_OPERATORS[comparison](metric_value, threshold)
    return VerificationCheck(
        name=name, description=description, metric_value=metric_value, threshold=threshold,
        comparison=comparison, passed=passed,
    )


def render_verification_table(checks: tuple[VerificationCheck, ...]) -> pd.DataFrame:
    """Render a sequence of `VerificationCheck` records as a display-ready table.

    Args:
    - `checks`: One or more `VerificationCheck` instances, populated or
      pending.

    Returns:
    - A `pandas.DataFrame` with one row per check and columns `"Check"`,
      `"Description"`, `"Metric value"`, `"Threshold"`, `"Comparison"`, and
      `"Passed"`; pending fields render as `"pending"` / `"—"`.

    Raises:
    - `ValueError`: If `checks` is empty.
    """
    if not checks:
        raise ValueError("checks must contain at least one VerificationCheck.")
    records = [
        {
            "Check": check.name,
            "Description": check.description,
            "Metric value": "pending" if check.metric_value is None else f"{check.metric_value:.3e}",
            "Threshold": "pending" if check.threshold is None else f"{check.threshold:.3e}",
            "Comparison": check.comparison,
            "Passed": "—" if check.passed is None else ("✓" if check.passed else "✗"),
        }
        for check in checks
    ]
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class VerificationThresholds:
    r"""Every calibratable tolerance a Phase 1 verification notebook compares against.

    Collecting every threshold into one immutable record - rather than
    scattering bare numeric literals through the analysis cells - keeps
    every tolerance visible and independently overridable in one place; see
    [`default_thresholds`][magnetofluidics_pinn.verification.reporting.default_thresholds]
    for how each field is derived from physically meaningful reference
    quantities rather than chosen as an arbitrary absolute constant.

    Args:
    - `exact_residual_tolerance`: Bound on the Stokes residual evaluated on
      the *closed-form* solution - a floating-point-accuracy question, not
      a physical one, so expressed as a multiple of machine epsilon rather
      than a reference-scale fraction.
    - `structural_tolerance`: Bound on the hard-constraint probes of an
      *untrained* network - likewise a floating-point-accuracy bound.
    - `global_l2_tolerance`: Bound on the trained network's global relative
      $L^2$ velocity error; already dimensionless, so no further scaling is
      applied.
    - `max_pointwise_tolerance`: Bound on the trained network's maximum
      pointwise velocity error, in the same (dimensionless velocity) units
      as `compute_max_pointwise_error`'s return value - a fraction of the
      centerline velocity scale, materialized as an absolute bound by
      `default_thresholds`.
    - `normalized_leakage_tolerance`: Bound on
      `compute_radial_leakage`'s `"normalized_radial_leakage"` (already a
      fraction of the centerline velocity).
    - `flow_rate_relative_tolerance`: Bound on
      `compute_flow_rate_errors`'s `"max_relative_flow_rate_error"`.
    - `pressure_gradient_relative_tolerance`: Bound on
      `compute_pressure_gradient_error`'s `"relative_error"`.
    - `held_out_residual_tolerance`: Bound on the held-out Stokes residual
      RMS - a physical-fidelity question, but expressed as an absolute
      bound in the package's own dimensionless residual units rather than
      rescaled by a reference quantity, since the residual is already
      nondimensional by construction of
      [`scaling.compute_scales`][magnetofluidics_pinn.scaling.compute_scales].
    - `conservation_of_mass_tolerance`: Bound on the conservation-of-mass
      residual RMS, in the package's own dimensionless residual units.
    - `noslip_tolerance`: Bound on the no-slip residual RMS, in the package's
       own dimensionless residual units.
    - `two_way_residual_tolerance`: Bound on the two-way residual RMS, in the
       package's own dimensionless residual units.
    - `manufactured_force_tolerance`: Bound on the manufactured-force residual
       RMS, in the package's own dimensionless residual units.
    - `quadrature_convergence_tolerance`: Bound on the quadrature convergence
       residual RMS, in the package's own dimensionless residual units.
    - `faxen_consistency_tolerance`: Bound on the Faxén consistency residual
      RMS, in the package's own dimensionless residual units.
    Raises:
    - `ValueError`: If any field is not finite and strictly positive.
    """

    exact_residual_tolerance: float
    structural_tolerance: float
    global_l2_tolerance: float
    max_pointwise_tolerance: float
    max_velocity_invariant_tolerance: float
    normalized_leakage_tolerance: float
    flow_rate_relative_tolerance: float
    pressure_gradient_relative_tolerance: float
    held_out_residual_tolerance: float
    conservation_of_mass_tolerance: float
    noslip_tolerance: float
    two_way_residual_tolerance: float      # NEW
    manufactured_force_tolerance: float    # NEW
    quadrature_convergence_tolerance: float  # NEW
    faxen_consistency_tolerance: float     # NEW

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field.name} must be finite and strictly positive; got {value!r}.")


def default_thresholds(
    dtype: torch.dtype,
    peak_velocity: float,
    *,
    exact_residual_epsilon_factor: float = 200.0,
    structural_epsilon_factor: float = 50.0,
    global_l2_tolerance: float = 0.05,
    max_pointwise_velocity_fraction: float = 0.10,
    max_velocity_invariant_tolerance: float = 0.10,
    normalized_leakage_fraction: float = 0.02,
    flow_rate_relative_tolerance: float = 0.03,
    pressure_gradient_relative_tolerance: float = 0.05,
    held_out_residual_tolerance: float = 1.0e-2,
    conservation_of_mass_tolerance: float = 5.0e-2,
    noslip_tolerance: float = 0.15,
    two_way_residual_tolerance: float = 2.0e-1,       # NEW — loose on purpose: the obstacle problem
                                                           # competes nine loss terms for a much smaller
                                                           # budget than the baseline Phase-1 recipe.
    manufactured_force_tolerance: float = 1.0e-4,     # NEW — matches the old MANUFACTURED_TOLERANCE
    quadrature_convergence_tolerance: float = 2.0e-2, # NEW — matches the old QUADRATURE_TOLERANCE
    faxen_consistency_tolerance: float = 0.15,        # NEW — matches the old FAXEN_TOLERANCE

) -> VerificationThresholds:
    r"""Build a `VerificationThresholds` instance from reference scales and calibratable factors.

    Every `*_factor` / `*_fraction` / `*_tolerance` keyword argument is a
    calibratable constant, exposed so a caller can override any subset
    without editing this module; the values above are starting points
    chosen for the Phase 1 straight-channel problem, not derived physical
    constants.

    Args:
    - `dtype`: Floating-point dtype the network and collocation tensors are
      evaluated in; sets the machine-epsilon scale for
      `exact_residual_tolerance` and `structural_tolerance`.
    - `peak_velocity`: Dimensionless centerline velocity $u_\text{peak}^*$,
      used to convert `max_pointwise_velocity_fraction` into an absolute
      bound.
    - `exact_residual_epsilon_factor`, `structural_epsilon_factor`:
      Multiples of `torch.finfo(dtype).eps` bounding the closed-form-
      residual and structural-probe checks, respectively.
    - `global_l2_tolerance`: Bound on the relative $L^2$ velocity error
      (already dimensionless).
    - `max_pointwise_velocity_fraction`: Fraction of `peak_velocity`
      bounding the maximum pointwise velocity error.
    - `normalized_leakage_fraction`: Bound on the radial leakage, already
      expressed as a fraction of `peak_velocity`.
    - `flow_rate_relative_tolerance`, `pressure_gradient_relative_tolerance`:
      Relative-error bounds on $Q(z)$ and the fitted pressure gradient.
    - `held_out_residual_tolerance`: Absolute bound on the held-out residual
      RMS, in the package's own dimensionless residual units.
    - `conservation_of_mass_tolerance`: Absolute bound on the conservation of
      mass residual, in the package's own dimensionless residual units.
    - `noslip_tolerance`: Absolute bound on the no-slip residual, in the
      package's own dimensionless residual units.
    - `two_way_residual_tolerance`: Absolute bound on the two-way residual, in the
      package's own dimensionless residual units.
    - `manufactured_force_tolerance`: Absolute bound on the manufactured-force residual, in the
      package's own dimensionless residual units.
    - `quadrature_convergence_tolerance`: Absolute bound on the quadrature convergence residual, in the
      package's own dimensionless residual units.
    - `faxen_consistency_tolerance`: Absolute bound on the Faxén consistency residual, in the
      package's own dimensionless residual units.

    Returns:
    - A populated
      [`VerificationThresholds`][magnetofluidics_pinn.verification.reporting.VerificationThresholds]
      instance.

    Raises:
    - `ValueError`: If `peak_velocity` is not finite and strictly positive;
      propagated from
      [`VerificationThresholds.__post_init__`][magnetofluidics_pinn.verification.reporting.VerificationThresholds]
      for any other invalid factor.
    """
    if not math.isfinite(peak_velocity) or peak_velocity <= 0.0:
        raise ValueError(f"peak_velocity must be finite and strictly positive; got {peak_velocity!r}.")
    machine_epsilon = torch.finfo(dtype).eps
    return VerificationThresholds(
        exact_residual_tolerance=exact_residual_epsilon_factor * machine_epsilon,
        structural_tolerance=structural_epsilon_factor * machine_epsilon,
        global_l2_tolerance=global_l2_tolerance,
        max_pointwise_tolerance=max_pointwise_velocity_fraction * peak_velocity,
        max_velocity_invariant_tolerance=max_velocity_invariant_tolerance * peak_velocity,
        normalized_leakage_tolerance=normalized_leakage_fraction,
        flow_rate_relative_tolerance=flow_rate_relative_tolerance,
        pressure_gradient_relative_tolerance=pressure_gradient_relative_tolerance,
        held_out_residual_tolerance=held_out_residual_tolerance,
        conservation_of_mass_tolerance=conservation_of_mass_tolerance,
        noslip_tolerance=noslip_tolerance,
        two_way_residual_tolerance=two_way_residual_tolerance,        # NEW
        manufactured_force_tolerance=manufactured_force_tolerance,    # NEW
        quadrature_convergence_tolerance=quadrature_convergence_tolerance,  # NEW
        faxen_consistency_tolerance=faxen_consistency_tolerance,      # NEW
    )
