r"""Gates comparing a warm-started run with its cold-started reference.

A warm start is acceptable only if it reproduces the cold-start observable
within the reference tolerance, at a lower cost. Both gates return a
`VerificationCheck`, so they plug into `render_verification_table`; report them
under the `convergence` category.
"""

from __future__ import annotations

import math

from magnetofluidics_pinn.training.stream_trainer import StreamLossHistory
from magnetofluidics_pinn.verification.reporting import VerificationCheck, evaluate_check


def warm_start_equivalence_gate(force_cold: float, force_warm: float, tolerance: float) -> VerificationCheck:
    r"""Check $\abs{F_\text{warm} - F_\text{cold}} / \abs{F_\text{cold}} \le$ `tolerance`.

    Args:
    - `force_cold`, `force_warm`: Forces from the cold and warm runs at the same geometry.
    - `tolerance`: Relative bound, for example from `reference_tolerance`.

    Returns:
    - A `VerificationCheck`.

    Raises:
    - `ValueError`: If an input is not finite, `force_cold` is zero, or `tolerance` is not positive.
    """
    if not all(math.isfinite(v) for v in (force_cold, force_warm, tolerance)):
        raise ValueError("Forces and tolerance must be finite.")
    if force_cold == 0.0:
        raise ValueError("force_cold must be non-zero for a relative comparison.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be strictly positive.")
    return evaluate_check(
        "Warm-start equivalence", "|F_warm - F_cold| / |F_cold| at the same geometry",
        abs(force_warm - force_cold) / abs(force_cold), tolerance, "<=",
    )


def warm_start_cost_ratio(history_cold: StreamLossHistory, history_warm: StreamLossHistory) -> float:
    """Ratio of recorded optimizer evaluations, warm over cold (Adam epochs plus L-BFGS closures).

    Args:
    - `history_cold`, `history_warm`: Histories of the two runs.

    Returns:
    - The cost ratio; below 1 means the warm start was cheaper.

    Raises:
    - `ValueError`: If the cold history is empty.
    """
    cost = lambda history: len(history.adam) + len(history.lbfgs)  # noqa: E731
    if cost(history_cold) == 0:
        raise ValueError("The cold history is empty.")
    return cost(history_warm) / cost(history_cold)
