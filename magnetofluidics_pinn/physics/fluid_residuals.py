r"""Physics residuals of the viscous flow equations.

Each residual function takes the network (mapping coordinates to velocity
and pressure) and a set of coordinates, and returns the pointwise PDE
residual through automatic differentiation. A residual of zero everywhere
means the network's output satisfies the governing equations exactly.


Phase 1 targets the axisymmetric, dimensionless Stokes (creeping-flow)
equations in cylindrical coordinates `(r, z)`, with velocity components
`(u_r, u_z)` and pressure `p`; see e.g. [@happel1983low] or
[@leal2007advanced] for the classical derivation. Phase 3 extends this to
the unsteady, convective (Navier-Stokes) case in `(r, z, t)`. With the
viscous pressure scale used by `scaling.compute_scales`, the dimensionless
viscosity is exactly 1 in both cases, so no viscosity factor appears below
— only, for `navier_stokes_residual`, the Reynolds number weighting the
unsteady and convective terms (see
[`FluidConfig.reynolds`][magnetofluidics_pinn.config.FluidConfig.reynolds]).

**Axis singularity, two ways.** Both `1 / r` and `1 / r^2` terms appear
below (the axisymmetric vector Laplacian's radial component, and the
continuity equation), which are singular at the symmetry axis, `r = 0`.
Every residual function here supports two mutually-exclusive strategies for
that, selected via `residual_form`:

- `"standard"` (the default, and the only form every release before this
  one implemented): evaluate the equations exactly as written above, and
  require `r > 0` at every collocation point — i.e., rely on the caller
  (typically
  [`sampling.sample_collocation_points`][magnetofluidics_pinn.sampling.collocation.sample_collocation_points]'s
  `_AXIS_CLEARANCE_FRACTION`) to never draw a point on, or numerically too
  close to, the axis.
- `"r_weighted"`: evaluate the same equations pre-multiplied through by $r$
  (continuity, $z$-momentum) or $r^2$ ($r$-momentum), which removes the
  singularity analytically — every term above becomes a regular (if,
  towards $r=0$, vanishing) function of $r$ — so `r = 0` is an admissible
  collocation point. This is not a strict improvement over `"standard"`:
  because the loss minimizes $\mathbb{E}[(r \cdot \text{residual})^2]$
  (or $r^2$), the multiplier itself shrinks the residual towards the axis
  independent of how well the underlying, unmultiplied equation is
  actually satisfied there, i.e. the PDE is enforced with *less relative
  weight* near $r=0$ than near the wall. What it buys is the ability to
  sample and enforce the PDE at all in the thin annulus between the axis
  and `"standard"`'s exclusion band, where today nothing is checked.

Neither strategy substitutes for
[`networks.apply_hard_wall_constraint`][magnetofluidics_pinn.networks.constraints.apply_hard_wall_constraint]:
that wrapper enforces the axis and wall *boundary conditions*
($u_r(0,z)=u_r(R,z)=0$,
$\partial u_z/\partial r|_{r=0}=0$) structurally, by construction of the
network itself; `residual_form` only concerns how safely the *interior* PDE
residual can be *evaluated* close to the axis. The two are complementary
and are expected to be used together.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import torch

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
from magnetofluidics_pinn.config import DomainConfig, FluidConfig

ResidualForm = Literal["standard", "r_weighted", "native"]
_RESIDUAL_FORMS: tuple[str, ...] = ("standard", "r_weighted", "native")

FieldsSZ = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class FlowDerivatives(NamedTuple):
    r"""Container for flow state and spatial derivatives evaluated via autodiff.

    Attributes:
    - `velocity_r`: Radial velocity component $u_r$, shape `(n_points, 1)`.
    - `velocity_z`: Axial velocity component $u_z$, shape `(n_points, 1)`.
    - `pressure`: Dimensionless pressure field $p$, shape `(n_points, 1)`.
    - `d_velocity_r_dr`: $\partial u_r / \partial r$, shape `(n_points, 1)`.
    - `d_velocity_r_dz`: $\partial u_r / \partial z$, shape `(n_points, 1)`.
    - `d_velocity_z_dr`: $\partial u_z / \partial r$, shape `(n_points, 1)`.
    - `d_velocity_z_dz`: $\partial u_z / \partial z$, shape `(n_points, 1)`.
    - `dp_dr`: $\partial p / \partial r$, shape `(n_points, 1)`.
    - `dp_dz`: $\partial p / \partial z$, shape `(n_points, 1)`.
    - `d2_velocity_r_dr2`: $\partial^2 u_r / \partial r^2$, shape `(n_points, 1)` or `None`.
    - `d2_velocity_r_dz2`: $\partial^2 u_r / \partial z^2$, shape `(n_points, 1)` or `None`.
    - `d2_velocity_z_dr2`: $\partial^2 u_z / \partial r^2$, shape `(n_points, 1)` or `None`.
    - `d2_velocity_z_dz2`: $\partial^2 u_z / \partial z^2$, shape `(n_points, 1)` or `None`.
    - `grad_velocity_r`: Full gradient tensor of $u_r$, shape `(n_points, n_dims)`.
    - `grad_velocity_z`: Full gradient tensor of $u_z$, shape `(n_points, n_dims)`.
    """
    velocity_r: torch.Tensor
    velocity_z: torch.Tensor
    pressure: torch.Tensor
    d_velocity_r_dr: torch.Tensor
    d_velocity_r_dz: torch.Tensor
    d_velocity_z_dr: torch.Tensor
    d_velocity_z_dz: torch.Tensor
    dp_dr: torch.Tensor
    dp_dz: torch.Tensor
    d2_velocity_r_dr2: torch.Tensor | None = None
    d2_velocity_r_dz2: torch.Tensor | None = None
    d2_velocity_z_dr2: torch.Tensor | None = None
    d2_velocity_z_dz2: torch.Tensor | None = None
    grad_velocity_r: torch.Tensor | None = None
    grad_velocity_z: torch.Tensor | None = None


def _validate_residual_form(residual_form: str) -> None:
    """Validate `residual_form` against the two values every residual function accepts.

    Args:
    - `residual_form`: The value passed by the caller.

    Raises:
    - `ValueError`: If `residual_form` is not one of `"standard"` or
      `"r_weighted"`.
    """
    if residual_form not in _RESIDUAL_FORMS:
        raise ValueError(
            f"residual_form must be one of {_RESIDUAL_FORMS!r}; got {residual_form!r}."
        )


def _validate_inputs(
    coordinates: torch.Tensor,
    expected_dim: int,
    residual_form: str,
    function_name: str,
) -> torch.Tensor:
    """Validate coordinates and return the radial coordinate column.

    CHANGED: Coordinates with $r = 0$ are now fully admissible under both
    `residual_form="standard"` and `residual_form="r_weighted"` due to analytical
    L'Hôpital axis limits.

    Args:
    - `coordinates`: Coordinate tensor requiring gradients.
    - `expected_dim`: Expected coordinate dimensions (2 for (r, z), 3 for (r, z, t)).
    - `residual_form`: Singularity treatment strategy.
    - `function_name`: Caller function name for descriptive error reporting.

    Returns:
    - Radial coordinate tensor of shape `(n_points, 1)`.
    """
    _validate_residual_form(residual_form)
    if coordinates.ndim != 2 or coordinates.shape[1] != expected_dim:
        dim_str = "(r, z)" if expected_dim == 2 else "(r, z, t)"
        raise ValueError(
            f"{function_name} requires shape (n_points, {expected_dim}) for "
            f"{dim_str}; got {tuple(coordinates.shape)}."
        )
    if not coordinates.requires_grad:
        raise ValueError(
            f"{function_name} coordinates must require gradients (`requires_grad_(True)`)."
        )
    radius = coordinates[:, 0:1]
    if torch.any(radius < 0.0):
        raise ValueError(
            f"{function_name} received negative radial coordinates; $r$ must be non-negative."
        )
    return radius


def compute_flow_derivatives(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    compute_second_order: bool = True,
) -> FlowDerivatives:
    r"""Evaluate flow network and compute spatial derivatives via automatic differentiation.

    Computes first-order spatial gradients $\nabla u_r$, $\nabla u_z$, $\nabla p$
    and, when requested, the second spatial derivatives needed for viscous diffusion
    $\nabla^2 \mathbf{u}$.

    Args:
    - `network`: Neural network callable mapping coordinates to $(u_r, u_z, p)$.
    - `coordinates`: Coordinate tensor requiring gradients, shape `(n_points, n_dims)`.
    - `compute_second_order`: If `True`, evaluates 2nd-order spatial derivatives.
      If `False`, skips 2nd-order backward passes to optimize GPU memory and latency.

    Returns:
    - `FlowDerivatives` containing evaluated fields and spatial derivatives.

    Raises:
    - `ValueError`: If `network` output does not have exactly 3 columns.
    """
    output = network(coordinates)
    if output.shape[1] != 3:
        raise ValueError(
            f"network must map coordinates to (u_r, u_z, p); got {output.shape[1]} output columns."
        )
    velocity_r = output[:, 0:1]
    velocity_z = output[:, 1:2]
    pressure = output[:, 2:3]

    grad_velocity_r = scalar_field_gradient(velocity_r, coordinates)
    grad_velocity_z = scalar_field_gradient(velocity_z, coordinates)
    grad_pressure = scalar_field_gradient(pressure, coordinates)

    d_velocity_r_dr = grad_velocity_r[:, 0:1]
    d_velocity_r_dz = grad_velocity_r[:, 1:2]
    d_velocity_z_dr = grad_velocity_z[:, 0:1]
    d_velocity_z_dz = grad_velocity_z[:, 1:2]
    dp_dr = grad_pressure[:, 0:1]
    dp_dz = grad_pressure[:, 1:2]

    if not compute_second_order:
        return FlowDerivatives(
            velocity_r=velocity_r,
            velocity_z=velocity_z,
            pressure=pressure,
            d_velocity_r_dr=d_velocity_r_dr,
            d_velocity_r_dz=d_velocity_r_dz,
            d_velocity_z_dr=d_velocity_z_dr,
            d_velocity_z_dz=d_velocity_z_dz,
            dp_dr=dp_dr,
            dp_dz=dp_dz,
            grad_velocity_r=grad_velocity_r,
            grad_velocity_z=grad_velocity_z,
        )

    d2_velocity_r_dr2 = scalar_field_gradient(d_velocity_r_dr, coordinates)[:, 0:1]
    d2_velocity_r_dz2 = scalar_field_gradient(d_velocity_r_dz, coordinates)[:, 1:2]
    d2_velocity_z_dr2 = scalar_field_gradient(d_velocity_z_dr, coordinates)[:, 0:1]
    d2_velocity_z_dz2 = scalar_field_gradient(d_velocity_z_dz, coordinates)[:, 1:2]

    return FlowDerivatives(
        velocity_r=velocity_r,
        velocity_z=velocity_z,
        pressure=pressure,
        d_velocity_r_dr=d_velocity_r_dr,
        d_velocity_r_dz=d_velocity_r_dz,
        d_velocity_z_dr=d_velocity_z_dr,
        d_velocity_z_dz=d_velocity_z_dz,
        dp_dr=dp_dr,
        dp_dz=dp_dz,
        d2_velocity_r_dr2=d2_velocity_r_dr2,
        d2_velocity_r_dz2=d2_velocity_r_dz2,
        d2_velocity_z_dr2=d2_velocity_z_dr2,
        d2_velocity_z_dz2=d2_velocity_z_dz2,
        grad_velocity_r=grad_velocity_r,
        grad_velocity_z=grad_velocity_z,
    )


def _axis_threshold(radius: torch.Tensor, eps: float | None) -> float:
    """Axis switch radius: sqrt(machine epsilon) of the dtype unless given."""
    return torch.finfo(radius.dtype).eps ** 0.5 if eps is None else eps


def axisymmetric_vector_laplacian(
    derivs: FlowDerivatives,
    radius: torch.Tensor,
    residual_form: ResidualForm = "standard",
    axis_regularization_eps: float | None = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Evaluate the axisymmetric vector Laplacian $\nabla^2 \mathbf{u} = (\nabla_r^2 \mathbf{u}, \nabla_z^2 \mathbf{u})$.

    Under standard cylindrical coordinates [@happel1983low; @leal2007advanced]:
    $$\nabla_r^2 \mathbf{u} = \frac{\partial^2 u_r}{\partial r^2} + \frac{1}{r}\frac{\partial u_r}{\partial r}
      - \frac{u_r}{r^2} + \frac{\partial^2 u_r}{\partial z^2}, \qquad
    \nabla_z^2 \mathbf{u} = \frac{\partial^2 u_z}{\partial r^2} + \frac{1}{r}\frac{\partial u_z}{\partial r}
      + \frac{\partial^2 u_z}{\partial z^2}.$$
    On the symmetry axis ($r \to 0$), symmetry requires $u_r(0, z) = 0$ and $\partial u_z / \partial r |_{r=0} = 0$.
    Applying L'Hôpital's rule yields the non-singular physical limits [@happel1983low]:
    $$\lim_{r \to 0} \left(\frac{1}{r}\frac{\partial u_z}{\partial r}\right) = \frac{\partial^2 u_z}{\partial r^2}, \qquad
    \lim_{r \to 0} \left(\frac{1}{r}\frac{\partial u_r}{\partial r} - \frac{u_r}{r^2}\right) = 0.$$
    Consequently, on $r = 0$:
    $$\left.\nabla_r^2 \mathbf{u}\right|_{r=0} = 0, \qquad
    \left.\nabla_z^2 \mathbf{u}\right|_{r=0} = 2 \frac{\partial^2 u_z}{\partial r^2} + \frac{\partial^2 u_z}{\partial z^2}.$$

    Args:
    - `derivs`: First and second spatial derivatives from `compute_flow_derivatives`.
    - `radius`: Tensor `(n_points, 1)` of radial coordinates.
    - `residual_form`: `"standard"` or `"r_weighted"`.
    - `axis_regularization_eps`: Radius below which the L'Hopital axis limit is used. `None` selects
      $\sqrt{\varepsilon_\text{mach}}$ of the dtype ($3.4\times10^{-4}$ in `float32`, $1.5\times10^{-8}$ in `float64`).

    Returns:
    - Tuple `(laplacian_r, laplacian_z)` of tensors shaped `(n_points, 1)`.

    Raises:
    - `ValueError`: If second-order derivatives were not computed in `derivs`.
    """
    if (
        derivs.d2_velocity_r_dr2 is None
        or derivs.d2_velocity_r_dz2 is None
        or derivs.d2_velocity_z_dr2 is None
        or derivs.d2_velocity_z_dz2 is None
    ):
        raise ValueError(
            "axisymmetric_vector_laplacian requires second-order derivatives; "
            "ensure compute_second_order=True in compute_flow_derivatives."
        )

    axis_threshold = _axis_threshold(radius,
                                     axis_regularization_eps)  # CHANGED (float32 -> 3.5e-4, float64 -> 1.5e-8)
    _validate_residual_form(residual_form)  # NEW
    if (derivs.d2_velocity_r_dr2 is None or derivs.d2_velocity_r_dz2 is None
        or derivs.d2_velocity_z_dr2 is None or derivs.d2_velocity_z_dz2 is None):
        raise ValueError(
            "axisymmetric_vector_laplacian requires second-order derivatives.")
    axis_threshold = _axis_threshold(radius, axis_regularization_eps)
    if residual_form == "standard":
        is_axis = radius < axis_threshold
        safe_radius = torch.where(is_axis, torch.full_like(radius, axis_threshold),
                                  radius)
        lap_r_standard = (
            derivs.d2_velocity_r_dr2 + derivs.d_velocity_r_dr / safe_radius
            - derivs.velocity_r / safe_radius.square() + derivs.d2_velocity_r_dz2)
        lap_z_standard = derivs.d2_velocity_z_dr2 + derivs.d_velocity_z_dr / safe_radius + derivs.d2_velocity_z_dz2
        lap_r_axis = torch.zeros_like(lap_r_standard)
        lap_z_axis = 2.0 * derivs.d2_velocity_z_dr2 + derivs.d2_velocity_z_dz2
        return torch.where(is_axis, lap_r_axis, lap_r_standard), torch.where(is_axis,
                                                                             lap_z_axis,
                                                                             lap_z_standard)
    r_2 = radius.square()
    laplacian_r = (r_2 * (derivs.d2_velocity_r_dr2 + derivs.d2_velocity_r_dz2)
                   + radius * derivs.d_velocity_r_dr - derivs.velocity_r)
    laplacian_z = radius * (
        derivs.d2_velocity_z_dr2 + derivs.d2_velocity_z_dz2) + derivs.d_velocity_z_dr
    return laplacian_r, laplacian_z


def _continuity_residual(
    velocity_r: torch.Tensor,
    d_velocity_r_dr: torch.Tensor,
    d_velocity_z_dz: torch.Tensor,
    radius: torch.Tensor,
    residual_form: ResidualForm = "standard",
    axis_regularization_eps: float = 1.0e-7,
) -> torch.Tensor:
    r"""Evaluate axisymmetric continuity residual $\nabla \cdot \mathbf{u} = 0$.

    For $r > 0$:
    $$\nabla \cdot \mathbf{u} = \frac{\partial u_r}{\partial r} + \frac{u_r}{r} + \frac{\partial u_z}{\partial z} = 0.$$
    As $r \to 0$, applying L'Hôpital's rule gives $\lim_{r \to 0} \frac{u_r}{r} = \frac{\partial u_r}{\partial r}$, yielding:
    $$\left.\nabla \cdot \mathbf{u}\right|_{r=0} = 2 \frac{\partial u_r}{\partial r} + \frac{\partial u_z}{\partial z} = 0.$$


    Args:
    - `velocity_r`, `d_velocity_r_dr`, `d_velocity_z_dz`, `radius`: Tensors `(n_points, 1)`.
    - `residual_form`: `"standard"` or `"r_weighted"`.
    - `axis_regularization_eps`: Axis switch radius; `None` selects the dtype-aware default.

    Returns:
    - Residual tensor `(n_points, 1)`.

    Raises:
    - `ValueError`: If `residual_form` is invalid.
    """
    _validate_residual_form(residual_form)  # NEW
    if residual_form == "standard":
        threshold = _axis_threshold(radius, axis_regularization_eps)  # CHANGED
        is_axis = radius < threshold
        safe_radius = torch.where(is_axis, torch.full_like(radius, threshold), radius)
        standard_div = d_velocity_r_dr + velocity_r / safe_radius + d_velocity_z_dz
        return torch.where(is_axis, 2.0 * d_velocity_r_dr + d_velocity_z_dz,
                           standard_div)
    return radius * (d_velocity_r_dr + d_velocity_z_dz) + velocity_r


# CHANGED: Refactored to compose with `axisymmetric_vector_laplacian`.
def _stokes_momentum_residuals(
    derivs: FlowDerivatives,
    radius: torch.Tensor,
    residual_form: ResidualForm = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the steady Stokes momentum residuals: -grad p + laplacian u."""
    laplacian_r, laplacian_z = axisymmetric_vector_laplacian(
        derivs=derivs,
        radius=radius,
        residual_form=residual_form,
    )
    if residual_form == "standard":
        momentum_r = -derivs.dp_dr + laplacian_r
        momentum_z = -derivs.dp_dz + laplacian_z
    else:
        momentum_r = -radius.square() * derivs.dp_dr + laplacian_r
        momentum_z = -radius * derivs.dp_dz + laplacian_z
    return momentum_r, momentum_z


def _gradient_or_zero(output: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:  # NEW
    """Differentiate `output` with respect to `sz`, returning zeros if `output` does not depend on it.

    A derivative that is independent of the coordinates (for example the first derivative of a linear field) carries no
    graph, so `torch.autograd.grad` would raise. Its next derivative is exactly zero.

    Args:
    - `output`: Tensor `(n, 1)`.
    - `sz`: Tensor `(n, 2)` requiring gradients.

    Returns:
    - Tensor `(n, 2)` with `d(output)/d(sz)`, or zeros when `output` has no graph.
    """
    if not output.requires_grad:
        return torch.zeros_like(sz)
    return scalar_field_gradient(output, sz)


def stokes_residual_sz(fields_sz: FieldsSZ, sz: torch.Tensor) -> torch.Tensor:  # NEW
    r"""Evaluate the Stokes residual in the regular variables $(s, z)$, $s = r^2$, with no division by $r$.

    The velocity is written $u_r = r\,g(s, z)$, $u_z = u_z(s, z)$ and the pressure $p(s, z)$. With
    $\partial_r = 2r\,\partial_s$ the force densities become

    $$ R_r = r\left(-2p_s + 4s\,g_{ss} + 8\,g_s + g_{zz}\right), \qquad
       R_z = -p_z + 4s\,u_{z,ss} + 4\,u_{z,s} + u_{z,zz}, $$

    and the divergence is $2g + 2s\,g_s + u_{z,z}$. Every term is regular on the axis. $R_r$ is the physical
    radial force density, identical to the `"standard"` form, and the factor $r = \sqrt{s}$ is applied after the
    derivatives, so it is never differentiated.

    Args:
    - `fields_sz`: Callable mapping a tensor `(n, 2)` holding $(s, z)$ to the tuple `(g, u_z, p)` of tensors `(n, 1)`,
      built so that the graph connects the outputs to its input (see `StreamFunctionSphereFlow.fields_sz`).
    - `sz`: Tensor `(n, 2)` with $(s, z)$, `s >= 0`, requiring gradients.

    Returns:
    - Tensor `(n, 3)` with columns $(R_r, R_z, \nabla\cdot\mathbf u)$. The graph is kept for training.

    Raises:
    - `ValueError`: If `sz` is not `(n, 2)`, does not require gradients, or has negative `s`.
    """
    if sz.ndim != 2 or sz.shape[1] != 2:
        raise ValueError(f"sz must have shape (n_points, 2); got {tuple(sz.shape)}.")
    if not sz.requires_grad:
        raise ValueError("sz must require gradients (`requires_grad_(True)`).")
    if torch.any(sz[:, 0:1] < 0.0):
        raise ValueError("sz contains negative s = r^2.")
    g, u_z, pressure = fields_sz(sz)
    grad_g = _gradient_or_zero(g, sz)  
    grad_u = _gradient_or_zero(u_z, sz)
    grad_p = _gradient_or_zero(pressure, sz)
    g_ss = _gradient_or_zero(grad_g[:, 0:1], sz)[:, 0:1]
    g_zz = _gradient_or_zero(grad_g[:, 1:2], sz)[:, 1:2]
    u_ss = _gradient_or_zero(grad_u[:, 0:1], sz)[:, 0:1]
    u_zz = _gradient_or_zero(grad_u[:, 1:2], sz)[:, 1:2]
    s = sz[:, 0:1]
    radial_over_r = -2.0 * grad_p[:, 0:1] + 4.0 * s * g_ss + 8.0 * grad_g[:, 0:1] + g_zz
    momentum_r = torch.sqrt(s.clamp(min=0.0)) * radial_over_r
    momentum_z = -grad_p[:, 1:2] + 4.0 * s * u_ss + 4.0 * grad_u[:, 0:1] + u_zz
    continuity = 2.0 * g + 2.0 * s * grad_g[:, 0:1] + grad_u[:, 1:2]
    return torch.cat([momentum_r, momentum_z, continuity], dim=1)


# TODO: make  `stokes_residual` use of `DomainConfig` instead of `FluidConfig` for consistency with
#  `navier_stokes_residual`
def stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    config : DomainConfig | FluidConfig,
    residual_form: ResidualForm = "standard",
) -> torch.Tensor:
    r"""Evaluate the incompressible Stokes-flow residual.

    Enforces continuity, $\nabla \cdot \mathbf{u} = 0$, and the steady
    momentum balance, $-\nabla p + \nabla^2 \mathbf{u} = 0$, in
    dimensionless form, at each coordinate.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, n_dims)`
      to a tensor of shape `(n_points, n_dims + 1)`, the last column being
      pressure and the preceding columns being velocity components.
    - `coordinates`: Tensor of shape `(n_points, n_dims)`, requiring
      gradients, sampled inside the domain.
    - `domain`: Fluid configuration; must have `regime == "stokes"`.
    - `residual_form`: NEW. `"standard"` (default) evaluates the equations
      as classically written, and requires `r > 0` everywhere (unchanged
      from every prior release). `"r_weighted"` evaluates the same
      equations pre-multiplied by $r$ or $r^2$ to remove the axis
      singularity analytically, and additionally accepts `r = 0`. See this
      module's docstring for the full derivation and the trade-off between
      the two.

    Returns:
    - Tensor of shape `(n_points, n_dims + 1)` with the continuity residual
      in the last column and the momentum residuals in the preceding
      columns. Under `"r_weighted"`, every column is the corresponding
      `"standard"` column multiplied by $r$ ($z$-momentum, continuity) or
      $r^2$ ($r$-momentum).

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"stokes"`, if
      `residual_form` is not `"standard"` or `"r_weighted"`, if
      `coordinates` does not have shape `(n_points, 2)` (the axisymmetric
      `(r, z)` convention this function implements), if `coordinates` does
      not require gradients, if any radial coordinate is negative (or, under
      `residual_form="standard"`, on or across the symmetry axis, `r <= 0`,
      where that form's residual is singular), or if `network`'s output does
      not have exactly 3 columns (`u_r`, `u_z`, `p`).
    """
    fluid_config = config .fluid if isinstance(config , DomainConfig) else config
    # fluid_config = domain.fluid
    if fluid_config.regime != "stokes":
        raise ValueError(
            f"Expected a Stokes fluid configuration, got regime={fluid_config.regime!r}."
        )
    # Handle "native" residual form by delegating to stokes_residual_sz if available
    if residual_form == "native":
        # Check if network has stokes_residual_sz method (StreamFunctionSphereFlow)
        if hasattr(network, 'stokes_residual_sz'):
            return network.stokes_residual_sz(coordinates)
        else:
            raise ValueError(
                "residual_form='native' requires a network with stokes_residual_sz() method "
                "(e.g., StreamFunctionSphereFlow)."
            )
    radius = _validate_inputs(
        coordinates=coordinates,
        expected_dim=2,
        residual_form=residual_form,
        function_name="stokes_residual",
    )

    derivs = compute_flow_derivatives(network, coordinates, compute_second_order=True)
    momentum_r, momentum_z = _stokes_momentum_residuals(derivs, radius, residual_form)
    continuity = _continuity_residual(
        velocity_r=derivs.velocity_r,
        d_velocity_r_dr=derivs.d_velocity_r_dr,
        d_velocity_z_dz=derivs.d_velocity_z_dz,
        radius=radius,
        residual_form=residual_form,
    )

    return torch.cat([momentum_r, momentum_z, continuity], dim=1)


# CHANGED: Cleaned up redundant input checks prior to `_validate_inputs`.
# WARNING: `stokes_residual` and `navier_stokes_residual` don't have the same args!!
def navier_stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    domain_config: DomainConfig,
    residual_form: ResidualForm = "standard",
) -> torch.Tensor:
    r"""Evaluate the incompressible, unsteady Navier-Stokes residual.

    Enforces continuity and the momentum balance including the convective and
    unsteady terms, in dimensionless form, at each coordinate:
    $$
    \nabla \cdot \mathbf{u} = 0, \qquad
    Re\left(\frac{\partial \mathbf{u}}{\partial t}
      + (\mathbf{u} \cdot \nabla)\mathbf{u}\right)
      + \nabla p - \nabla^2 \mathbf{u} = \mathbf{0},
    $$
    with $Re$ = [`fluid_config.reynolds`][magnetofluidics_pinn.config.FluidConfig.reynolds].
    Continuity carries no time derivative for an incompressible fluid, so
    its form (and $r$-weighted variant) is identical to
    [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]'s;
    only the momentum equations gain the $Re$-weighted unsteady and
    convective terms. `Re` multiplies through consistently for the
    `"r_weighted"` form as well: the full momentum equation, including its
    $Re$ term, is what gets pre-multiplied by $r$ or $r^2$, exactly as for
    `stokes_residual`'s viscous/pressure terms — see this module's
    docstring for that derivation.

    No external (e.g. magnetic) body-force term appears here, matching
    `stokes_residual`: this package's magnetic dipole force
    ([`physics.magnetic_forcing.dipole_force`][magnetofluidics_pinn.physics.magnetic_forcing.dipole_force])
    acts on a tracer particle's equation of motion in
    `trajectory.integrate_trajectory`, not on the carrier fluid itself
    (an ordinary, non-magnetized Newtonian fluid is assumed throughout,
    not a ferrofluid), so it has no place in the fluid's own momentum
    balance.

    Args:
    - `network`: Callable mapping coordinates of shape `(n_points, 3)` to a
      tensor of shape `(n_points, 3)`: `(u_r, u_z, p)`.
    - `coordinates`: Tensor of shape `(n_points, 3)`, requiring gradients,
      sampled inside the domain, in the `(r, z, t)` convention (an explicit
      time column, appended after the steady `(r, z)` pair
      [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
      uses).
    - `fluid_config`: Fluid configuration; must have
      `fluid_config.regime == "navier_stokes"`.
    - `residual_form`: NEW. Same meaning as
      [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]'s
      `residual_form`: `"standard"` (default) requires `r > 0`;
      `"r_weighted"` removes the axis singularity analytically and accepts
      `r = 0`.

    Returns:
    - Tensor of shape `(n_points, 3)` with the continuity residual in the
      last column and the momentum residuals in the preceding columns,
      exactly as
      [`stokes_residual`][magnetofluidics_pinn.physics.fluid_residuals.stokes_residual]
      returns them.

    Raises:
    - `ValueError`: If `fluid_config.regime` is not `"navier_stokes"`, if
      `residual_form` is not `"standard"` or `"r_weighted"`, if
      `coordinates` does not have shape `(n_points, 3)` (the axisymmetric
      `(r, z, t)` convention this function implements), if `coordinates`
      does not require gradients, if any radial coordinate is negative (or,
      under `residual_form="standard"`, on or across the symmetry axis),
      or if `network`'s output does not have exactly 3 columns.
    """
    if domain_config.fluid.regime != "navier_stokes":
        raise ValueError(
            "Expected a Navier-Stokes fluid configuration, "
            f"got regime={domain_config.fluid.regime!r}."
        )
    radius = _validate_inputs(
        coordinates=coordinates,
        expected_dim=3,
        residual_form=residual_form,
        function_name="navier_stokes_residual",
    )

    derivs = compute_flow_derivatives(network, coordinates, compute_second_order=True)

    # Autodiff temporal gradients from 3rd coordinate column (time t)
    d_velocity_r_dt = derivs.grad_velocity_r[:, 2:3]
    d_velocity_z_dt = derivs.grad_velocity_z[:, 2:3]

    material_derivative_r = (
        d_velocity_r_dt
        + derivs.velocity_r * derivs.d_velocity_r_dr
        + derivs.velocity_z * derivs.d_velocity_r_dz
    )
    material_derivative_z = (
        d_velocity_z_dt
        + derivs.velocity_r * derivs.d_velocity_z_dr
        + derivs.velocity_z * derivs.d_velocity_z_dz
    )

    stokes_mom_r, stokes_mom_z = _stokes_momentum_residuals(derivs, radius,
                                                            residual_form)

    reynolds = domain_config.reynolds
    if residual_form == "standard":
        momentum_r = reynolds * material_derivative_r - stokes_mom_r
        momentum_z = reynolds * material_derivative_z - stokes_mom_z
    else:
        momentum_r = radius.square() * reynolds * material_derivative_r - stokes_mom_r
        momentum_z = radius * reynolds * material_derivative_z - stokes_mom_z

    continuity = _continuity_residual(
        velocity_r=derivs.velocity_r,
        d_velocity_r_dr=derivs.d_velocity_r_dr,
        d_velocity_z_dz=derivs.d_velocity_z_dz,
        radius=radius,
        residual_form=residual_form,
    )

    return torch.cat([momentum_r, momentum_z, continuity], dim=1)
