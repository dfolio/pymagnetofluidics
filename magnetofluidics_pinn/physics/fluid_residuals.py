"""Physics residuals of the viscous flow equations.

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
  because the loss minimizes $\\mathbb{E}[(r \\cdot \\text{residual})^2]$
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
$\\partial u_z/\\partial r|_{r=0}=0$) structurally, by construction of the
network itself; `residual_form` only concerns how safely the *interior* PDE
residual can be *evaluated* close to the axis. The two are complementary
and are expected to be used together.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import torch

from magnetofluidics_pinn.autodiff_utils import scalar_field_gradient
from magnetofluidics_pinn.config import FluidConfig

# The two supported interior-residual singularity treatments; shared between
# `stokes_residual` and `navier_stokes_residual` so both validate and
# document the option identically rather than drifting apart.
_RESIDUAL_FORMS = ("standard", "r_weighted")


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
    """Validate coordinates and residual form, returning the radial coordinates tensor.

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
            f"coordinates must have shape (n_points, {expected_dim}) for the axisymmetric "
            f"{dim_str} formulation; got {tuple(coordinates.shape)}."
        )
    if not coordinates.requires_grad:
        raise ValueError(
            "coordinates must require gradients (call `.requires_grad_(True)`) "
            "so that the residual can be evaluated through automatic differentiation."
        )
    radius = coordinates[:, 0:1]
    if residual_form == "standard":
        if torch.any(radius <= 0.0):
            raise ValueError(
                f"{function_name} received coordinates on or across the symmetry "
                "axis (r <= 0) under residual_form='standard'; exclude the axis "
                "when sampling interior points, or use residual_form='r_weighted'."
            )
    elif torch.any(radius < 0.0):
        raise ValueError(
            f"{function_name} received a negative radial coordinate under "
            f"residual_form={residual_form!r}; r must be non-negative."
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


# NEW: Shared axisymmetric vector Laplacian operator for cylindrical flow fields.
def axisymmetric_vector_laplacian(
    derivs: FlowDerivatives,
    radius: torch.Tensor,
    residual_form: Literal["standard", "r_weighted"] = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Evaluate the axisymmetric vector Laplacian $\nabla^2 \mathbf{u} = (\nabla_r^2 \mathbf{u}, \nabla_z^2 \mathbf{u})$.

    Under standard cylindrical coordinates [@happel1983low; @leal2007advanced]:
    $$\nabla_r^2 \mathbf{u} = \frac{\partial^2 u_r}{\partial r^2} + \frac{1}{r}\frac{\partial u_r}{\partial r}
      - \frac{u_r}{r^2} + \frac{\partial^2 u_r}{\partial z^2}, \qquad
    \nabla_z^2 \mathbf{u} = \frac{\partial^2 u_z}{\partial r^2} + \frac{1}{r}\frac{\partial u_z}{\partial r}
      + \frac{\partial^2 u_z}{\partial z^2}.$$
    Under `"r_weighted"`, equations are multiplied analytically by $r^2$ (radial) and $r$ (axial)
    to eliminate coordinate singularities at $r = 0$:
    $$r^2 \nabla_r^2 \mathbf{u} = r^2\left(\frac{\partial^2 u_r}{\partial r^2} + \frac{\partial^2 u_r}{\partial z^2}\right)
      + r \frac{\partial u_r}{\partial r} - u_r, \qquad
    r \nabla_z^2 \mathbf{u} = r\left(\frac{\partial^2 u_z}{\partial r^2} + \frac{\partial^2 u_z}{\partial z^2}\right)
      + \frac{\partial u_z}{\partial r}.$$

    Args:
    - `derivs`: Flow derivatives containing first and second spatial derivatives.
    - `radius`: Radial coordinates $r$ of shape `(n_points, 1)`.
    - `residual_form`: `"standard"` or `"r_weighted"`.

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

    if residual_form == "standard":
        laplacian_r = (
            derivs.d2_velocity_r_dr2
            + derivs.d_velocity_r_dr / radius
            - derivs.velocity_r / radius.square()
            + derivs.d2_velocity_r_dz2
        )
        laplacian_z = (
            derivs.d2_velocity_z_dr2
            + derivs.d_velocity_z_dr / radius
            + derivs.d2_velocity_z_dz2
        )
    else:
        r_2 = radius.square()
        laplacian_r = (
            r_2 * (derivs.d2_velocity_r_dr2 + derivs.d2_velocity_r_dz2)
            + radius * derivs.d_velocity_r_dr
            - derivs.velocity_r
        )
        laplacian_z = (
            radius * (derivs.d2_velocity_z_dr2 + derivs.d2_velocity_z_dz2)
            + derivs.d_velocity_z_dr
        )
    return laplacian_r, laplacian_z


def _continuity_residual(
    velocity_r: torch.Tensor,
    d_velocity_r_dr: torch.Tensor,
    d_velocity_z_dz: torch.Tensor,
    radius: torch.Tensor,
    residual_form: Literal["standard", "r_weighted"] = "standard",
) -> torch.Tensor:
    """Evaluate axisymmetric continuity residual under standard or r-weighted form."""
    if residual_form == "standard":
        return d_velocity_r_dr + velocity_r / radius + d_velocity_z_dz
    return radius * (d_velocity_r_dr + d_velocity_z_dz) + velocity_r


# CHANGED: Refactored to compose with `axisymmetric_vector_laplacian`.
def _stokes_momentum_residuals(
    derivs: FlowDerivatives,
    radius: torch.Tensor,
    residual_form:  Literal["standard", "r_weighted"] = "standard",
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


# CHANGED: Cleaned up redundant input checks prior to `_validate_inputs`.
def stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    fluid_config: FluidConfig,
    residual_form: Literal["standard", "r_weighted"] = "standard",
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
    - `fluid_config`: Fluid configuration; must have `regime == "stokes"`.
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
    if fluid_config.regime != "stokes":
        raise ValueError(
            f"Expected a Stokes fluid configuration, got regime={fluid_config.regime!r}."
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
def navier_stokes_residual(
    network: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    fluid_config: FluidConfig,
    residual_form: Literal["standard", "r_weighted"] = "standard",
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
      `regime == "navier_stokes"`.
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
    if fluid_config.regime != "navier_stokes":
        raise ValueError(
            "Expected a Navier-Stokes fluid configuration, "
            f"got regime={fluid_config.regime!r}."
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
    
    stokes_mom_r, stokes_mom_z = _stokes_momentum_residuals(derivs, radius, residual_form)
    
    reynolds = fluid_config.reynolds
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
