---
bibliography: docs/references.bib
---
# `pymagnetofluidics`

Physics-informed neural networks (PINNs) [@raissi2019physics; @cai2021physics]
for predicting the motion of magnetic particles transported by a viscous flow
inside a microfluidic vessel, under a prescribed magnetic field.

Note: The name of the project, `pymagnetofluidics`, is a placeholder. Future prospect may include other models such as the Fourier Neural Operator (FNO) [@li2021fourier].

## Scope

The package couples two physical models:

1. **Viscous flow.** The incompressible Stokes equations (low Reynolds number,
   single particle, straight channel) [@happel1983low; @leal2007advanced] as a
   first approximation, extended to the incompressible Navier-Stokes
   equations for particle swarms and bifurcated geometries.
2. **Magnetic actuation.** The magnetic field is a prescribed input to the
   solver (an inlet/outlet-like boundary condition), not a quantity solved for
   by the network. The resulting dipole force on each particle is computed
   from that prescribed field and used to drive particle trajectories
   [@abbott2020magnetic].

## Installation (editable, development mode)

```bash
pip install -e ".[dev]"
```

## Package layout

```bash
pymagnetofluidics/
├── pyproject.toml
├── README.md
├── docs/
│   └── references.bib             # single project-wide bibliography
├── notebooks/
│   └── phase1_verification.qmd    # Quarto verification document
├── magnetofluidics_pinn/
│   ├── __init__.py                # public API re-exports, __version__
│   ├── config.py                  # DomainConfig, FluidConfig, FieldConfig, TrainingConfig
│   ├── device_utils.py            # centralized torch.device resolution
│   ├── io_utils.py                # checkpointing, config (de)serialization
│   ├── scaling.py                 # SI <-> dimensionless conversions
│   ├── types.py                   # Domain, CollocationPoints, FieldSample, ParticleState
│   ├── geometry/
│   │   ├── channel.py             # straight-vessel domain builder
│   │   └── bifurcation.py         # bifurcated-vessel domain builder (later phase)
│   ├── boundary_conditions/
│   │   ├── flow_bc.py             # inlet/outlet pressure or velocity, no-slip walls
│   │   └── magnetic_field_bc.py   # prescribed B-field as an input
│   ├── physics/
│   │   ├── fluid_residuals.py     # stokes_residual, navier_stokes_residual (later phase)
│   │   ├── magnetic_forcing.py    # dipole_force(moment, field_fn, coords)
│   │   └── hydrodynamic_drag.py   # faxen_corrected_velocity (particle-flow coupling)
│   ├── sampling/
│   │   └── collocation.py         # interior / boundary / initial point generators
│   ├── networks/
│   │   ├── mlp.py                 # CUDA-aware MLP builder
│   │   └── constraints.py         # apply_hard_wall_constraint (axis + wall)
│   ├── training/
│   │   ├── losses.py              # functional loss composition
│   │   └── trainer.py             # Adam + L-BFGS training loop
│   ├── trajectory/
│   │   └── integrator.py          # particle-path RK4 integration
│   └── visualization/
│       └── plotting.py            # streamlines, trajectories
└── tests/
    └── ...                        # mirrors src/ structure, one test module per source module
```

## Status

- **Phase 1** — single sphere, uniform $\mathbf{B}$, Stokes flow, straight
   channel — implemented and verified.** Interior residual, boundary
   conditions, network, sampling, training (Adam + L-BFGS refinement
   [@raissi2019physics]), and particle trajectory integration
   (with an optional Faxén-law finite-size correction
   [@faxen1922widerstand; @kim2005microhydrodynamics; @maxey1983equation])
   are complete; see `notebooks/phase1_verification.qmd` for the full,
   executable verification against the analytical Poiseuille solution.

- **Phase 2** — particle swarm (tens of particles), still Stokes, non-uniform $\mathbf{B}$.
- **Phase 3** — Navier–Stokes flow, bifurcated geometry, time-dependent magnetic actuation.

Architectural scaffold only. Function bodies raise `NotImplementedError`
until the corresponding implementation stage is approved.

## References
