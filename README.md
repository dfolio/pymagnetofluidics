# magnetofluidics_pinn

Physics-informed neural networks (PINNs) for predicting the motion of magnetic
particles transported by a viscous flow inside a microfluidic vessel, under a
prescribed magnetic field.

## Scope

The package couples two physical models:

1. **Viscous flow.** The incompressible Stokes equations (low Reynolds number,
   single particle, straight channel) as a first approximation, extended to
   the incompressible Navier-Stokes equations for particle swarms and
   bifurcated geometries.
2. **Magnetic actuation.** The magnetic field is a prescribed input to the
   solver (an inlet/outlet-like boundary condition), not a quantity solved for
   by the network. The resulting dipole force on each particle is computed
   from that prescribed field and used to drive particle trajectories.

## Installation (editable, development mode)

```bash
pip install -e ".[dev]"
```

## Package layout

See `src/magnetofluidics_pinn/` for the module breakdown: `geometry`,
`boundary_conditions`, `physics`, `sampling`, `networks`, `training`,
`trajectory`, and `visualization`.

```bash
magnetofluidics_pinn/
├── pyproject.toml
├── README.md
├── src/
│   └── magnetofluidics_pinn/
│       ├── __init__.py                # public API re-exports, __version__
│       ├── config.py                  # DomainConfig, FluidConfig, FieldConfig, TrainingConfig
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── channel.py             # straight-vessel domain builder
│       │   └── bifurcation.py         # bifurcated-vessel domain builder
│       ├── boundary_conditions/
│       │   ├── __init__.py
│       │   ├── flow_bc.py             # inlet/outlet pressure or velocity, no-slip walls
│       │   └── magnetic_field_bc.py   # prescribed B-field as an input (see note below)
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── fluid_residuals.py     # stokes_residual, navier_stokes_residual
│       │   └── magnetic_forcing.py    # dipole_force(moment, field_fn, coords)
│       ├── sampling/
│       │   ├── __init__.py
│       │   └── collocation.py         # interior / boundary / initial point generators
│       ├── networks/
│       │   ├── __init__.py
│       │   └── mlp.py                 # CUDA-aware MLP builder
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py              # functional loss composition
│       │   └── trainer.py             # training loop
│       ├── trajectory/
│       │   ├── __init__.py
│       │   └── integrator.py          # particle-path ODE integration
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plotting.py            # streamlines, trajectories
│       └── io_utils.py                # checkpointing, config (de)serialization
└── tests/
    └── ...                            # mirrors src/ structure, one test module per source module
```

## Status

Architectural scaffold only. Function bodies raise `NotImplementedError`
until the corresponding implementation stage is approved.


Phase 1 — single sphere, uniform $\mathbf{B}$, Stokes flow, straight channel.
Phase 2 — particle swarm (tens of particles), still Stokes, non-uniform $\mathbf{B}$.
Phase 3 — Navier–Stokes flow, bifurcated geometry, time-dependent magnetic actuation.