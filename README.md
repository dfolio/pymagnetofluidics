---
bibliography: docs/references.bib
---

# `pymagnetofluidics` 

[![Python](https://img.shields.io/badge/python-3.10+-white?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.13+-white?logo=pytorch)](https://pytorch.org)
[![Numpy](https://img.shields.io/badge/Numpy-1.5.0+-blue?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Physics-Informed AI, currently Physics-Informed Neural Networks (PINNs) [@raissi2019physics; @cai2021physics] for 
predicting viscous microfluidic flows and tracking magnetic microrobot trajectories under prescribed magnetic actuation.

---

## Executive Summary & Scope

`pymagnetofluidics` provides a modular, GPU-accelerated framework combining deep learning, physics-informed differential constraints, and dynamic particle integration. The package couples two core physical models:

1. **Viscous Hydrodynamics (Stokes & Navier–Stokes):** Solves axisymmetric creeping flow equations in cylindrical microchannels [@happel1983low; @leal2007advanced]. Structural hard constraints [@lu2021physics] enforce exact radial wall no-penetration ($u_r(R, z) \equiv 0$) and axis regularity ($u_r(0, z) \equiv 0$), guaranteeing exact volumetric mass conservation along the channel.
2. **Magnetic Actuation & Particle Transport:** Computes point-dipole forces $\mathbf{F}_{\mathrm{mag}} = (\mathbf{m}
   \cdot \nabla) \mathbf{B}$ under prescribed static or dynamic magnetic fields [@abbott2020magnetic]. Hydrodynamic drag includes optional finite-size Faxén corrections [@faxen1922widerstand; @kim2005microhydrodynamics] to couple continuum flow fields with discrete particle dynamics.

---

## Key Features

- **Hard Boundary Constraints:** Built-in network transformations enforcing exact no-slip wall conditions and axisymmetric pole regularity ($r=0$).
- **Hybrid Optimization Pipeline:** Warmup training with Adam followed by high-precision quasi-Newton L-BFGS refinement [@raissi2019physics].
- **Adaptive Particle Integration:** Trajectory tracking using SciPy `solve_ivp` (Dormand–Prince RK45) with exact surface-wall collision detection.
- **Nondimensionalization Engine:** Automatic conversion between physical SI units and dimensionless training domains to preserve numerical conditioning.
- **CUDA-Aware & Device-Agnostic:** Centralized `torch.device` resolution preventing cross-device or dtype tensor mismatches.

---

## Mathematical Formulation

### 1. Incompressible Axisymmetric Stokes Flow
In cylindrical coordinates $(r, z)$ under axisymmetry ($\partial_\theta \equiv 0$, $u_\theta \equiv 0$), the dimensionless momentum and continuity equations read:
$$\begin{aligned}
-\frac{\partial p}{\partial r} + \left( \nabla^2 u_r - \frac{u_r}{r^2} \right) &= 0, \\
-\frac{\partial p}{\partial z} + \nabla^2 u_z &= 0, \\
\frac{1}{r}\frac{\partial (r u_r)}{\partial r} + \frac{\partial u_z}{\partial z} &= 0.
\end{aligned}$$

### 2. Hard Wall and Axis Regularity Transformation
To eliminate non-physical axial flow decay caused by soft boundary loss residuals, outputs are transformed via:
$$u_r(r, z) = \left(\frac{r}{R}\right)\left(1 - \frac{r^2}{R^2}\right)\tilde{u}_r(r, z), \quad u_z(r, z) = \tilde{u}_z(r, z), \quad p(r, z) = \tilde{p}(r, z).$$

### 3. Overdamped Magnetic Particle Motion
Particles follow overdamped dynamics governed by local hydrodynamics and magnetic forces:
$$\frac{d\mathbf{x}_p}{dt} = \mathbf{u}_{\mathrm{Faxén}}(\mathbf{x}_p) + \mathbf{M}^* \cdot \mathbf{F}_{\mathrm{mag}}(\mathbf{x}_p),$$
where $\mathbf{M}^*$ is the dimensionless mobility tensor and $\mathbf{u}_{\mathrm{Faxén}}$ incorporates the local velocity Laplacian correction.

---

## Installation

### Editable Development Installation
Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/dfolio/pymagnetofluidics.git
cd pymagnetofluidics
pip install -e ".[dev]"
```

## Quick Start & Usage

Below is a minimal example demonstrating configuration, neural network construction, PINN training, and particle trajectory integration.

```python
import torch
import magnetofluidics_pinn as mfp

# 1. Define physical problem configurations
domain_cfg = mfp.DomainConfig(kind="channel", length=3.0e-3, radius=7.5e-4)
fluid_cfg = mfp.FluidConfig(
    regime="stokes",
    dynamic_viscosity=1.0e-3,
    density=1.0e3,
    reference_velocity=1.0e-3,
    reference_length=domain_cfg.radius,
)
field_cfg = mfp.FieldConfig(source="uniform", magnitude=1.0e-2, orientation=(1.0, 0.0))
particle_cfg = mfp.ParticleConfig(radius=10.0e-6, magnetic_moment=[1.0e-11, 0.0])

# 2. Derive scales and nondimensionalize domain
physical_domain = mfp.build_channel_domain(domain_cfg)
scales = mfp.compute_scales(fluid_cfg, field_cfg)
domain = mfp.nondimensionalize_domain(physical_domain, scales)

# 3. Construct MLP network with structural hard wall constraints
raw_net = mfp.build_mlp(n_inputs=2, n_outputs=3, hidden_layers=(32, 32, 32), activation="tanh")
network = mfp.apply_hard_wall_constraint(raw_net, domain)

# 4. Train PINN model (Adam Warmup + L-BFGS Refinement)
train_cfg = mfp.TrainingConfig(
    n_epochs=200,
    use_lbfgs_refinement=True,
    lbfgs_rounds=2,
    lbfgs_iterations_per_round=200,
)
trained_network, history = mfp.train(
    network=network,
    domain=domain,
    fluid_config=fluid_cfg,
    field_config=field_cfg,
    training_config=train_cfg,
    verbose=True,
)

# 5. Integrate particle trajectories
initial_state = mfp.ParticleState(
    position=torch.tensor([0.2 * domain.radius, 0.1 * domain.length]),
    velocity=torch.zeros(2),
    time=0.0,
)
field_fn = lambda x: mfp.uniform_field(x, magnitude=field_cfg.magnitude, orientation=field_cfg.orientation)
mobility = mfp.nondimensionalize_mobility(particle_cfg, fluid_cfg, scales).unsqueeze(0)

trajectories = mfp.integrate_trajectory(
    flow_network=trained_network,
    field_fn=field_fn,
    domain=domain,
    scales=scales,
    particle_config=particle_cfg,
    initial_states=[initial_state],
    mobility_tensor=mobility,
    time_span=(0.0, 2.0),
)

print(f"Trajectory integrated successfully with {len(trajectories[0])} steps.")
```

## Verification & Executable Notebooks

Quantitative verification against analytical solutions is maintained in Quarto executable notebooks:

- `notebooks/phase1.4_verification.qmd`: Full numerical validation of Phase 1 (v0.1.4+).
  Demonstrates exact residual evaluation, velocity profile invariance, radial leakage quenching ($|u_r| \approx 0$), mass flux conservation $Q(z)$, and particle streamline alignment.

To execute and render the verification notebook:

```bash
quarto render notebooks/phase1.4_verification.qmd --to html
```
## Package layout

```bash
├── config.py             # Dataclass configs (Domain, Fluid, Field, Particle, Training)
├── device_utils.py       # Centralized torch.device & dtype management
├── io_utils.py           # Safe checkpointing & architecture descriptors
├── scaling.py            # SI <-> Dimensionless unit conversion engine
├── types.py              # Immutable data containers (Domain, CollocationPoints, ParticleState)
├── geometry/             # Vessel builders (straight channel, bifurcations)
├── boundary_conditions/  # Flow profiles & prescribed magnetic fields
├── physics/              # Stokes residuals, dipole forcing, Faxén law
├── sampling/             # Collocation point generators (volume-uniform sampling)
├── networks/             # MLP construction & hard constraint wrappers
├── training/             # Loss composition & Adam/L-BFGS training orchestrator
├── trajectory/           # SciPy RK45 particle trajectory integrator
└── visualization/        # Streamline & trajectory matplotlib renderers
```

## Roadmap & Development Phases


- [x] **Phase 1** — single sphere, uniform $\mathbf{B}$, Stokes flow, straight
  channel — implemented and verified. Interior residual, boundary
  conditions, network, sampling, training (Adam + L-BFGS refinement
  [@raissi2019physics]), and particle trajectory integration
  (with an optional Faxén-law finite-size correction
  [@faxen1922widerstand; @kim2005microhydrodynamics; @maxey1983equation])
  are complete; see `notebooks/phase1.4_verification.qmd` for the current,
  executable verification against the analytical Poiseuille solution
  (`notebooks/phase1_verification.qmd` predates the present `integrate_trajectory`
  API and is kept only for history).

- [ ] **Phase 2**: Particle swarms (tens of microrobots), non-uniform magnetic gradients, dipolar inter-particle 
  interactions.

- [ ] **Phase 3**: Unsteady Navier–Stokes regime, bifurcated geometries, dynamic time-varying magnetic actuation.

- [ ] **Future Horizons**: Extension to Fourier Neural Operators (FNO) [@li2021fno; @li2021fourier] for complex 
fluidic circuits.

## References
