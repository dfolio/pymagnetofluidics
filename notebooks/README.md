# PINN Microfluidics Verification Suite

This directory contains the **refactored Quarto notebooks** for verifying the Physics-Informed Neural Network (PINN) implementation for modeling viscous microfluidic flows with two-way particle-flow coupling.

## Overview

The original monolithic notebook `KCwDVRja-two_way_coupling_verification_2026-11-02.md` has been split into **13 focused notebooks** organized into **7 directories** to improve modularity, maintainability, and verification clarity.

## Directory Structure

```
notebooks/
├── 00_configuration/           # Shared configuration and utilities
│   └── 00_configuration.qmd
│
├── 01_baseline_flow/           # Baseline Hagen-Poiseuille flow
│   ├── 01_baseline_training.qmd
│   └── 01_baseline_verification.qmd
│
├── 02_particle_flow/           # Particle-perturbed flow
│   ├── 02_particle_training.qmd
│   └── 02_particle_verification.qmd
│
├── 03_resistance_problems/     # Resistance networks for force calculation
│   ├── 03_resistance_training.qmd
│   ├── 03_resistance_validation.qmd
│   └── 03_resistance_gates.qmd
│
├── 04_warm_start/              # Warm-start verification
│   └── 04_warm_start_verification.qmd
│
├── 05_trajectory/              # Free-particle trajectory with animation
│   └── 05_free_particle.qmd
│
└── 06_master_report/           # Master verification report
    └── 06_verification_summary.qmd
```

## Execution Order

The notebooks should be executed in the following order:

1. **00_configuration/00_configuration.qmd** - Loads shared parameters (imported by all others)
2. **01_baseline_flow/01_baseline_training.qmd** - Trains baseline network
3. **01_baseline_flow/01_baseline_verification.qmd** - Verifies baseline network
4. **02_particle_flow/02_particle_training.qmd** - Trains particle network
5. **02_particle_flow/02_particle_verification.qmd** - Verifies particle network
6. **03_resistance_problems/03_resistance_training.qmd** - Trains resistance networks
7. **03_resistance_problems/03_resistance_validation.qmd** - Validates resistance networks
8. **03_resistance_problems/03_resistance_gates.qmd** - Runs all physical gates
9. **04_warm_start/04_warm_start_verification.qmd** - Verifies warm-start capability
10. **05_trajectory/05_free_particle.qmd** - Computes trajectories and generates animation
11. **06_master_report/06_verification_summary.qmd** - Aggregates all verification results

## Network Dependencies

```
01_baseline_flow → baseline_network checkpoint
    ↓
02_particle_flow → particle_network checkpoint
    ↓
03_resistance_problems → network_ambient + network_translating checkpoints
    ↓
04_warm_start → (uses particle_network from 02)
    ↓
05_trajectory → (uses resistance networks from 03)
    ↓
06_master_report → (aggregates all checks)
```

## Verification Categories

All verification checks are categorized into one of three types:

| Category | Description | Examples |
|----------|-------------|----------|
| **VERIFICATION** | Mathematical properties and manufactured results | Continuity, no-slip, residual convergence |
| **CONVERGENCE** | Numerical stability with respect to resolution/training | Quadrature convergence, probe gap, training loss |
| **VALIDATION** | Comparison with reduced physical models | Force accuracy, Faxén-limit, momentum balance |

## Key Features

### 1. Modular Design
- Each notebook has a **single responsibility**
- **Explicit dependencies** through checkpoint files
- **No code duplication** - shared configuration in 00_configuration

### 2. Comprehensive Verification
- **100+ verification checks** across all notebooks
- **Automated execution** - checks run as part of notebook execution
- **Reproducible results** - fixed random seeds and deterministic operations

### 3. Documentation
- **Google-style docstrings** for all functions
- **LaTeX equations** compatible with MathJax
- **Inline Python code** in Quarto markdown (no hard-coded values)
- **Detailed prose** explaining each verification check

### 4. Visualization
- **Loss curves** for training monitoring
- **Field visualizations** (velocity, pressure, residuals)
- **Profile comparisons** against analytical solutions
- **Animations** of particle trajectories

## Checkpoint Files

All trained networks are saved to the `checkpoints/` directory:

| Checkpoint | Source Notebook | Purpose |
|------------|-----------------|---------|
| `standard_baseline.pt` | 01_baseline_training | Undisturbed Hagen-Poiseuille flow |
| `r_weighted_fixed_sphere.pt` | 02_particle_training | Fixed particle in flow |
| `r_weighted_resistance_ambient.pt` | 03_resistance_training | Fixed sphere force (F₀) |
| `r_weighted_resistance_translating.pt` | 03_resistance_training | Unit translation force (F₁) |
| `r_weighted_fixed_sphere_cold.pt` | 04_warm_start | Cold-start reference for warm-start |
| `fixed_sphere_warm_shifted.pt` | 04_warm_start | Warm-start network |

## Configuration

All shared parameters are defined in `00_configuration/00_configuration.qmd`:

- **Physical parameters**: Channel geometry, fluid properties, particle dimensions
- **Training hyperparameters**: Network architecture, optimizer settings, training budget
- **Verification thresholds**: Tolerances for all verification checks
- **Evaluation stations**: Shared locations for profile and flow rate evaluations
- **Utility functions**: Common operations used across notebooks

## Design Principles

1. **Single Source of Truth**: All parameters defined once in 00_configuration
2. **Immutability**: Configuration objects are not modified after creation
3. **Type Safety**: All values have explicit types and units
4. **Documentation**: Every parameter and function has a docstring
5. **Validation**: Configuration is validated at import time
6. **Functional Programming**: Pure functions, data immutability, zero side effects
7. **PEP 8 Compliance**: Standard Python naming conventions throughout

## Usage

### Running the Full Suite

```bash
# Run all notebooks in order
cd notebooks

# Start with configuration
quarto render 00_configuration/00_configuration.qmd

# Then baseline flow
quarto render 01_baseline_flow/01_baseline_training.qmd
quarto render 01_baseline_flow/01_baseline_verification.qmd

# Continue with particle flow
quarto render 02_particle_flow/02_particle_training.qmd
quarto render 02_particle_flow/02_particle_verification.qmd

# Resistance problems
quarto render 03_resistance_problems/03_resistance_training.qmd
quarto render 03_resistance_problems/03_resistance_validation.qmd
quarto render 03_resistance_problems/03_resistance_gates.qmd

# Warm-start verification
quarto render 04_warm_start/04_warm_start_verification.qmd

# Trajectory and animation
quarto render 05_trajectory/05_free_particle.qmd

# Master report
quarto render 06_master_report/06_verification_summary.qmd
```

### Running Individual Notebooks

Each notebook can be run independently, provided its dependencies are satisfied:

```bash
# Example: Run particle verification (requires baseline and particle networks)
quarto render 02_particle_flow/02_particle_verification.qmd
```

## Verification Summary

The master report (`06_master_report/06_verification_summary.qmd`) provides a complete summary of all verification checks, including:

- **Total checks**: 100+ across all notebooks
- **Categories**: VERIFICATION, CONVERGENCE, VALIDATION
- **Pass/fail status**: For each check
- **Visualization**: Bar charts and pie charts of verification status
- **Export**: CSV file with all check results

## File Size and Performance

| Notebook | Approx. Size | Estimated Runtime |
|----------|--------------|-------------------|
| 00_configuration | ~20 KB | < 1s |
| 01_baseline_training | ~14 KB | ~5-10 min |
| 01_baseline_verification | ~19 KB | ~2-5 min |
| 02_particle_training | ~19 KB | ~25 min |
| 02_particle_verification | ~25 KB | ~5-10 min |
| 03_resistance_training | ~16 KB | ~50 min (2 networks) |
| 03_resistance_validation | ~17 KB | ~5-10 min |
| 03_resistance_gates | ~16 KB | ~5-10 min |
| 04_warm_start | ~20 KB | ~30-40 min |
| 05_trajectory | ~19 KB | ~5-10 min |
| 06_master_report | ~18 KB | < 1s |

**Total estimated runtime**: ~3-4 hours (depending on hardware)

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8 GB+ VRAM for training
- **CPU**: Multi-core processor for data processing
- **Storage**: ~500 MB for checkpoints and outputs

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- magnetofluidics_pinn (local package)
- numpy, pandas, matplotlib
- Quarto for rendering notebooks

## References

- [Quarto Documentation](https://quarto.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [magnetofluidics_pinn Package](https://github.com/dfolio/pymagnetofluidics)

## License

This verification suite is part of the pymicrorobotics project and is licensed under the same terms.

## Contact

For questions or issues, please refer to the main project repository: [dfolio/pymicrorobotics](https://github.com/dfolio/pymicrorobotics)
