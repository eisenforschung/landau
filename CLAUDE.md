# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**landau.py** is a Python library for calculating thermodynamic equilibria and plotting phase diagrams in the (semi-)grand ensemble. It enables computational thermodynamics research, particularly for alloy systems.

## Development Setup

### Python Version
- **Supported**: Python 3.11, 3.12, 3.13
- **Required**: Python >=3.11,<3.14 (from pyproject.toml)

### Installation

```bash
# Development install with all test dependencies
pip install -e .[test,constraints,fast-tsp,python-tsp]

# Minimal install
pip install -e .

# Build distributions
python -m build --sdist
```

### Code Style

The project uses **Ruff** and **Flake8** for code quality:
- **Max line length**: 120 characters
- Configuration: `.flake8` and `pyproject.toml` (`[tool.ruff]` section)

## Testing

### Running Tests

```bash
# All tests
pytest

# Single test file
pytest tests/unit/test_calculate.py

# Single test function
pytest tests/unit/test_calculate.py::test_function_name

# Tests matching a pattern
pytest -k "pattern_name"

# With verbose output
pytest -v
```

### Test Organization

- **Unit tests**: `tests/unit/` — test individual components
- **Regression tests**: `tests/regression/` — test against known issues (e.g., pandas groupby compatibility)
- **Property-based tests**: Uses **Hypothesis** for generating test cases

## Architecture

### Core Package Structure

**`landau/phases.py`** — Phase definitions (abstract base classes and implementations)
- `Phase` — Abstract base for all phases
- `AbstractLinePhase` — Base for fixed-concentration phases
- `LinePhase`, `TemperatureDependentLinePhase` — Simple line phases
- `IdealSolution`, `RegularSolution` — Solution models
- `InterpolatingPhase`, `SlowInterpolatingPhase` — Interpolation-based phases
- `AbstractPointDefect`, `ConstantPointDefect`, `PointDefectSublattice`, `PointDefectedPhase` — Point defect handling

**`landau/interpolate/`** — Subpackage for interpolation methods
- `interpolate/basic.py` — Core interpolators
  - `Interpolator` — Base class
  - `TemperatureInterpolator`, `ConcentrationInterpolator` — Dimension-specific interpolators
  - `PolyFit` — Polynomial fitting
  - `SGTE` — Gibbs energy standard model
  - `RedlichKister` — Thermodynamic model implementation
  - `StitchedFit` — Combines multiple interpolators
- `interpolate/softplus.py` — Smoothed step interpolation
  - `SoftplusFit` — Neural network-inspired smooth transitions

**`landau/calculate.py`** — Core thermodynamic calculations

**`landau/plot.py`** — Phase diagram plotting
- `plot_phase_diagram()` — Main plotting function

**`landau/poly.py`** — Polynomial utilities and helpers

**`landau/resample.py`** — Data resampling and interpolation utilities

### Key Design Patterns

1. **Abstract Base Classes with dataclasses**: Phases use `@dataclass(frozen=True)` with ABC for immutability and interface definition
2. **Interpolator Abstraction**: Multiple interpolation strategies (SGTE, PolyFit, RedlichKister, SoftplusFit) implement a common interface
3. **Semi-grand Potential**: Core thermodynamic property calculated by phases for equilibrium determination

## Key Dependencies

- **Scientific computing**: numpy, scipy, scikit-learn
- **Data handling**: pandas (>=2.2), shapely
- **Visualization**: matplotlib, seaborn
- **Infrastructure**: pyiron_snippets
- **Optional**: polyfit, python-tsp, fast-tsp (for TSP-based optimization)

### Recent Important Changes

- **Pandas 3.0 compatibility fix**: Replaced deprecated `include_groups=True` in `groupby.apply` (commit ae0a455)
- **Interpolate subpackage split**: Recent refactoring split interpolation into its own subpackage (commit d7f679a)

## Documentation

- **Sphinx-based**: Located in `docs/` directory
- **Build tool**: ReadTheDocs (Python 3.12)
- **Extensions**: myst-nb (Jupyter notebook integration), sphinx-autodoc-typehints

## Git Conventions

- Main branch: `main`
- Use conventional commits when possible
- Reference issues in commit messages

## Publication & Citation

If modifying this code for research, cite:
```
@article{poul2025automated,
    title = {Automated generation of structure datasets for machine learning potentials and alloys},
    journal = {npj Computational Materials},
    author={Poul, Marvin and Huber, Liam and Neugebauer, Jörg},
    volume = {11},
    number = {1},
    pages = {174},
    year={2025},
    doi = {10.1038/s41524-025-01669-4},
}
```

## Common Tasks

### Adding a New Phase Type

1. Inherit from `Phase` or `AbstractLinePhase`
2. Implement `semigrand_potential()` and `concentration()` methods
3. Add to `landau/__init__.py` exports
4. Add tests in `tests/unit/` or `tests/regression/`

### Adding an Interpolation Method

1. Create a class inheriting from `Interpolator` in `landau/interpolate/basic.py`
2. Implement the interpolation interface
3. Export from `landau/interpolate/__init__.py`
4. Add comprehensive tests with various data inputs

### Running Documentation Locally

```bash
pip install -e .[docs]
cd docs
sphinx-build -b html . _build/html
```
