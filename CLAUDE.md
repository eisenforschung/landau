# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**landau.py** is a Python library for calculating thermodynamic equilibria and plotting phase diagrams in the (semi-)grand ensemble. It enables computational thermodynamics research, particularly for alloy systems.

## Working Style (read this first)

These are project-specific preferences distilled from past PR/issue feedback. Some restate harness defaults but are repeated here because they have been violated often enough to matter.

### Tone in PRs, commits, and comments
- Match a terse, imperative tone. No "Happy to...", no apologies, no enthusiasm filler. (Restates the harness default — keep it in mind anyway.)
- PR bodies are plain technical reports. No **bold** "headline numbers", no "Conclusion" sections, no `[Fix this →]` action links, no marketing claims ("dramatic speedup", "significantly improves").
- Do not invoke the maintainer's name or authority in PR bodies or comments (PR #94: "don't use my name in vain").

### Evidence over claims
- Cite the specific commit hash, file path, test count, or command output backing any claim. If a number appears in the PR body, the script that produced it goes in `benchmarks/` in the same PR (PR #94).
- For physics/numerics claims, prefer saying nothing over saying something unverified (PR #112: "Rather say nothing than something wrong").
- If asked to verify, actually verify — read the docs, run the code. Do not re-assert. (PR #113: "I'm not entirely sure I believe you. Read the docs and get back to me.")

### Scope and abstraction
- Prefer the minimal change. "I'd rather merge less features now and extend later than break API" (PR #68).
- Before writing a new helper, check the codebase and existing deps (numpy, scipy, shapely, matplotlib.testing, pyiron_snippets) for one that already does the job. Reuse over reimplement (PR #115).
- One purpose per PR. Notebooks, benchmarks, refactors, and unrelated fixes go in their own PRs (PR #68, #82, #85). Notebooks are committed with executed outputs only.
- When a PR grows a second concept, propose splitting it before pushing more.
- Speculative or "might be useful" work gets parked, not landed (PR #129 closed, #130 parked).

### Test quality bar
(Expands on the Testing section below.)
- Tests must assert tight conditions a degenerate/constant solution would fail. Loose `atol=0.05` "it ran" tests get rejected (PR #80, #101).
- Use Hypothesis for round-trip recovery (random coeffs in → `fit()` → coeffs out) where applicable (PR #82).
- Share a single `atol` constant across related tests rather than scattering magic numbers (PR #82).
- Use the module-wide pytest skip marker for optional-dep tests (PR #111).

### Debugging numerics visually
- When diagnosing a fit/solver/physics issue, generate a plot of the relevant landscape (loss vs parameter, etc.), attach it inline to the PR comment, and add a walkthrough section to a notebook so the maintainer can step through interactively (PR #82, #115).

### Naming and layout
- Module names: no underscores. `asewrapper`, not `ase_wrapper` (PR #68).
- Promote single-file modules to subpackages once they grow (as was done for `interpolate/`).

### Do not commit
- `.hypothesis/`, `_version.py`, stray top-level scripts, duplicate exploratory files (PR #71, #94). Check for duplicates of what you are about to add before pushing.

### Pandas 2/3 compatibility (hard constraint)
(Historical context is in "Recent Important Changes" below.)
- All `groupby().apply()` calls must pass `include_groups=False`.
- Code must work on both pandas 2 and 3. Do not drop pandas 2 (PR #93, #113).

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
