# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gauss_flows: JAX/FlowJax-based Gaussianization flows, SurVAE bijections/surjections/stochastic transforms, and information-theoretic utilities (entropy, mutual information, total correlation, KL, negentropy) — all compatible with NumPyro for Bayesian inference.

## Architecture

### Three-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Transforms | `MixtureGaussianCDF`, `HouseholderRotation`, `RQSplineCoupling`, `Invertible1x1Conv`, … (bijections on the FlowJax backbone) |
| 1 | Flow factories + NumPyro integration | `gaussianization_flow`, `coupling_gaussianization_flow`, `iterative_rbig`, `FlowDist`, `FlowGuide` |
| 2 | Training + Information theory | `fit_gaussianization_flow`, `entropy`, `total_correlation`, `mutual_information`, `kl_divergence`, `negentropy` |

### Package structure

All implementation lives in `src/gauss_flows/`. The public API is re-exported through `src/gauss_flows/__init__.py`. Private implementation is in `src/gauss_flows/_src/`.

### Key directories

| Path | Purpose |
|------|---------|
| `src/gauss_flows/` | Main package source code |
| `src/gauss_flows/_src/transforms/` | Layer 0 — bijections, surjections, stochastic transforms |
| `src/gauss_flows/_src/flows.py` | Layer 1 — flow factories (`gaussianization_flow`, etc.) |
| `src/gauss_flows/_src/numpyro_compat.py` | Layer 1 — `FlowDist` NumPyro distribution wrapper |
| `src/gauss_flows/_src/numpyro_guide.py` | Layer 1 — `FlowGuide` variational guide |
| `src/gauss_flows/_src/train.py` | Layer 2 — training wrapper around `flowjax.train.fit_to_data` |
| `src/gauss_flows/_src/info_theory.py` | Layer 2 — information-theoretic measures |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `docs/notebooks/` | Example notebooks (jupytext percent-format `.py` + executed `.ipynb`) |

### Key dependencies

| Package | Role |
|---------|------|
| `jax` / `jaxlib` | Array backend, autodiff |
| `equinox` | Module system, PyTrees, `filter_vmap` |
| `flowjax` | `AbstractBijection`, `Transformed`, `Scan`, `fit_to_data` |
| `numpyro` | Distribution base, `AutoContinuous`, SVI, MCMC |
| `optax` | Optimization (via `flowjax.train`) |
| `paramax` | Parameter constraints |
| `jaxtyping` | Array type annotations |

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests with coverage: uv run pytest -v
make test-fast            # Skip slow-marked tests
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/gauss_flows
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_flows.py::test_gaussianization_flow -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                                    # Tests
uv run --group lint ruff check .                    # Lint — ENTIRE repo, not just src/gauss_flows/
uv run --group lint ruff format --check .           # Format — ENTIRE repo
uv run --group typecheck ty check src/gauss_flows   # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/gauss_flows/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Coding Conventions

- All transforms are `equinox.Module` subclasses (immutable, PyTree-compatible)
- All transforms subclass `flowjax.bijections.AbstractBijection` or a SurVAE base
- Use `eqx.filter_vmap` to build layer stacks; compose with FlowJax `Scan` (memory-efficient for 100+ layers)
- Shape annotations via `jaxtyping` (`Float[Array, "D"]`, etc.)
- Google-style docstrings
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Execute locally via `jupytext --to notebook --execute foo.py -o foo.ipynb`
3. Commit both the `.py` source and the executed `.ipynb` (which contains inline figure outputs)
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).
