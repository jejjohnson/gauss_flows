---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `dataclasses` or `attrs` for data containers
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## JAX & FlowJax Conventions (project-specific)

- All transforms are `equinox.Module` subclasses (immutable, PyTree-compatible)
- All transforms subclass `flowjax.bijections.AbstractBijection`
  (or one of the SurVAE bases in `src/gauss_flows/_src/transforms/base.py`)
- Shape annotations via `jaxtyping` (e.g. `Float[Array, "D"]`)
- Use `eqx.filter_vmap` to create layer stacks; compose with FlowJax `Scan`
- Pure functions where possible; side effects isolated and explicit
- Use `einops` for non-trivial reshape/einsum operations
- **Transform methods operate on a single event, not batches.** The abstract methods
  (`transform_and_log_det`, `inverse_and_log_det`, `forward_and_log_det`) assume the
  input has `shape == self.shape` and return a scalar `log_det`. Do **not** add
  internal `vmap`/flatten/reshape machinery to support a leading batch axis — callers
  vectorise explicitly with `jax.vmap` / `eqx.filter_vmap`, and the `SurVAEFlow`
  container already vmaps `log_prob` / `sample` over any leading `sample_shape`. This
  mirrors flowjax's `AbstractBijection` convention (it runtime-enforces
  `x.shape == bijection.shape`).

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings
  (e.g. `# H(X) = −E_p[log p(X)]`)
- Public classes and functions should include 2–3 example use cases in docstrings
