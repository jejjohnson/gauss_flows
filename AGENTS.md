# Agent Instructions

## Before Every Commit

Always ensure all of the following pass before committing:

1. **Tests** — `uv run pytest tests/ -m "not slow"`
2. **Type checks** — `uv run mypy gauss_flows/ --ignore-missing-imports`
3. **Format** — `uv run ruff format gauss_flows/ tests/`
4. **Lint** — `uv run ruff check gauss_flows/ tests/`

## PR Title and Description

Keep the original PR title and description intact across sessions. When new work is done, **incrementally add** to the existing PR description rather than replacing it. Never overwrite the original intent of the PR with a new title or description.

## Commit and PR Naming

Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages and PR titles:

```
<type>(<optional scope>): <short description>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`.

Examples:
- `feat: add FlowGuide variational guide for NumPyro SVI`
- `fix: add missing support attribute to FlowDist`
- `docs: update README with FlowGuide usage example`
- `test: add SVI smoke tests for FlowGuide`
