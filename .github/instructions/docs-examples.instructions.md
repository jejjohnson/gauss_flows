---
applyTo: "docs/notebooks/*.ipynb,docs/**/*.md"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **executed `.ipynb` files only**. There are no jupytext `.py` pairs — the `.ipynb` is the single source of truth for code, prose, and rendered figures.

**Execution model**: `mkdocs-jupyter` renders notebooks with `execute: false`. Authors execute the `.ipynb` locally so cell outputs (including inline figures) are committed alongside the source.

## Directory Layout

```
docs/
├── images/
│   └── {notebook_name}/      # one subdirectory per notebook (only if savefig is used)
├── notebooks/
│   ├── demo_foo.ipynb        # executed notebook (outputs + inline figures)
│   └── benchmark_bar.ipynb
└── guide.md                  # markdown page embedding rendered notebook outputs
```

## Notebook Structure

Every example notebook should follow this order:

1. **Title & overview** (markdown) — what the notebook demonstrates, prerequisites
2. **Imports** (code)
3. **Problem setup** (markdown + code) — data, parameters
4. **Core computation** (markdown + code) — the actual demonstration
5. **Figures & tables** (code) — `plt.show()` inline; outputs are embedded in the executed `.ipynb`
6. **Summary / takeaways** (markdown)

## Figures

For inline rendering via `mkdocs-jupyter`, use `plt.show()` and commit the executed `.ipynb` (which contains the cell outputs). Do **not** `savefig` + embed-as-markdown unless there is a specific reason — the `.ipynb` cell outputs are the single source of rendered figures.

## Matplotlib Backend

In a Jupyter kernel the inline backend is selected automatically when the cell containing `plt.show()` runs, so no explicit backend selection is needed. If you re-execute via `jupyter nbconvert --to notebook --execute` the same default applies. Avoid `matplotlib.use("Agg")` — it is non-interactive and prevents inline figures from being captured into cell outputs.

## Re-executing a Notebook

```
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/foo.ipynb
```

Then commit the updated `.ipynb` (cell outputs included).

## Checklist for New Notebooks

- [ ] Notebook executed in-place (cell outputs embedded)
- [ ] At least one `plt.show()` (or trailing-`fig` expression) per plotting cell
- [ ] No `matplotlib.use("Agg")` calls
- [ ] No `savefig` and no separate PNG files
- [ ] Notebook listed in `mkdocs.yml` nav
