---
applyTo: "docs/**/*.py,docs/**/*.md,notebooks/**/*.py"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **jupytext percent-format `.py` files**. They are the single source of truth for all figures, tables, and timing data shown in the documentation.

**Execution model**: `mkdocs-jupyter` renders notebooks with `execute: false`. Authors write the `.py` source, execute locally to produce an `.ipynb` with inline outputs, and commit **both** the `.py` source and the executed `.ipynb`.

## Directory Layout

```
docs/
├── images/
│   └── {notebook_name}/      # one subdirectory per notebook (for savefig PNGs if used)
├── notebooks/
│   ├── demo_foo.py            # jupytext percent-format source
│   ├── demo_foo.ipynb         # executed notebook (outputs + inline figures)
│   └── benchmark_bar.py
└── guide.md                   # markdown page embedding saved images
```

## Jupytext Header

Every notebook `.py` file must start with this header:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Cell Markers

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import jax.numpy as jnp
```

## Notebook Structure

Every example notebook should follow this order:

1. **Title & overview** (markdown) — what the notebook demonstrates, prerequisites
2. **Imports** (code)
3. **Problem setup** (markdown + code) — data, parameters
4. **Core computation** (markdown + code) — the actual demonstration
5. **Figures & tables** (code) — `plt.show()` inline; outputs are embedded in the executed `.ipynb`
6. **Summary / takeaways** (markdown)

## Paragraph Formatting (important)

Each markdown paragraph must be a **single long line** inside the `.py` source — do NOT wrap prose at 80 chars. Jupytext treats hard line breaks as within-paragraph breaks, and wrapped lines cause weird line-break rendering in the executed notebook.

```python
# %% [markdown]
# ## Section heading
#
# This is one paragraph written on a single long line. Markdown renderers soft-wrap it to the viewport; do not hard-wrap it in the source or jupytext will insert unwanted breaks.
#
# This is a second paragraph. Same rule — one physical line per paragraph.
```

## Figures

For inline rendering via `mkdocs-jupyter`, use `plt.show()` and commit the executed `.ipynb` (which contains the cell outputs). Do **not** `savefig` + embed-as-markdown unless there is a specific reason — the `.ipynb` cell outputs are the single source of rendered figures.

## Matplotlib Backend

Use the non-interactive backend at the top of every notebook to avoid display issues in CI or headless environments:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

## Checklist for New Notebooks

- [ ] Jupytext header present
- [ ] `matplotlib.use("Agg")` before any `plt` import
- [ ] Each markdown paragraph is a single long line (no hard-wrap)
- [ ] Notebook executed locally (`jupytext --to notebook --execute foo.py -o foo.ipynb`)
- [ ] Both `.py` and `.ipynb` committed
- [ ] Notebook listed in `mkdocs.yml` nav
