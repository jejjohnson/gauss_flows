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

# %% [markdown]
# # FFJORD on the two-moons density
#
# This notebook trains a [FFJORD](https://arxiv.org/abs/1810.01367) continuous
# normalizing flow on the canonical two-moons toy. FFJORD parameterises the
# density transform as the solution of an ODE,
#
# $$\frac{dx}{dt} = f_\theta(t, x), \qquad \frac{d\log\,|\det J|}{dt} = \mathrm{tr}(\partial f_\theta/\partial x),$$
#
# integrated over $[t_0, t_1]$ with `diffrax`. The trace can be computed
# **exactly** with `jax.jacfwd` (cost $O(\dim)$ per ODE step) or stochastically
# with **Hutchinson's identity** via `matfree` (cost $O(\text{n\_samples})$ per
# step). The Hutchinson probes in `gauss_flows`'s FFJORD are *fixed per
# instance* — see the docstring — so the log-determinant is a deterministic
# function of parameters, which stabilises gradient training.
#
# We follow the standard recipe: train with cheap **Hutchinson** divergence,
# evaluate with the **exact** trace by re-instantiating with the same trained
# vector field.

# %%
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from flowjax.distributions import Normal, Transformed
from flowjax.train import fit_to_data
from sklearn.datasets import make_moons

from gauss_flows import FFJORD, DiffeqMLP

# --- global plot styling ------------------------------------------------------
sns.set_theme(context="poster", style="whitegrid", palette="deep", font_scale=0.85)
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "axes.grid.which": "both",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.5,
        "axes.titleweight": "semibold",
    }
)


def style_axes(ax, *, aspect=None):
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)
    if aspect is not None:
        ax.set_aspect(aspect)
    return ax


palette = sns.color_palette("deep")

# %% [markdown]
# ## 1. Dataset
#
# A standardised two-moons sample. Standardisation matters because the FFJORD
# base distribution is a unit-variance Normal — without it, training has to
# absorb the scale through the vector field.

# %%
n_samples = 1000
X_raw, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
X = jnp.asarray(X, dtype=jnp.float32)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.asarray(X[:, 0]), np.asarray(X[:, 1]), s=10, alpha=0.6, color=palette[0])
ax.set_title("Two moons (standardised)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
style_axes(ax, aspect="equal")
plt.show()

# %% [markdown]
# ## 2. Build a FFJORD bijection
#
# The vector field is a small `DiffeqMLP` that takes the ODE time and state and
# returns $dx/dt$. The bijection is unconditional (`control_dim=0`), so no
# packed condition is required at call time and `bij.cond_shape is None`.
#
# For a 2D problem, **exact** divergence costs only two Jacobian-vector products
# per ODE step — comparable to Hutchinson with two probes — and avoids the
# trace estimator's variance entirely. We use it for both training and
# evaluation here. (Hutchinson becomes the right choice as $\dim$ grows; see
# section 7 for a comparison.)

# %%
key = jr.key(0)
vf_key, ffjord_key, train_key, sample_key = jr.split(key, 4)

vector_field = DiffeqMLP(
    vf_key,
    in_dim=2,
    control_dim=0,
    hidden=(64, 64),
)

ffjord = FFJORD(
    ffjord_key,
    shape=(2,),
    vector_field=vector_field,
    control_dim=0,
    divergence_mode="exact",
    solver="tsit5",
    adjoint="recursive_checkpoint",
    rtol=1e-4,
    atol=1e-4,
)

dist = Transformed(Normal(jnp.zeros(2)), ffjord)

n_params = sum(
    int(np.prod(p.shape))
    for p in jax.tree_util.tree_leaves(eqx.filter(vector_field, eqx.is_array))
)
print("unconditional FFJORD on shape (2,)")
print("  vector field        : DiffeqMLP, hidden=(64, 64), tanh")
print(f"  trainable parameters: {n_params}")
print("  divergence mode     : exact (2 JVPs / step in 2D)")
print(f"  cond_shape          : {ffjord.cond_shape}")

# %% [markdown]
# ## 3. Train
#
# `fit_to_data` minimises the negative log-likelihood under the flow. Each
# gradient step requires a vmapped ODE solve over the batch — FFJORD is
# expensive compared with closed-form flows. We keep the run short (30 epochs,
# batch 256) for a notebook; for a real fit you would train longer.

# %%
optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.adam(learning_rate=1e-3),
)

trained_dist, losses = fit_to_data(
    train_key,
    dist,
    X,
    optimizer=optimizer,
    max_epochs=80,
    max_patience=15,
    batch_size=128,
    val_prop=0.1,
    show_progress=False,
)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
epochs = jnp.arange(len(losses["train"]))
ax.plot(epochs, losses["train"], label="train", color=palette[0])
ax.plot(epochs, losses["val"], label="val", color=palette[1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("FFJORD training curve")
ax.legend()
style_axes(ax)
plt.show()

# %% [markdown]
# ## 4. Tighten the evaluation solver
#
# Training used loose tolerances ($10^{-4}$) for speed; for clean density plots
# we re-instantiate with the **same trained vector field** and tighten the
# solver. We also switch to the `direct` adjoint, which is more accurate at
# evaluation time when we don't need backward gradients.

# %%
trained_ffjord = trained_dist.bijection
exact_ffjord = FFJORD(
    jr.key(99),  # trace_key is unused in exact mode
    shape=(2,),
    vector_field=trained_ffjord.vector_field,
    control_dim=0,
    divergence_mode="exact",
    solver="tsit5",
    adjoint="direct",
    rtol=1e-6,
    atol=1e-6,
)
exact_dist = Transformed(Normal(jnp.zeros(2)), exact_ffjord)

# %% [markdown]
# ## 5. Density and samples
#
# Plot the learned density on a grid (using the exact-mode flow) alongside
# samples drawn from the trained flow.

# %%
grid_n = 60
lim = 2.5
xs = jnp.linspace(-lim, lim, grid_n)
grid_x, grid_y = jnp.meshgrid(xs, xs)
grid_pts = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


@eqx.filter_jit
def grid_log_prob(pts):
    return jax.vmap(exact_dist.log_prob)(pts)


log_probs = grid_log_prob(grid_pts).reshape(grid_n, grid_n)
density = jnp.exp(log_probs)

flow_samples = trained_dist.sample(sample_key, (n_samples,))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_d, ax_s = axes
mesh = ax_d.contourf(
    np.asarray(grid_x),
    np.asarray(grid_y),
    np.asarray(density),
    levels=20,
    cmap="viridis",
)
ax_d.set_title("Learned density (exact trace)")
ax_d.set_xlabel("$x_1$")
ax_d.set_ylabel("$x_2$")
fig.colorbar(mesh, ax=ax_d, label="$p_\\theta(x)$")
style_axes(ax_d, aspect="equal")

ax_s.scatter(
    np.asarray(X[:, 0]),
    np.asarray(X[:, 1]),
    s=12,
    alpha=0.5,
    color=palette[0],
    label="data",
    edgecolors="none",
)
ax_s.scatter(
    np.asarray(flow_samples[:, 0]),
    np.asarray(flow_samples[:, 1]),
    s=12,
    alpha=0.5,
    color=palette[3],
    label="flow samples",
    marker="x",
    linewidths=0.8,
)
ax_s.set_xlim(-lim, lim)
ax_s.set_ylim(-lim, lim)
ax_s.set_title("Data vs. flow samples")
ax_s.set_xlabel("$x_1$")
ax_s.set_ylabel("$x_2$")
ax_s.legend(loc="upper right", framealpha=0.9)
style_axes(ax_s, aspect="equal")
plt.show()

# %% [markdown]
# ## 6. Vector field at three times
#
# The vector field $f_\theta(t, x)$ is what the ODE integrates. Visualising it
# as a quiver plot at a few times shows how the flow morphs the base Gaussian
# into the data distribution. At $t=0$ the trajectory starts (data side); at
# $t=1$ it arrives at the base side (since `inverse_and_log_det` integrates
# from $t_1$ back to $t_0$, "data → base" corresponds to $t \to 1$).

# %%
quiver_n = 18
qs = jnp.linspace(-lim, lim, quiver_n)
qx, qy = jnp.meshgrid(qs, qs)
q_pts = jnp.stack([qx.ravel(), qy.ravel()], axis=-1)

trained_vf = trained_ffjord.vector_field
times = (0.0, 0.5, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
for ax, t in zip(axes, times, strict=True):
    velocities = jax.vmap(lambda p: trained_vf(t, p, None))(q_pts)
    u = np.asarray(velocities[:, 0]).reshape(quiver_n, quiver_n)
    v = np.asarray(velocities[:, 1]).reshape(quiver_n, quiver_n)
    speed = np.sqrt(u * u + v * v)
    ax.scatter(
        np.asarray(X[:, 0]),
        np.asarray(X[:, 1]),
        s=6,
        alpha=0.35,
        color="0.45",
        edgecolors="none",
        zorder=1,
    )
    ax.quiver(
        np.asarray(qx),
        np.asarray(qy),
        u,
        v,
        speed,
        cmap="magma",
        width=0.004,
        zorder=2,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"$f_\\theta(t={t}, x)$")
    ax.set_xlabel("$x_1$")
    style_axes(ax, aspect="equal")
axes[0].set_ylabel("$x_2$")
plt.show()

# %% [markdown]
# ## 7. Hutchinson vs. exact: how many probes do you need?
#
# `gauss_flows`'s FFJORD uses **fixed** Hutchinson probes per instance. That
# stabilises training (the log-det is a deterministic function of parameters)
# but the estimate still carries the variance of a finite probe set. The mean
# over many ODE steps converges to the trace, so error scales as
# $\mathcal{O}(1/\sqrt{n_\text{samples}})$.
#
# Here we hold the trained vector field fixed and rebuild FFJORD with various
# probe counts, then compute the **bias** (mean difference vs. exact) and
# **RMSE** of $\log p$ on a held-out batch.

# %%
eval_pts = X[:200]
exact_lp_eval = jax.vmap(exact_dist.log_prob)(eval_pts)


def hutch_eval_log_prob(n_samples: int, key_seed: int) -> jnp.ndarray:
    bij = FFJORD(
        jr.key(key_seed),
        shape=(2,),
        vector_field=trained_ffjord.vector_field,
        control_dim=0,
        divergence_mode="hutchinson",
        n_hutchinson_samples=n_samples,
        solver="tsit5",
        adjoint="direct",
        rtol=1e-5,
        atol=1e-5,
    )
    flow = Transformed(Normal(jnp.zeros(2)), bij)
    return jax.vmap(flow.log_prob)(eval_pts)


probe_counts = (1, 4, 16, 64, 256)
print(f"exact   mean log p = {jnp.mean(exact_lp_eval):+.4f} (reference)\n")
print(f"{'n_probes':>10s}  {'mean log p':>12s}  {'bias':>10s}  {'RMSE':>10s}")
hutch_means, hutch_bias, hutch_rmse = [], [], []
for n in probe_counts:
    lp = hutch_eval_log_prob(n, key_seed=12345 + n)
    bias = float(jnp.mean(lp - exact_lp_eval))
    rmse = float(jnp.sqrt(jnp.mean((lp - exact_lp_eval) ** 2)))
    hutch_means.append(float(jnp.mean(lp)))
    hutch_bias.append(bias)
    hutch_rmse.append(rmse)
    print(f"{n:>10d}  {hutch_means[-1]:>+12.4f}  {bias:>+10.4f}  {rmse:>10.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(probe_counts, hutch_rmse, "o-", color=palette[0], label="RMSE")
ax.loglog(probe_counts, np.abs(hutch_bias), "s--", color=palette[3], label="|bias|")
# Reference: 1/sqrt(n) line anchored at the n=1 RMSE
ref = hutch_rmse[0] / jnp.sqrt(jnp.asarray(probe_counts) / probe_counts[0])
ax.loglog(probe_counts, np.asarray(ref), "k:", alpha=0.5, label=r"$\propto 1/\sqrt{n}$")
ax.set_xlabel("Hutchinson probe count")
ax.set_ylabel("error in $\\log p$ vs. exact (200 points)")
ax.set_title("Hutchinson convergence to the exact trace")
ax.legend()
style_axes(ax)
plt.show()

# %% [markdown]
# Two takeaways:
#
# - With a single Rademacher probe, the per-point error in $\log p$ is the
#   integrated trace-estimator variance over the ODE — non-trivial. It shrinks
#   as $1/\sqrt{n}$ as expected.
# - The mean log-density bias is much smaller than the RMSE: averaging across
#   evaluation points cancels the per-point variance, so reported NLLs are
#   reasonable even at small $n$. **Per-point** values (e.g. for OOD scoring
#   or anomaly detection) need either more probes or the exact trace.
