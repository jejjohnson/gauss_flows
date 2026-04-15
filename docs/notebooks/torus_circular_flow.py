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
# # Circular spline coupling on a toroidal density
# Fits a 2D torus with circular spline couplings that keep each angle periodic.

# %%
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flowjax.bijections import Chain, Permute
from flowjax.distributions import Normal, Transformed
from flowjax.train import fit_to_data

from gauss_flows import CircularRQSplineCoupling


plt.switch_backend("Agg")


def style_ax(ax):
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()


SCATTER_KW = dict(s=30, edgecolors="k", linewidths=0.5, alpha=0.5, zorder=5)

# %% [markdown]
# ## Generate a toroidal dataset
# Each sample is an angle pair; a two-mode mixture builds torus structure.

# %%
data_key, theta_key, phi_key, mix_key = jr.split(jr.key(0), 4)
n_samples = 3000
component = jr.bernoulli(mix_key, 0.5, (n_samples,))
theta_raw = jr.normal(theta_key, (n_samples,)) * 0.35 + jnp.where(component, 0.9, -0.9)
phi_raw = jr.normal(phi_key, (n_samples,)) * 0.3 + jnp.where(component, 1.3, -1.3)
wrap = lambda arr: ((arr + jnp.pi) % (2 * jnp.pi)) - jnp.pi
theta = wrap(theta_raw)
phi = wrap(phi_raw)
data = jnp.stack([theta, phi], axis=1)

# %%
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(data[:, 0], data[:, 1], **SCATTER_KW)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")
style_ax(ax)
ax.set_title("Samples on the torus (angles in radians)")

# %% [markdown]
# ## Build and train a circular coupling flow
# Stack circular spline couplings with permutations and train by maximum likelihood.

# %%
layer_keys = jr.split(data_key, 4)
layers = []
for layer_key in layer_keys:
    bij_key, perm_key = jr.split(layer_key)
    layers.append(
        CircularRQSplineCoupling(
            bij_key,
            shape=(2,),
            periodic_dims=(0, 1),
            n_bins=12,
            nn_width=64,
            nn_depth=1,
        )
    )
    layers.append(Permute(jr.permutation(perm_key, jnp.arange(2))))

bijection = Chain(layers).merge_chains()
base = Normal(jnp.zeros(2))
dist = Transformed(base, bijection)

trained_dist, losses = fit_to_data(
    data_key,
    dist,
    data,
    learning_rate=5e-4,
    max_epochs=150,
    max_patience=10,
    batch_size=256,
    val_prop=0.1,
    show_progress=False,
)

# %%
loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
epoch_axis = jnp.arange(len(losses["train"]))
loss_ax.plot(epoch_axis, losses["train"], label="train")
loss_ax.plot(epoch_axis, losses["val"], label="val")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Negative log-likelihood")
loss_ax.legend()
style_ax(loss_ax)
loss_ax.set_title("Training curve")

# %% [markdown]
# ## Sample from the trained circular flow
# Samples respect angular topology: a $2\\pi$ shift leaves log-probability unchanged.

# %%
sample_key = jr.key(1)
flow_samples = trained_dist.sample(sample_key, (3000,))

fig, (ax_data, ax_flow) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
ax_data.scatter(data[:, 0], data[:, 1], **SCATTER_KW)
ax_data.set_title("Training data")
style_ax(ax_data)
ax_flow.scatter(flow_samples[:, 0], flow_samples[:, 1], **SCATTER_KW)
ax_flow.set_title("Flow samples")
style_ax(ax_flow)
for ax in (ax_data, ax_flow):
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_xlim(-jnp.pi, jnp.pi)
    ax.set_ylim(-jnp.pi, jnp.pi)
fig.tight_layout()
