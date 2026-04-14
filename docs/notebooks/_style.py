"""Shared plot styling for gauss_flows documentation notebooks."""

from __future__ import annotations


def style_ax(ax):
    """Apply gauss_flows-style grid and ticks to an axis."""
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()


SCATTER_KW = dict(s=30, edgecolors="k", linewidths=0.5, alpha=0.5, zorder=5)
