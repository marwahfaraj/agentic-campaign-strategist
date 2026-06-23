from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


BLUE = "#0B5CAD"
LIGHT_BLUE = "#D9EAFD"
GREEN = "#F28E2B"
TEAL = "#008A8A"
ORANGE = "#F28E2B"
GRAY = "#5F6B7A"
DARK = "#172033"
GRID = "#E7ECF3"


def apply_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5DCE6",
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def clean_label(value: object) -> str:
    return str(value).replace("_", " ").title()


def format_count(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def add_subtitle(ax: plt.Axes, subtitle: str) -> None:
    ax.text(
        0,
        1.04,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=10,
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D5DCE6")
    ax.spines["bottom"].set_color("#D5DCE6")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


percent_formatter = FuncFormatter(lambda value, _: f"{value:.0%}")
