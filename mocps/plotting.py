from pathlib import Path

import numpy as np

COLORS_EVOLUTION = {
    "Natural": "#8E8E8E",
    "Base Model": "#5B9BD5",
    "CSI Fine-tuned": "#F4B942",
    "ICR (Ours)": "#E84C3D",
}
COLORS_BENCHMARK = {
    "IDT": "#7FCDCD",
    "Twist": "#5B9BD5",
    "Genewiz": "#70AD47",
    "CT Fine-tuned": "#F4B942",
    "ICR (Ours)": "#E84C3D",
    "Expert Iter": "#8E6FBF",
}
COLORS_ABLATION = {
    "A: CSI only": "#B4D4E8",
    "B: CSI+CFD": "#7FCDCD",
    "C: CSI+CFD+CIS": "#F4B942",
    "D: Full (Ours)": "#E84C3D",
}


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save_figure(fig, output_dir: str | Path, stem: str, formats=("png", "pdf")) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}")


def make_violin(
    ax,
    data_list,
    labels,
    colors,
    ylabel,
    title,
    invert_note: bool = False,
    higher_note: bool = False,
    show_mean_text: bool = True,
    fontsize_label: int = 10,
    fontsize_title: int = 12,
):
    data_list = [np.array(data).flatten() for data in data_list]
    data_list = [data[~np.isnan(data)] for data in data_list]
    positions = range(1, len(labels) + 1)

    parts = ax.violinplot(
        data_list, positions=positions, showmedians=False, showextrema=False, widths=0.6
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.85)
        pc.set_edgecolor("white")
        pc.set_linewidth(1.2)

    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#555", "linewidth": 0.9},
        capprops={"color": "#555", "linewidth": 0.9},
        boxprops={"color": "#444", "linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    for i, (data, color) in enumerate(zip(data_list, colors)):
        jitter = np.random.uniform(-0.1, 0.1, size=len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            color=color,
            alpha=0.35,
            s=12,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )

    if show_mean_text:
        global_min = min(np.min(data) for data in data_list)
        global_max = max(np.max(data) for data in data_list)
        offset = (global_max - global_min) * 0.03
        for i, (data, color) in enumerate(zip(data_list, colors)):
            ax.text(
                i + 1,
                np.max(data) + offset,
                f"{np.mean(data):.3f}",
                ha="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=11)
    note = ""
    if invert_note:
        note = "\n(lower is better)"
    elif higher_note:
        note = "\n(higher is better)"
    ax.set_title(title + note, fontsize=fontsize_title, fontweight="bold", pad=10)
