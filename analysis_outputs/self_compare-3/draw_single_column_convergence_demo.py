"""基于真实统计 CSV 绘制适合正文单栏排版的收敛曲线 demo。"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

# 为 Matplotlib 和字体缓存准备可写目录，避免沙箱环境下出现缓存权限告警。
TEMP_DIR = Path(tempfile.gettempdir())
MPLCONFIG_DIR = TEMP_DIR / "mplconfig"
XDG_CACHE_DIR = TEMP_DIR / "xdg-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

# 强制使用无界面后端，确保终端环境里也能稳定出图。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# 使用较克制的学术配色，让图在正文里不喧宾夺主，同时保证区分度。
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.titlesize"] = 10.5
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8.8
plt.rcParams["figure.dpi"] = 300

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "convergence_curves_by_train_times.csv"
FIGS_DIR = SCRIPT_DIR / "figs"
OUTPUT_PATH = FIGS_DIR / "ablation_curves_single_column_demo.png"

# 正文版图只强调最关键的早期阶段，因此仅保留前 16 次优化。
KEY_TRAIN_TIMES = [0, 4, 8, 12, 16]
METHOD_ORDER = ["baseline", "ours"]
METHOD_STYLES = {
    "baseline": {
        "label": "Baseline",
        "color": "#9AA0A6",
        "marker": "o",
        "linestyle": "--",
    },
    "ours": {
        "label": "Ours",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-",
    },
}
METRIC_CONFIGS = [
    {
        "name": "psnr",
        "label": "PSNR (dB)",
        "title": "PSNR",
        "higher_is_better": True,
    },
    {
        "name": "ssim",
        "label": "SSIM",
        "title": "SSIM",
        "higher_is_better": True,
    },
    {
        "name": "lpips",
        "label": "LPIPS",
        "title": "LPIPS",
        "higher_is_better": False,
    },
]


def load_points(csv_path: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    """读取 CSV 中指定优化轮次的均值曲线。"""
    metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            train_times = int(row["train_times"])
            if train_times not in KEY_TRAIN_TIMES:
                continue

            metric_name = row["metric"].strip().lower()
            method_name = row["method"].strip().lower()
            metric_to_method_to_points.setdefault(metric_name, {})
            metric_to_method_to_points[metric_name].setdefault(method_name, [])
            metric_to_method_to_points[metric_name][method_name].append(
                {
                    "train_times": train_times,
                    "mean": float(row["mean"]),
                }
            )

    # 显式排序，避免未来 CSV 行顺序变化影响曲线形态。
    for method_to_points in metric_to_method_to_points.values():
        for points in method_to_points.values():
            points.sort(key=lambda item: item["train_times"])

    return metric_to_method_to_points


def compute_y_limits(points_by_method: dict[str, list[dict[str, float]]]) -> tuple[float, float]:
    """根据当前指标的数据范围自适应设置纵轴边距。"""
    values = [point["mean"] for points in points_by_method.values() for point in points]
    y_min = min(values)
    y_max = max(values)
    value_range = max(y_max - y_min, 1e-6)

    # 不同量纲下分别设置最小边距，避免图过于贴边。
    min_margin = 0.006 if y_max <= 1.0 else 0.18
    margin = max(value_range * 0.16, min_margin)
    return y_min - margin, y_max + margin


def format_delta_text(metric_name: str, baseline_value: float, ours_value: float) -> str:
    """生成图内的 16 次优化差值标注。"""
    delta = ours_value - baseline_value
    if metric_name == "psnr":
        return f"$\\Delta$@16 = {delta:+.2f} dB"
    if metric_name == "ssim":
        return f"$\\Delta$@16 = {delta:+.3f}"
    return f"$\\Delta$@16 = {delta:+.3f}"


def get_last_value(points: list[dict[str, float]]) -> float:
    """读取当前曲线最后一个点的均值。"""
    return points[-1]["mean"]


def plot_single_column_demo(metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]]) -> Path:
    """绘制三行共享横轴的正文单栏 demo。"""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # 3.45 英寸接近论文单栏宽度，6.1 英寸高度可容纳三行子图与图例。
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(3.45, 6.1),
        sharex=True,
    )

    for axis, metric_config in zip(axes, METRIC_CONFIGS):
        metric_name = metric_config["name"]
        points_by_method = metric_to_method_to_points[metric_name]

        for method_name in METHOD_ORDER:
            style = METHOD_STYLES[method_name]
            points = points_by_method[method_name]
            x_values = [point["train_times"] for point in points]
            y_values = [point["mean"] for point in points]

            axis.plot(
                x_values,
                y_values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=2.0,
                label=style["label"],
            )

        y_min, y_max = compute_y_limits(points_by_method)
        axis.set_ylim(y_min, y_max)
        axis.set_ylabel(metric_config["label"], fontweight="bold")
        axis.set_title(metric_config["title"], loc="left", fontweight="bold", pad=2)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        trend_hint = "Higher is better" if metric_config["higher_is_better"] else "Lower is better"
        axis.text(
            0.985,
            0.08,
            trend_hint,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.6,
            color="#666666",
        )

        baseline_last = get_last_value(points_by_method["baseline"])
        ours_last = get_last_value(points_by_method["ours"])
        axis.text(
            0.985,
            0.84,
            format_delta_text(metric_name, baseline_last, ours_last),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.9,
            color=METHOD_STYLES["ours"]["color"],
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    axes[-1].set_xlabel("Local optimization steps after insertion", fontweight="bold")
    axes[-1].set_xlim(KEY_TRAIN_TIMES[0] - 0.4, KEY_TRAIN_TIMES[-1] + 0.4)
    axes[-1].set_xticks(KEY_TRAIN_TIMES)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    # 使用手工留白，让图例、标题和子图间距更可控，更贴近论文版式。
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.08, top=0.93, hspace=0.22)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    """读取 CSV 并生成单栏正文版 demo 图。"""
    metric_to_method_to_points = load_points(CSV_PATH)
    output_path = plot_single_column_demo(metric_to_method_to_points)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
