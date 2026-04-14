"""基于真实统计 CSV 生成图 4 的两种版式 demo。"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

# 为 Matplotlib 缓存准备可写目录，避免终端环境中的权限告警。
TEMP_DIR = Path(tempfile.gettempdir())
MPLCONFIG_DIR = TEMP_DIR / "mplconfig"
XDG_CACHE_DIR = TEMP_DIR / "xdg-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

# 强制使用无界面后端，保证脚本在本地终端与服务器上都能稳定出图。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 统一论文绘图基础风格，减少两个 demo 之间的视觉跳变。
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "convergence_curves_by_train_times.csv"
FIGS_DIR = SCRIPT_DIR / "figs"
PAPER_OUTPUT_PATH = FIGS_DIR / "ablation_curves_paper_compact_demo.png"
SUPP_OUTPUT_PATH = FIGS_DIR / "ablation_curves_supplementary_demo.png"

METHOD_ORDER = ["baseline", "ours"]
METHOD_STYLES = {
    "baseline": {
        "label": "Baseline",
        "color": "#9AA0A6",
        "linestyle": "--",
        "marker": "o",
    },
    "ours": {
        "label": "Ours",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
    },
}
METRIC_CONFIGS = [
    {
        "name": "psnr",
        "title": "PSNR",
        "ylabel": "dB",
        "higher_is_better": True,
    },
    {
        "name": "ssim",
        "title": "SSIM",
        "ylabel": "Score",
        "higher_is_better": True,
    },
    {
        "name": "lpips",
        "title": "LPIPS",
        "ylabel": "Score",
        "higher_is_better": False,
    },
]

# 正文版聚焦于前 16 次优化，补充材料版保留更完整的前 28 次趋势。
PAPER_TRAIN_TIMES = [0, 4, 8, 12, 16]
SUPP_MAX_TRAIN_TIMES = 28
SUPP_INSET_MAX_TRAIN_TIMES = 8


def load_curve_points(csv_path: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    """读取 CSV 中三项指标、两种方法的均值与标准差曲线。"""
    metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metric_name = row["metric"].strip().lower()
            method_name = row["method"].strip().lower()
            train_times = int(row["train_times"])

            metric_to_method_to_points.setdefault(metric_name, {})
            metric_to_method_to_points[metric_name].setdefault(method_name, [])
            metric_to_method_to_points[metric_name][method_name].append(
                {
                    "train_times": train_times,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                }
            )

    # 显式按 train_times 排序，避免未来 CSV 行顺序变化带来错位曲线。
    for method_to_points in metric_to_method_to_points.values():
        for points in method_to_points.values():
            points.sort(key=lambda item: item["train_times"])

    return metric_to_method_to_points


def filter_points(points: list[dict[str, float]], allowed_train_times: list[int] | None = None, max_train_times: int | None = None) -> list[dict[str, float]]:
    """按照给定轮次集合或最大轮次过滤绘图点。"""
    filtered_points: list[dict[str, float]] = []
    for point in points:
        train_times = point["train_times"]
        if allowed_train_times is not None and train_times not in allowed_train_times:
            continue
        if max_train_times is not None and train_times > max_train_times:
            continue
        filtered_points.append(point)
    return filtered_points


def compute_axis_limits(points_by_method: dict[str, list[dict[str, float]]], use_std: bool) -> tuple[float, float]:
    """根据当前指标的均值或均值±标准差范围自适应设置纵轴。"""
    values: list[float] = []
    for points in points_by_method.values():
        for point in points:
            if use_std:
                values.append(point["mean"] - point["std"])
                values.append(point["mean"] + point["std"])
            else:
                values.append(point["mean"])

    y_min = min(values)
    y_max = max(values)
    value_range = max(y_max - y_min, 1e-6)
    min_margin = 0.012 if y_max <= 1.0 else 0.22
    margin = max(value_range * 0.14, min_margin)
    return y_min - margin, y_max + margin


def draw_curves(ax: plt.Axes, points_by_method: dict[str, list[dict[str, float]]], include_std_band: bool, linewidth: float, marker_size: float) -> None:
    """在给定坐标轴上绘制两种方法的收敛曲线。"""
    for method_name in METHOD_ORDER:
        style = METHOD_STYLES[method_name]
        points = points_by_method[method_name]
        x_values = [point["train_times"] for point in points]
        y_values = [point["mean"] for point in points]

        if include_std_band:
            lower_values = [point["mean"] - point["std"] for point in points]
            upper_values = [point["mean"] + point["std"] for point in points]
            ax.fill_between(
                x_values,
                lower_values,
                upper_values,
                color=style["color"],
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )

        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=marker_size,
            linewidth=linewidth,
            label=style["label"],
            zorder=2,
        )


def plot_paper_compact_demo(metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]]) -> Path:
    """绘制适合正文单栏宽度的低矮横排 demo。"""
    plt.rcParams["axes.titlesize"] = 8.8
    plt.rcParams["axes.labelsize"] = 8.2
    plt.rcParams["xtick.labelsize"] = 7.2
    plt.rcParams["ytick.labelsize"] = 7.2
    plt.rcParams["legend.fontsize"] = 7.5

    # 按单栏宽度 3.45 英寸控制，并把高度压到较低，模拟论文中的紧凑摆放。
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.42))

    for axis, metric_config in zip(axes, METRIC_CONFIGS):
        metric_name = metric_config["name"]
        points_by_method = {
            method_name: filter_points(metric_to_method_to_points[metric_name][method_name], allowed_train_times=PAPER_TRAIN_TIMES)
            for method_name in METHOD_ORDER
        }

        draw_curves(axis, points_by_method, include_std_band=False, linewidth=1.55, marker_size=2.8)
        y_min, y_max = compute_axis_limits(points_by_method, use_std=False)

        axis.set_title(metric_config["title"], pad=2.0, fontweight="bold")
        axis.set_xlim(PAPER_TRAIN_TIMES[0] - 0.4, PAPER_TRAIN_TIMES[-1] + 0.4)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([0, 8, 16])
        axis.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        # 为节省空间，只有最左侧子图保留纵轴标签，其余子图仅保留刻度。
        if axis is axes[0]:
            axis.set_ylabel(metric_config["ylabel"], fontweight="bold", labelpad=1.5)
        else:
            axis.set_ylabel("")

        # 轻量提示指标方向，避免正文 caption 再花太多文字解释。
        axis.text(
            0.97,
            0.08,
            "up" if metric_config["higher_is_better"] else "down",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#666666",
        )

    axes[1].set_xlabel("Optimization steps", fontweight="bold", labelpad=1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
        handlelength=1.8,
        columnspacing=1.2,
    )

    # 手工控制留白，让图保持“短而稳”的单栏比例。
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.28, top=0.83, wspace=0.28)
    fig.savefig(PAPER_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return PAPER_OUTPUT_PATH


def add_supplementary_inset(ax: plt.Axes, points_by_method: dict[str, list[dict[str, float]]], metric_name: str) -> None:
    """为补充材料版添加 0 到 8 次优化的局部放大图。"""
    inset_ax = inset_axes(ax, width="36%", height="48%", loc="upper right", borderpad=1.0)
    inset_points_by_method = {
        method_name: filter_points(points_by_method[method_name], max_train_times=SUPP_INSET_MAX_TRAIN_TIMES)
        for method_name in METHOD_ORDER
    }

    draw_curves(inset_ax, inset_points_by_method, include_std_band=False, linewidth=1.2, marker_size=2.4)
    inset_y_min, inset_y_max = compute_axis_limits(inset_points_by_method, use_std=False)
    inset_ax.set_xlim(-0.3, SUPP_INSET_MAX_TRAIN_TIMES + 0.3)
    inset_ax.set_ylim(inset_y_min, inset_y_max)
    inset_ax.set_xticks([0, 4, 8])
    inset_ax.tick_params(labelsize=7)
    inset_ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="#888888", linestyle="--", linewidth=0.8)


def plot_supplementary_demo(metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]]) -> Path:
    """绘制更适合 supplementary 的信息更完整版 demo。"""
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 9.5
    plt.rcParams["xtick.labelsize"] = 8.4
    plt.rcParams["ytick.labelsize"] = 8.4
    plt.rcParams["legend.fontsize"] = 8.8

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))

    for axis, metric_config in zip(axes, METRIC_CONFIGS):
        metric_name = metric_config["name"]
        points_by_method = {
            method_name: filter_points(metric_to_method_to_points[metric_name][method_name], max_train_times=SUPP_MAX_TRAIN_TIMES)
            for method_name in METHOD_ORDER
        }

        draw_curves(axis, points_by_method, include_std_band=True, linewidth=2.0, marker_size=3.4)
        y_min, y_max = compute_axis_limits(points_by_method, use_std=True)

        axis.set_title(metric_config["title"], pad=4.0, fontweight="bold")
        axis.set_xlim(-0.5, SUPP_MAX_TRAIN_TIMES + 0.5)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
        axis.set_xlabel("Optimization steps", fontweight="bold")
        axis.set_ylabel(metric_config["ylabel"], fontweight="bold")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        add_supplementary_inset(axis, points_by_method, metric_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.18, top=0.78, wspace=0.28)
    fig.savefig(SUPP_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return SUPP_OUTPUT_PATH


def main() -> None:
    """读取统计 CSV 并输出两种版式 demo。"""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    metric_to_method_to_points = load_curve_points(CSV_PATH)
    output_paths = [
        plot_paper_compact_demo(metric_to_method_to_points),
        plot_supplementary_demo(metric_to_method_to_points),
    ]

    for output_path in output_paths:
        print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
