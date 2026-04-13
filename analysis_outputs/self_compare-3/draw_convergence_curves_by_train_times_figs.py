"""基于汇总均值 CSV 绘制前三十次优化的收敛曲线图。"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

# 为 Matplotlib 和字体系统准备可写缓存目录，避免当前环境权限限制带来告警。
TEMP_DIR = Path(tempfile.gettempdir())
MPLCONFIG_DIR = TEMP_DIR / "mplconfig"
XDG_CACHE_DIR = TEMP_DIR / "xdg-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

# 强制使用无界面的 Agg 后端，确保脚本在终端和服务器环境都能稳定出图。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator

# 统一设置论文风格的基础参数，保持三张图的排版一致。
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 300

# 统一定义脚本相关路径，保证脚本在任意工作目录下都能直接运行。
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "convergence_curves_by_train_times.csv"
FIGS_DIR = SCRIPT_DIR / "figs"

# 主图只展示前 30 次优化，局部放大区域展示 0 到 8 次优化。
MAIN_MAX_TRAIN_TIMES = 30
INSET_MAX_TRAIN_TIMES = 8

# 定义两种方法的视觉样式。
# 按当前用户要求，Baseline 使用更浅的紫色，Ours 使用稍深的紫色。
METHOD_STYLES = {
    "baseline": {
        "label": "Baseline",
        "color": "#E3D1E6",
        "marker": "o",
        "linestyle": "--",
    },
    "ours": {
        "label": "Ours",
        "color": "#C68FBF",
        "marker": "s",
        "linestyle": "-",
    },
}

# 定义每个指标的标题、纵轴标签和输出文件名。
METRIC_CONFIGS = {
    "psnr": {
        "title": "PSNR",
        "ylabel": "PSNR (dB)",
        "invert_yaxis": False,
        "output_name": "psnr_convergence_top30.png",
    },
    "ssim": {
        "title": "SSIM",
        "ylabel": "SSIM",
        "invert_yaxis": False,
        "output_name": "ssim_convergence_top30.png",
    },
    "lpips": {
        "title": "LPIPS",
        "ylabel": "LPIPS",
        "invert_yaxis": False,
        "output_name": "lpips_convergence_top30.png",
    },
}


def load_mean_curves(csv_path: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    """从汇总 CSV 中读取两种方法在三个指标上的均值曲线。"""
    # 返回结构为 metric -> method -> [point, ...]，每个点只保留横轴和均值。
    metric_to_method_to_points: dict[str, dict[str, list[dict[str, float]]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metric_name = row["metric"].strip().lower()
            method_name = row["method"].strip().lower()
            train_times = int(row["train_times"])

            # 用户要求主图只画前 30 次优化，因此超出范围的点直接跳过。
            if train_times > MAIN_MAX_TRAIN_TIMES:
                continue

            metric_to_method_to_points.setdefault(metric_name, {})
            metric_to_method_to_points[metric_name].setdefault(method_name, [])
            metric_to_method_to_points[metric_name][method_name].append(
                {
                    "train_times": train_times,
                    "mean": float(row["mean"]),
                }
            )

    # 显式按 train_times 排序，避免 CSV 顺序变化影响绘图结果。
    for method_to_points in metric_to_method_to_points.values():
        for points in method_to_points.values():
            points.sort(key=lambda point: point["train_times"])

    return metric_to_method_to_points


def compute_axis_limits(points_by_method: dict[str, list[dict[str, float]]]) -> tuple[float, float]:
    """根据均值曲线自动计算当前指标的纵轴范围。"""
    all_values: list[float] = []
    for points in points_by_method.values():
        all_values.extend(point["mean"] for point in points)

    y_min = min(all_values)
    y_max = max(all_values)
    margin = max((y_max - y_min) * 0.12, 0.02 if y_max <= 1.0 else 0.3)
    return y_min - margin, y_max + margin


def compute_inset_axis_limits(points_by_method: dict[str, list[dict[str, float]]]) -> tuple[float, float]:
    """为局部放大图计算更紧的纵轴范围，以强化早期差异的视觉效果。"""
    all_values: list[float] = []
    for points in points_by_method.values():
        all_values.extend(point["mean"] for point in points)

    y_min = min(all_values)
    y_max = max(all_values)

    # 相比主图，这里使用更小的边距，让局部窗口把差异放得更明显。
    value_range = max(y_max - y_min, 1e-6)
    margin = max(value_range * 0.04, 0.008 if y_max <= 1.0 else 0.08)
    return y_min - margin, y_max + margin


def add_metric_curves(ax: plt.Axes, points_by_method: dict[str, list[dict[str, float]]]) -> None:
    """在指定坐标轴上绘制当前指标的两条均值曲线。"""
    for method_name, style in METHOD_STYLES.items():
        method_points = points_by_method.get(method_name, [])
        if not method_points:
            continue

        x_values = [point["train_times"] for point in method_points]
        y_values = [point["mean"] for point in method_points]

        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=2.3,
            # 保持完全不透明，避免曲线显得发灰。
            alpha=1.0,
            label=style["label"],
        )


def add_zoomed_inset(
    ax: plt.Axes,
    points_by_method: dict[str, list[dict[str, float]]],
    metric_name: str,
) -> None:
    """为主图添加 0 到 8 次优化的局部放大区域。"""
    metric_config = METRIC_CONFIGS[metric_name]

    # 将局部放大图放在右上角，既方便比较早期差异，也不容易遮挡主图主体。
    inset_ax = inset_axes(ax, width="43%", height="43%", loc="upper right", borderpad=1.4)

    # 局部放大图只取 0 到 8 次优化的均值点。
    inset_points_by_method: dict[str, list[dict[str, float]]] = {}
    for method_name, points in points_by_method.items():
        inset_points_by_method[method_name] = [
            point for point in points if point["train_times"] <= INSET_MAX_TRAIN_TIMES
        ]

    add_metric_curves(inset_ax, inset_points_by_method)

    inset_y_min, inset_y_max = compute_inset_axis_limits(inset_points_by_method)
    inset_ax.set_xlim(-0.5, INSET_MAX_TRAIN_TIMES + 0.5)
    inset_ax.set_ylim(inset_y_min, inset_y_max)
    inset_ax.set_xticks([0, 4, 8])
    # 减少主刻度数量，让纵轴显得更疏一些，同时把相邻曲线差距放大出来。
    inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    inset_ax.tick_params(labelsize=9)
    inset_ax.grid(True, linestyle=":", alpha=1)

    # LPIPS 越低越好，因此 inset 也保持相同的纵轴方向。
    if metric_config["invert_yaxis"]:
        inset_ax.invert_yaxis()

    # 使用连接框标记主图中被放大的区域。
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="gray", linestyle="--", linewidth=1.0)


def plot_single_metric(metric_name: str, points_by_method: dict[str, list[dict[str, float]]]) -> Path:
    """绘制单个指标的均值曲线图，并返回输出路径。"""
    metric_config = METRIC_CONFIGS[metric_name]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    add_metric_curves(ax, points_by_method)

    y_min, y_max = compute_axis_limits(points_by_method)
    ax.set_xlim(-0.5, MAIN_MAX_TRAIN_TIMES + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
    ax.set_xlabel("Optimization Times", fontweight="bold")
    ax.set_ylabel(metric_config["ylabel"], fontweight="bold")
    ax.set_title(
        f"{metric_config['title']} Mean Convergence in the First 30 Optimizations",
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.45)

    # LPIPS 越低越好，翻转纵轴后更符合阅读直觉。
    if metric_config["invert_yaxis"]:
        ax.invert_yaxis()

    add_zoomed_inset(ax, points_by_method, metric_name)

    # 图中包含 inset 时，手工留白通常比 tight_layout 更稳定。
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.9)
    output_path = FIGS_DIR / metric_config["output_name"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """读取汇总均值 CSV 并生成三张收敛曲线图。"""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    metric_to_method_to_points = load_mean_curves(CSV_PATH)
    saved_paths: list[Path] = []

    # 按固定顺序输出 3 个指标的图片。
    for metric_name in ("psnr", "ssim", "lpips"):
        saved_paths.append(plot_single_metric(metric_name, metric_to_method_to_points[metric_name]))

    for saved_path in saved_paths:
        print(f"saved figure: {saved_path}")


if __name__ == "__main__":
    main()
