#!/usr/bin/env python3
"""
Aggregate self-compare metric histories, render convergence plots,
and export summary tables for paper use.

Example:
    python3 analyze_self_compare.py \
        --root images/self_compare \
        --output-dir analysis_outputs/self_compare \
        --round-cutoff 32 \
        --psnr-threshold 24.0 \
        --ssim-threshold 0.80 \
        --lpips-threshold 0.20
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("psnr", "ssim", "lpips")
DEFAULT_METHOD_LABELS = {
    "baseline": "Baseline",
    "ours": "Ours",
}
METRIC_PRECISION = {
    "psnr": 2,
    "ssim": 3,
    "lpips": 3,
}
METRIC_DIRECTION = {
    "psnr": "high",
    "ssim": "high",
    "lpips": "low",
}
METHOD_COLORS = {
    "baseline": "#d95f02",
    "ours": "#1b9e77",
}


@dataclass(frozen=True)
class MetricPoint:
    train_times: int
    optimize_round: int
    psnr: float
    ssim: float
    lpips: float

    def x_value(self, axis: str) -> int:
        if axis == "train_times":
            return self.train_times
        return self.optimize_round

    def metric(self, name: str) -> float:
        return getattr(self, name)


@dataclass(frozen=True)
class FrameHistory:
    method: str
    frame: str
    csv_path: Path
    points: Tuple[MetricPoint, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze self-compare metric histories and export figures/tables."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("images/self_compare"),
        help="Root directory containing method/frame/metrics_history.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/self_compare"),
        help="Directory for generated plots and tables.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("optimize_round", "train_times"),
        default="train_times",
        help=(
            "Horizontal axis used in curve, AUC, and time-to-quality statistics. "
            "Use train_times for per-frame convergence after insertion; "
            "optimize_round is the global optimization round in the whole sequence."
        ),
    )
    parser.add_argument(
        "--round-cutoff",
        type=int,
        default=32,
        help="Round cutoff used for round-specific stats and AUC@T.",
    )
    parser.add_argument(
        "--plot-max-round",
        type=int,
        default=None,
        help="Optional maximum x-axis value for convergence plots.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=None,
        help="Optional PSNR threshold for time-to-quality.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=None,
        help="Optional SSIM threshold for time-to-quality.",
    )
    parser.add_argument(
        "--lpips-threshold",
        type=float,
        default=None,
        help="Optional LPIPS threshold for time-to-quality.",
    )
    parser.add_argument(
        "--require-common-frames",
        action="store_true",
        default=True,
        help="Restrict analysis to frame IDs present in every method directory.",
    )
    parser.add_argument(
        "--allow-noncommon-frames",
        action="store_false",
        dest="require_common_frames",
        help="Allow each method to use its own available frame IDs.",
    )
    return parser.parse_args()


def load_histories(root: Path) -> List[FrameHistory]:
    csv_paths = sorted(root.glob("*/*/metrics_history.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No metrics_history.csv files found under {root}")

    histories: List[FrameHistory] = []
    for csv_path in csv_paths:
        method = csv_path.parent.parent.name
        frame = csv_path.parent.name
        points: List[MetricPoint] = []
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"train_times", "optimize_round", "psnr", "ssim", "lpips"}
            if set(reader.fieldnames or []) != expected:
                raise ValueError(
                    f"Unexpected header in {csv_path}: {reader.fieldnames}. "
                    f"Expected {sorted(expected)}"
                )
            for row in reader:
                points.append(
                    MetricPoint(
                        train_times=int(row["train_times"]),
                        optimize_round=int(row["optimize_round"]),
                        psnr=float(row["psnr"]),
                        ssim=float(row["ssim"]),
                        lpips=float(row["lpips"]),
                    )
                )
        points.sort(key=lambda p: p.optimize_round)
        histories.append(
            FrameHistory(
                method=method,
                frame=frame,
                csv_path=csv_path,
                points=tuple(points),
            )
        )
    return histories


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_summary(values: Sequence[float], precision: int) -> str:
    if not values:
        return "--"
    mean, std = summarize(values)
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def method_label(method: str) -> str:
    return DEFAULT_METHOD_LABELS.get(method, method)


def frame_intersection(histories: Sequence[FrameHistory]) -> Dict[str, List[FrameHistory]]:
    by_method: Dict[str, List[FrameHistory]] = defaultdict(list)
    for history in histories:
        by_method[history.method].append(history)
    return dict(by_method)


def filter_histories(
    histories: Sequence[FrameHistory], require_common_frames: bool
) -> Dict[str, List[FrameHistory]]:
    by_method = frame_intersection(histories)
    if not require_common_frames:
        return {k: sorted(v, key=lambda h: h.frame) for k, v in by_method.items()}

    frame_sets = [set(h.frame for h in group) for group in by_method.values()]
    common = set.intersection(*frame_sets)
    filtered: Dict[str, List[FrameHistory]] = {}
    for method, group in by_method.items():
        filtered[method] = sorted(
            [history for history in group if history.frame in common],
            key=lambda h: h.frame,
        )
    return filtered


def get_value_at_x(history: FrameHistory, axis: str, target: int) -> Optional[Dict[str, float]]:
    values: Dict[str, float] = {}
    for metric in METRICS:
        interpolated = interpolate_metric_at_x(history, axis, target, metric)
        if interpolated is None:
            return None
        values[metric] = interpolated
    return values


def interpolate_metric_at_x(
    history: FrameHistory, axis: str, target: int, metric: str
) -> Optional[float]:
    points = history.points
    xs = [point.x_value(axis) for point in points]
    if not xs or target < xs[0] or target > xs[-1]:
        return None
    for point in points:
        if point.x_value(axis) == target:
            return point.metric(metric)
    for left, right in zip(points[:-1], points[1:]):
        x0 = left.x_value(axis)
        x1 = right.x_value(axis)
        if x0 < target < x1:
            y0 = left.metric(metric)
            y1 = right.metric(metric)
            alpha = (target - x0) / float(x1 - x0)
            return y0 * (1.0 - alpha) + y1 * alpha
    return None


def auc_until(history: FrameHistory, axis: str, cutoff: int, metric: str) -> Optional[float]:
    points = history.points
    xs = [point.x_value(axis) for point in points]
    if len(points) < 2 or xs[-1] < cutoff:
        return None

    clipped_xs: List[float] = []
    clipped_ys: List[float] = []
    for point in points:
        x = point.x_value(axis)
        if x <= cutoff:
            clipped_xs.append(float(x))
            clipped_ys.append(point.metric(metric))
        else:
            break

    if clipped_xs and clipped_xs[-1] < cutoff:
        y_cut = interpolate_metric_at_x(history, axis, cutoff, metric)
        if y_cut is None:
            return None
        clipped_xs.append(float(cutoff))
        clipped_ys.append(y_cut)

    if len(clipped_xs) < 2:
        return None

    auc = 0.0
    for x0, x1, y0, y1 in zip(clipped_xs[:-1], clipped_xs[1:], clipped_ys[:-1], clipped_ys[1:]):
        auc += 0.5 * (y0 + y1) * (x1 - x0)
    return auc


def first_time_to_quality(
    history: FrameHistory, axis: str, metric: str, threshold: float
) -> Optional[float]:
    points = history.points
    direction = METRIC_DIRECTION[metric]

    def is_good(value: float) -> bool:
        if direction == "low":
            return value <= threshold
        return value >= threshold

    if not points:
        return None

    prev_x = None
    prev_y = None
    for point in points:
        x = point.x_value(axis)
        y = point.metric(metric)
        if is_good(y):
            if prev_x is None or prev_y is None:
                return float(x)
            if is_good(prev_y):
                return float(prev_x)
            if y == prev_y:
                return float(x)
            alpha = (threshold - prev_y) / (y - prev_y)
            if direction == "low":
                alpha = (prev_y - threshold) / (prev_y - y)
            alpha = max(0.0, min(1.0, alpha))
            return prev_x + alpha * (x - prev_x)
        prev_x = float(x)
        prev_y = y
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_tex_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines = []
    column_spec = "l" + "c" * (len(headers) - 1)
    lines.append(r"\begin{tabular}{" + column_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def build_round_table(
    histories_by_method: Dict[str, List[FrameHistory]], axis: str, target_round: int
) -> Tuple[List[str], List[List[str]]]:
    header = ["Method", "N", "PSNR", "SSIM", "LPIPS"]
    rows: List[List[str]] = []
    for method, histories in sorted(histories_by_method.items()):
        values = {metric: [] for metric in METRICS}
        count = 0
        for history in histories:
            row = get_value_at_x(history, axis, target_round)
            if row is None:
                continue
            count += 1
            for metric in METRICS:
                values[metric].append(row[metric])
        rows.append(
            [
                method_label(method),
                str(count),
                format_summary(values["psnr"], METRIC_PRECISION["psnr"]),
                format_summary(values["ssim"], METRIC_PRECISION["ssim"]),
                format_summary(values["lpips"], METRIC_PRECISION["lpips"]),
            ]
        )
    return header, rows


def build_auc_table(
    histories_by_method: Dict[str, List[FrameHistory]], axis: str, cutoff: int
) -> Tuple[List[str], List[List[str]]]:
    header = ["Method", "N", f"PSNR AUC@{cutoff}", f"SSIM AUC@{cutoff}", f"LPIPS AUC@{cutoff}"]
    rows: List[List[str]] = []
    for method, histories in sorted(histories_by_method.items()):
        auc_values = {metric: [] for metric in METRICS}
        for history in histories:
            for metric in METRICS:
                auc = auc_until(history, axis, cutoff, metric)
                if auc is not None:
                    auc_values[metric].append(auc)
        count = min((len(auc_values[metric]) for metric in METRICS), default=0)
        rows.append(
            [
                method_label(method),
                str(count),
                format_summary(auc_values["psnr"], METRIC_PRECISION["psnr"]),
                format_summary(auc_values["ssim"], METRIC_PRECISION["ssim"]),
                format_summary(auc_values["lpips"], METRIC_PRECISION["lpips"]),
            ]
        )
    return header, rows


def build_t2q_table(
    histories_by_method: Dict[str, List[FrameHistory]],
    axis: str,
    thresholds: Dict[str, Optional[float]],
) -> Tuple[List[str], List[List[str]]]:
    active_metrics = [metric for metric in METRICS if thresholds.get(metric) is not None]
    header = ["Method"]
    for metric in active_metrics:
        direction = ">=" if METRIC_DIRECTION[metric] == "high" else "<="
        header.append(f"{metric.upper()} {direction} {thresholds[metric]}")
    rows: List[List[str]] = []
    for method, histories in sorted(histories_by_method.items()):
        row = [method_label(method)]
        for metric in active_metrics:
            values: List[float] = []
            for history in histories:
                t2q = first_time_to_quality(history, axis, metric, thresholds[metric])  # type: ignore[arg-type]
                if t2q is not None:
                    values.append(t2q)
            total = len(histories)
            row.append(
                f"{format_summary(values, 1) if values else '--'} "
                f"({len(values)}/{total})"
            )
        rows.append(row)
    return header, rows


def build_curve_csv_rows(
    histories_by_method: Dict[str, List[FrameHistory]], axis: str, plot_max_round: Optional[int]
) -> List[List[str]]:
    rows: List[List[str]] = []
    for method, histories in sorted(histories_by_method.items()):
        round_values: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {metric: [] for metric in METRICS}
        )
        for history in histories:
            for point in history.points:
                x = point.x_value(axis)
                if plot_max_round is not None and x > plot_max_round:
                    continue
                for metric in METRICS:
                    round_values[x][metric].append(point.metric(metric))
        for x in sorted(round_values):
            for metric in METRICS:
                values = round_values[x][metric]
                mean, std = summarize(values)
                rows.append(
                    [
                        method,
                        method_label(method),
                        str(x),
                        metric,
                        f"{mean:.6f}",
                        f"{std:.6f}",
                        str(len(values)),
                    ]
                )
    return rows


def plot_curves(
    histories_by_method: Dict[str, List[FrameHistory]],
    axis: str,
    output_path: Path,
    plot_max_round: Optional[int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        for method, histories in sorted(histories_by_method.items()):
            round_values: Dict[int, List[float]] = defaultdict(list)
            for history in histories:
                for point in history.points:
                    x = point.x_value(axis)
                    if plot_max_round is not None and x > plot_max_round:
                        continue
                    round_values[x].append(point.metric(metric))
            xs = sorted(round_values)
            means = []
            stds = []
            for x in xs:
                mean, std = summarize(round_values[x])
                means.append(mean)
                stds.append(std)

            color = METHOD_COLORS.get(method, None)
            label = method_label(method)
            lowers = [m - s for m, s in zip(means, stds)]
            uppers = [m + s for m, s in zip(means, stds)]
            ax.plot(xs, means, label=label, color=color, linewidth=2.0)
            ax.fill_between(xs, lowers, uppers, color=color, alpha=0.18)

        ax.set_title(metric.upper())
        ax.set_xlabel(axis.replace("_", " "))
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        if metric == "lpips":
            ax.invert_yaxis()
    axes[0].legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    histories = load_histories(args.root)
    histories_by_method = filter_histories(histories, args.require_common_frames)

    ensure_dir(args.output_dir)

    curve_plot_path = args.output_dir / f"convergence_curves_by_{args.x_axis}.png"
    plot_curves(histories_by_method, args.x_axis, curve_plot_path, args.plot_max_round)

    curve_rows = build_curve_csv_rows(histories_by_method, args.x_axis, args.plot_max_round)
    write_csv(
        args.output_dir / f"convergence_curves_by_{args.x_axis}.csv",
        ["method", "method_label", args.x_axis, "metric", "mean", "std", "count"],
        curve_rows,
    )

    round_header, round_rows = build_round_table(
        histories_by_method, args.x_axis, args.round_cutoff
    )
    write_csv(args.output_dir / f"round_{args.round_cutoff}_stats.csv", round_header, round_rows)
    write_tex_table(
        args.output_dir / f"round_{args.round_cutoff}_stats.tex", round_header, round_rows
    )

    auc_header, auc_rows = build_auc_table(histories_by_method, args.x_axis, args.round_cutoff)
    write_csv(args.output_dir / f"auc_{args.round_cutoff}_stats.csv", auc_header, auc_rows)
    write_tex_table(
        args.output_dir / f"auc_{args.round_cutoff}_stats.tex", auc_header, auc_rows
    )

    thresholds = {
        "psnr": args.psnr_threshold,
        "ssim": args.ssim_threshold,
        "lpips": args.lpips_threshold,
    }
    if any(value is not None for value in thresholds.values()):
        t2q_header, t2q_rows = build_t2q_table(histories_by_method, args.x_axis, thresholds)
        write_csv(args.output_dir / "time_to_quality_stats.csv", t2q_header, t2q_rows)
        write_tex_table(args.output_dir / "time_to_quality_stats.tex", t2q_header, t2q_rows)

    summary_lines = [
        f"Loaded methods: {', '.join(sorted(histories_by_method))}",
        f"Frame counts: "
        + ", ".join(
            f"{method_label(method)}={len(histories)}"
            for method, histories in sorted(histories_by_method.items())
        ),
        f"x-axis: {args.x_axis}",
        f"round cutoff: {args.round_cutoff}",
        f"curve plot: {curve_plot_path}",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
