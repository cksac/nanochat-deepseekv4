"""Render system-card style figures from compare_deepseekv4 CSV outputs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["native", "param_matched", "deepseekv4", "deepseekv4_mhc", "deepseekv4_attnres", "deepseekv4_attnmhc", "deepseekv4_attnmhc_ch", "deepseekv4_attnmhc_2h", "deepseekv4_attnmhc_swa", "deepseekv4_attnmhc_hca", "deepseekv4_dn", "deepseekv4_attnmhc_dnd"]
MODEL_LABELS = {
    "native": "native",
    "param_matched": "param-matched",
    "deepseekv4": "deepseekv4 (mHc)",
    "deepseekv4_mhc": "deepseekv4 (mHc C+2H)",
    "deepseekv4_attnres": "deepseekv4 (AttnRes C+H+DN)",
    "deepseekv4_attnmhc": "deepseekv4 (AttnMhc C+H+DN)",
    "deepseekv4_attnmhc_ch": "deepseekv4 (AttnMhc C+H)",
    "deepseekv4_attnmhc_2h": "deepseekv4 (AttnMhc C+2H)",
    "deepseekv4_attnmhc_swa": "deepseekv4 (AttnMhc 2SWA+C+H+DN)",
    "deepseekv4_attnmhc_hca": "deepseekv4 (AttnMhc 2HCA+C+H+DN)",
    "deepseekv4_dn": "deepseekv4 (mHc C+H+DN)",
    "deepseekv4_attnmhc_dnd": "deepseekv4 (AttnMhc C+H+DND)",
    "deepseekv4_no_hash": "no hash",
    "deepseekv4_no_shared": "no shared",
}
DATASET_LABELS = {
    "tiny_shakespeare": "Tiny Shakespeare",
    "wikitext2": "WikiText-2",
}
COLORS = {
    "native": "#6b7280",
    "param_matched": "#b5bac3",
    "deepseekv4": "#c84a35",
    "deepseekv4_mhc": "#e67e22",
    "deepseekv4_attnres": "#2e7d32",
    "deepseekv4_attnmhc": "#1565c0",
    "deepseekv4_attnmhc_ch": "#42a5f5",
    "deepseekv4_attnmhc_2h": "#f57c00",
    "deepseekv4_attnmhc_swa": "#d81b60",
    "deepseekv4_attnmhc_hca": "#5e35b1",
    "deepseekv4_dn": "#8e24aa",
    "deepseekv4_attnmhc_dnd": "#00897b",
    "deepseekv4_no_hash": "#e09a8a",
    "deepseekv4_no_shared": "#8f2e21",
}
GRID_COLOR = "#d9dde3"
TEXT_COLOR = "#111827"


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


def as_int(row, key):
    return int(float(row[key]))


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "axes.titleweight": "regular",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.55)
    ax.grid(axis="x", visible=False)


def config_order(rows):
    names = sorted({row["config"] for row in rows})
    preferred = ["small", "medium"]
    return [name for name in preferred if name in names] + [name for name in names if name not in preferred]


def metric_cells(summary_rows):
    cells = []
    for dataset in ["tiny_shakespeare", "wikitext2"]:
        for split in ["validation", "test"]:
            if any(row["dataset"] == dataset and row["split"] == split for row in summary_rows):
                cells.append((dataset, split))
    return cells


def summary_lookup(summary_rows, metric):
    lookup = {}
    for row in summary_rows:
        lookup[(row["config"], row["model"], row["dataset"], row["split"])] = (
            as_float(row, f"{metric}_mean"),
            as_float(row, f"{metric}_std"),
            as_int(row, "n"),
        )
    return lookup


def draw_grouped_metric_bars(summary_rows, metric, ylabel, title, output_path):
    configs = config_order(summary_rows)
    cells = metric_cells(summary_rows)
    lookup = summary_lookup(summary_rows, metric)
    fig, axes = plt.subplots(len(configs), 1, figsize=(13, 3.8 * len(configs)), squeeze=False)
    width = 0.18
    x = np.arange(len(cells))
    n_models = len(MODEL_ORDER)

    for row_idx, config_name in enumerate(configs):
        ax = axes[row_idx, 0]
        all_tops = []
        for model_idx, model_name in enumerate(MODEL_ORDER):
            means = []
            stds = []
            for dataset, split in cells:
                mean, std, _ = lookup.get((config_name, model_name, dataset, split), (np.nan, 0.0, 0))
                means.append(mean)
                stds.append(std)
            all_tops.extend([
                mean + std for mean, std in zip(means, stds)
                if np.isfinite(mean)
            ])
            xpos = x + (model_idx - (n_models - 1) / 2) * width
            bars = ax.bar(
                xpos,
                means,
                width=width,
                color=COLORS[model_name],
                label=MODEL_LABELS[model_name],
                yerr=stds,
                error_kw={"elinewidth": 1.0, "ecolor": "#374151", "capsize": 3},
            )
            for bar, mean in zip(bars, means):
                if np.isfinite(mean):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{mean:.2f}" if metric == "ppl" else f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=TEXT_COLOR,
                    )

        for cell_idx, (dataset, split) in enumerate(cells):
            deep = lookup.get((config_name, "deepseekv4", dataset, split))
            base = lookup.get((config_name, "param_matched", dataset, split))
            if deep and base:
                delta = deep[0] - base[0]
                y = max(deep[0] + deep[1], base[0] + base[1])
                ax.text(
                    cell_idx,
                    y * 1.015,
                    f"Δ vs PM {delta:+.2f}" if metric == "ppl" else f"Δ vs PM {delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=COLORS["deepseekv4"],
                )

        if all_tops:
            y_top = max(all_tops)
            ax.set_ylim(0, y_top * 1.18)
        ax.set_title(f"{config_name} config")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{DATASET_LABELS.get(dataset, dataset)}\n{split}" for dataset, split in cells
        ])
        clean_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.09))

    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_curves(curve_rows):
    grouped = defaultdict(list)
    for row in curve_rows:
        key = (row["config"], row["model"], row["dataset"], as_int(row, "tokens_trained"))
        grouped[key].append(as_float(row, "loss"))
    rows = []
    for key, values in grouped.items():
        config_name, model, dataset, tokens = key
        arr = np.array(values, dtype=np.float64)
        rows.append({
            "config": config_name,
            "model": model,
            "dataset": dataset,
            "tokens_trained": tokens,
            "loss_mean": float(arr.mean()),
            "loss_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        })
    return rows


def draw_validation_curves(curve_rows, output_path):
    rows = summarize_curves(curve_rows)
    configs = config_order(rows)
    datasets = [dataset for dataset in ["tiny_shakespeare", "wikitext2"] if any(row["dataset"] == dataset for row in rows)]
    fig, axes = plt.subplots(len(configs), len(datasets), figsize=(6.1 * len(datasets), 3.8 * len(configs)), squeeze=False)

    for row_idx, config_name in enumerate(configs):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            for model_name in MODEL_ORDER:
                series = sorted(
                    [
                        row for row in rows
                        if row["config"] == config_name and row["dataset"] == dataset and row["model"] == model_name
                    ],
                    key=lambda row: row["tokens_trained"],
                )
                if not series:
                    continue
                x = np.array([row["tokens_trained"] for row in series], dtype=np.float64)
                mean = np.array([row["loss_mean"] for row in series], dtype=np.float64)
                std = np.array([row["loss_std"] for row in series], dtype=np.float64)
                ax.plot(x, mean, color=COLORS[model_name], linewidth=2.0, label=MODEL_LABELS[model_name])
                ax.fill_between(x, mean - std, mean + std, color=COLORS[model_name], alpha=0.14, linewidth=0)

            native = sorted(
                [row for row in rows if row["config"] == config_name and row["dataset"] == dataset and row["model"] == "native"],
                key=lambda row: row["tokens_trained"],
            )
            deep = sorted(
                [row for row in rows if row["config"] == config_name and row["dataset"] == dataset and row["model"] == "deepseekv4"],
                key=lambda row: row["tokens_trained"],
            )
            if native and deep:
                native_by_tokens = {row["tokens_trained"]: row["loss_mean"] for row in native}
                for row in deep:
                    token = row["tokens_trained"]
                    if token in native_by_tokens and row["loss_mean"] <= native_by_tokens[token]:
                        ax.scatter([token], [row["loss_mean"]], s=38, color=COLORS["deepseekv4"], zorder=5)
                        ax.annotate(
                            f"crossover\n{token / 1000:.0f}k tokens",
                            xy=(token, row["loss_mean"]),
                            xytext=(10, -26),
                            textcoords="offset points",
                            fontsize=8,
                            color=COLORS["deepseekv4"],
                            arrowprops={"arrowstyle": "-", "color": COLORS["deepseekv4"], "linewidth": 0.9},
                        )
                        break

            ax.set_title(f"{config_name}: {DATASET_LABELS.get(dataset, dataset)}")
            ax.set_xlabel("training tokens")
            ax.set_ylabel("validation loss")
            clean_axis(ax)
            ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Validation loss curves, mean ±1σ across seeds", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_expert_heatmap(expert_rows, output_path, config_name=None, dataset=None, model="deepseekv4"):
    rows = [row for row in expert_rows if row["model"] == model]
    if config_name is not None:
        rows = [row for row in rows if row["config"] == config_name]
    if dataset is not None:
        rows = [row for row in rows if row["dataset"] == dataset]
    if not rows:
        return False

    max_tokens = max(as_int(row, "tokens_trained") for row in rows)
    rows = [row for row in rows if as_int(row, "tokens_trained") == max_tokens]
    max_layer = max(as_int(row, "layer") for row in rows)
    max_expert = max(as_int(row, "expert") for row in rows)
    matrix = np.zeros((max_layer + 1, max_expert + 1), dtype=np.float64)
    counts = np.zeros_like(matrix)
    for row in rows:
        layer = as_int(row, "layer")
        expert = as_int(row, "expert")
        matrix[layer, expert] += as_float(row, "count")
        counts[layer, expert] += 1
    matrix = np.divide(matrix, np.maximum(counts, 1))
    row_totals = matrix.sum(axis=1, keepdims=True)
    pct = 100 * matrix / np.maximum(row_totals, 1)

    fig, ax = plt.subplots(figsize=(1.0 * pct.shape[1] + 2.6, 0.62 * pct.shape[0] + 2.2))
    im = ax.imshow(pct, cmap="Reds", vmin=0, vmax=max(1, pct.max()))
    ax.set_title("DeepSeek expert utilization at final checkpoint")
    ax.set_xlabel("expert")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(pct.shape[1]))
    ax.set_yticks(np.arange(pct.shape[0]))
    for layer in range(pct.shape[0]):
        for expert in range(pct.shape[1]):
            ax.text(expert, layer, f"{pct[layer, expert]:.0f}%", ha="center", va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("share of routed assignments")
    subtitle = []
    if config_name:
        subtitle.append(config_name)
    if dataset:
        subtitle.append(DATASET_LABELS.get(dataset, dataset))
    if subtitle:
        ax.text(0, -0.18, " · ".join(subtitle), transform=ax.transAxes, fontsize=9, color="#4b5563")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def draw_ablation_panel(summary_rows, output_path, config_name=None, dataset=None, split="validation"):
    models = ["deepseekv4", "deepseekv4_no_hash", "deepseekv4_no_shared"]
    available_configs = config_order(summary_rows)
    config_name = config_name or (available_configs[-1] if available_configs else None)
    dataset = dataset or ("wikitext2" if any(row["dataset"] == "wikitext2" for row in summary_rows) else None)
    if config_name is None or dataset is None:
        return False
    rows_by_model = {
        row["model"]: row
        for row in summary_rows
        if row["config"] == config_name and row["dataset"] == dataset and row["split"] == split and row["model"] in models
    }
    if not all(model in rows_by_model for model in models):
        return False

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    means = [as_float(rows_by_model[model], "loss_mean") for model in models]
    stds = [as_float(rows_by_model[model], "loss_std") for model in models]
    x = np.arange(len(models))
    bars = ax.bar(
        x,
        means,
        color=[COLORS[model] for model in models],
        yerr=stds,
        error_kw={"elinewidth": 1.0, "ecolor": "#374151", "capsize": 3},
        width=0.58,
    )
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    full = means[0]
    for idx, mean in enumerate(means[1:], start=1):
        ax.text(
            idx,
            mean + max(stds[idx], 0) + 0.006,
            f"Δ {mean - full:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS[models[idx]],
        )
    y_min = min(mean - std for mean, std in zip(means, stds))
    y_max = max(mean + std for mean, std in zip(means, stds))
    pad = max((y_max - y_min) * 0.45, 0.015)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in models])
    ax.set_ylabel(f"{split} loss")
    ax.set_title(f"Ablation: {config_name} · {DATASET_LABELS.get(dataset, dataset)}")
    clean_axis(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/deepseekv4_full_curve")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None, help="config to prefer for heatmap/ablation")
    parser.add_argument("--dataset", default=None, help="dataset to prefer for heatmap/ablation")
    parser.add_argument("--ablation-split", default="validation", choices=["train", "train_eval", "validation", "test"])
    args = parser.parse_args()

    setup_style()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    summary = read_csv(input_dir / "summary.csv")
    curves = read_csv(input_dir / "curves.csv")
    expert_counts = read_csv(input_dir / "expert_counts.csv")

    draw_grouped_metric_bars(
        summary,
        "ppl",
        "perplexity",
        "Final perplexity, mean ±1σ across seeds",
        output_dir / "perplexity_grouped_bars.png",
    )
    draw_validation_curves(curves, output_dir / "validation_loss_curves.png")
    draw_grouped_metric_bars(
        summary,
        "bits_per_byte",
        "bits/byte",
        "Bits per byte, mean ±1σ across seeds",
        output_dir / "bits_per_byte_grouped_bars.png",
    )
    draw_expert_heatmap(
        expert_counts,
        output_dir / "expert_utilization_heatmap.png",
        config_name=args.config,
        dataset=args.dataset,
    )
    draw_ablation_panel(
        summary,
        output_dir / "ablation_panel.png",
        config_name=args.config,
        dataset=args.dataset,
        split=args.ablation_split,
    )

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
