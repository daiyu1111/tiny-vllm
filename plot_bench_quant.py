import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Nano-vLLM benchmark JSON results.")
    parser.add_argument("input", help="Path to bench_quant JSON report.")
    parser.add_argument(
        "--output-dir",
        help="Directory to write plots into. Defaults to <input_stem>_plots beside the JSON file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI for saved plots.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def successful_results(report: dict) -> list[dict]:
    return [result for result in report.get("results", []) if result.get("status") == "ok"]


def normalize_label(result: dict) -> str:
    mode = result["mode"]
    backend = result.get("backend")
    return mode if backend in (None, "") else f"{mode}-{backend}"


def metric_title(metric: str) -> str:
    titles = {
        "throughput": "Total Throughput",
        "prefill_tps": "Prefill Throughput",
        "decode_tps": "Decode Throughput",
        "ttft_seconds": "TTFT",
        "runtime_peak_gib": "Runtime Peak Memory",
    }
    return titles.get(metric, metric)


def metric_unit(metric: str) -> str:
    units = {
        "throughput": "tok/s",
        "prefill_tps": "tok/s",
        "decode_tps": "tok/s",
        "ttft_seconds": "s",
        "runtime_peak_gib": "GiB",
    }
    return units.get(metric, "")


def result_metric(result: dict, metric: str) -> float:
    if metric == "runtime_peak_gib":
        return result["runtime_memory_gib"]["max_memory_allocated"]
    return result[metric]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_context(report: dict) -> str:
    workload = report.get("workload", {})
    profile = workload.get("profile", "unknown")
    return f"{report.get('report_type', 'report')} | profile={profile}"


def plot_single(report: dict, results: list[dict], output_dir: Path, dpi: int):
    import matplotlib.pyplot as plt

    labels = [normalize_label(result) for result in results]
    metrics = ["throughput", "prefill_tps", "decode_tps", "ttft_seconds", "runtime_peak_gib"]

    for metric in metrics:
        values = [result_metric(result, metric) for result in results]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color=["#4E79A7", "#F28E2B", "#59A14F"][: len(labels)])
        ax.set_title(f"{metric_title(metric)} ({report_context(report)})")
        ax.set_ylabel(metric_unit(metric))
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=dpi)
        plt.close(fig)


def grouped_sweep(results: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for result in results:
        grouped.setdefault(normalize_label(result), []).append(result)
    for label in grouped:
        grouped[label] = sorted(
            grouped[label],
            key=lambda item: (
                item["config"]["max_num_seqs"],
                item["config"]["max_num_batched_tokens"],
            ),
        )
    return grouped


def point_label(result: dict) -> str:
    config = result["config"]
    return f"seqs={config['max_num_seqs']}\ntoks={config['max_num_batched_tokens']}"


def plot_sweep_lines(report: dict, results: list[dict], output_dir: Path, dpi: int):
    import matplotlib.pyplot as plt

    grouped = grouped_sweep(results)
    metrics = ["throughput", "prefill_tps", "decode_tps", "ttft_seconds", "runtime_peak_gib"]

    all_points = sorted(
        {
            (result["config"]["max_num_seqs"], result["config"]["max_num_batched_tokens"])
            for result in results
        }
    )
    x_labels = [f"seqs={seqs}\ntoks={tokens}" for seqs, tokens in all_points]
    x_positions = list(range(len(all_points)))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for label, series in grouped.items():
            value_map = {
                (item["config"]["max_num_seqs"], item["config"]["max_num_batched_tokens"]): result_metric(item, metric)
                for item in series
            }
            y_values = [value_map.get(point) for point in all_points]
            ax.plot(x_positions, y_values, marker="o", linewidth=2, label=label)
            for xpos, value in zip(x_positions, y_values):
                if value is not None:
                    ax.text(xpos, value, f"{value:.1f}", fontsize=8, ha="center", va="bottom")
        ax.set_title(f"{metric_title(metric)} ({report_context(report)})")
        ax.set_ylabel(metric_unit(metric))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_sweep.png", dpi=dpi)
        plt.close(fig)


def plot_sweep_heatmaps(report: dict, results: list[dict], output_dir: Path, dpi: int):
    import matplotlib.pyplot as plt

    metrics = ["throughput", "prefill_tps", "decode_tps", "ttft_seconds"]
    grouped = grouped_sweep(results)
    max_num_seqs_values = sorted({result["config"]["max_num_seqs"] for result in results})
    max_num_batched_tokens_values = sorted({result["config"]["max_num_batched_tokens"] for result in results})

    for label, series in grouped.items():
        for metric in metrics:
            value_map = {
                (item["config"]["max_num_seqs"], item["config"]["max_num_batched_tokens"]): result_metric(item, metric)
                for item in series
            }
            matrix = []
            for seqs in max_num_seqs_values:
                row = []
                for tokens in max_num_batched_tokens_values:
                    row.append(value_map.get((seqs, tokens), float("nan")))
                matrix.append(row)

            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(matrix, aspect="auto")
            ax.set_title(f"{label}: {metric_title(metric)} ({report_context(report)})")
            ax.set_xlabel("max_num_batched_tokens")
            ax.set_ylabel("max_num_seqs")
            ax.set_xticks(range(len(max_num_batched_tokens_values)))
            ax.set_xticklabels([str(value) for value in max_num_batched_tokens_values])
            ax.set_yticks(range(len(max_num_seqs_values)))
            ax.set_yticklabels([str(value) for value in max_num_seqs_values])
            for row_idx, seqs in enumerate(max_num_seqs_values):
                for col_idx, tokens in enumerate(max_num_batched_tokens_values):
                    value = value_map.get((seqs, tokens))
                    if value is not None:
                        ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color="white", fontsize=9)
            fig.colorbar(image, ax=ax, shrink=0.9, label=metric_unit(metric))
            fig.tight_layout()
            safe_label = label.replace("[", "_").replace("]", "").replace("=", "_")
            fig.savefig(output_dir / f"{safe_label}_{metric}_heatmap.png", dpi=dpi)
            plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input)
    report = load_report(input_path)
    results = successful_results(report)
    if not results:
        raise SystemExit("No successful benchmark results found in the JSON report.")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.stem}_plots")
    ensure_output_dir(output_dir)

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first, for example: pip install matplotlib"
        ) from exc

    if report.get("report_type") == "sweep":
        plot_sweep_lines(report, results, output_dir, args.dpi)
        plot_sweep_heatmaps(report, results, output_dir, args.dpi)
    else:
        plot_single(report, results, output_dir, args.dpi)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
