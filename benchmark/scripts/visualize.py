"""
Visualize benchmark results: generate comparison charts and upload to WandB.
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCURRENCY_LEVELS, WANDB_PROJECT


def load_results(results_dir):
    """Load all result JSON files."""
    data = {}
    for name in ["baseline_results", "vllm_results", "sglang_results"]:
        path = results_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                data[name.replace("_results", "")] = json.load(f)
            print(f"Loaded {path}")
        else:
            print(f"Warning: {path} not found, skipping.")
    return data


def extract_metrics(data, prompt_type="short_prompts"):
    """Extract metrics per framework per concurrency level."""
    frameworks = {}

    if "baseline" in data:
        fw = data["baseline"]
        metrics = {"throughput": [], "ttft": [], "latency": [], "concurrency": []}
        for n in CONCURRENCY_LEVELS:
            key = str(n)
            if key in fw.get(prompt_type, {}):
                r = fw[prompt_type][key]
                metrics["concurrency"].append(n)
                metrics["throughput"].append(r["total_throughput_tps"])
                metrics["ttft"].append(r["avg_ttft_s"] * 1000)
                metrics["latency"].append(r["avg_latency_s"] * 1000)
        if metrics["concurrency"]:
            frameworks["HuggingFace Baseline"] = metrics

    if "vllm" in data:
        fw = data["vllm"]
        metrics = {"throughput": [], "ttft": [], "latency": [], "concurrency": []}
        for n in CONCURRENCY_LEVELS:
            key = str(n)
            if key in fw.get(prompt_type, {}):
                r = fw[prompt_type][key]
                metrics["concurrency"].append(n)
                metrics["throughput"].append(r["total_throughput_tps"])
                metrics["ttft"].append(r["avg_ttft_s"] * 1000)
                metrics["latency"].append(r["avg_latency_s"] * 1000)
        if metrics["concurrency"]:
            frameworks["vLLM (PagedAttention)"] = metrics

    if "sglang" in data:
        fw = data["sglang"]
        for variant_key, label in [
            ("no_radix", "SGLang (no RadixCache)"),
            ("with_radix", "SGLang (RadixAttention)"),
        ]:
            if variant_key not in fw:
                continue
            variant = fw[variant_key]
            metrics = {"throughput": [], "ttft": [], "latency": [], "concurrency": []}
            for n in CONCURRENCY_LEVELS:
                key = str(n)
                if key in variant.get(prompt_type, {}):
                    r = variant[prompt_type][key]
                    metrics["concurrency"].append(n)
                    metrics["throughput"].append(r["total_throughput_tps"])
                    metrics["ttft"].append(r["avg_ttft_s"] * 1000)
                    metrics["latency"].append(r["avg_latency_s"] * 1000)
            if metrics["concurrency"]:
                frameworks[label] = metrics

    return frameworks


def plot_metric(frameworks, metric_key, ylabel, title, save_path, log_scale=False):
    """Generic plotting function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, metrics) in enumerate(frameworks.items()):
        ax.plot(
            metrics["concurrency"],
            metrics[metric_key],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=name,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(CONCURRENCY_LEVELS)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_gpu_memory(data, save_path):
    """Bar chart comparing GPU memory usage across frameworks."""
    labels = []
    values = []

    if "baseline" in data:
        mem = data["baseline"].get("gpu_memory_model_mb", 0)
        if mem:
            labels.append("HF Baseline")
            values.append(mem)

    if "vllm" in data:
        mem = data["vllm"].get("gpu_memory_serving_mb", 0)
        if mem:
            labels.append("vLLM")
            values.append(mem)

    if "sglang" in data:
        mem = data["sglang"].get("gpu_memory_serving_mb", 0)
        if mem:
            labels.append("SGLang")
            values.append(mem)
        mem_r = data["sglang"].get("gpu_memory_serving_radix_mb", 0)
        if mem_r:
            labels.append("SGLang (Radix)")
            values.append(mem_r)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:.0f} MB", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage by Framework", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def upload_to_wandb(data, figures_dir):
    """Upload results and figures to WandB."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping upload.")
        return

    run = wandb.init(project=WANDB_PROJECT, name="benchmark-comparison", reinit=True)

    # Log summary metrics
    summary = {}
    for fw_name, fw_data in data.items():
        if fw_name == "baseline":
            for pt in ["short_prompts", "shared_prefix_prompts"]:
                if "16" in fw_data.get(pt, {}):
                    r = fw_data[pt]["16"]
                    prefix = f"baseline/{pt}"
                    summary[f"{prefix}/throughput_tps"] = r["total_throughput_tps"]
                    summary[f"{prefix}/avg_ttft_ms"] = r["avg_ttft_s"] * 1000
        elif fw_name == "vllm":
            for pt in ["short_prompts", "shared_prefix_prompts"]:
                if "16" in fw_data.get(pt, {}):
                    r = fw_data[pt]["16"]
                    prefix = f"vllm/{pt}"
                    summary[f"{prefix}/throughput_tps"] = r["total_throughput_tps"]
                    summary[f"{prefix}/avg_ttft_ms"] = r["avg_ttft_s"] * 1000
        elif fw_name == "sglang":
            for variant in ["no_radix", "with_radix"]:
                if variant not in fw_data:
                    continue
                for pt in ["short_prompts", "shared_prefix_prompts"]:
                    if "16" in fw_data[variant].get(pt, {}):
                        r = fw_data[variant][pt]["16"]
                        prefix = f"sglang_{variant}/{pt}"
                        summary[f"{prefix}/throughput_tps"] = r["total_throughput_tps"]
                        summary[f"{prefix}/avg_ttft_ms"] = r["avg_ttft_s"] * 1000

    wandb.log(summary)

    # Log figures
    for img_path in sorted(figures_dir.glob("*.png")):
        wandb.log({img_path.stem: wandb.Image(str(img_path))})

    run.finish()
    print("Results uploaded to WandB.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / args.results_dir
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_dir)
    if not data:
        print("No results found. Run benchmarks first.")
        return

    # --- Short prompts ---
    fw_short = extract_metrics(data, "short_prompts")
    if fw_short:
        plot_metric(fw_short, "throughput",
                    "Throughput (tokens/sec)", "Throughput vs Concurrency (Short Prompts)",
                    figures_dir / "throughput_short.png")
        plot_metric(fw_short, "ttft",
                    "TTFT (ms)", "Time to First Token vs Concurrency (Short Prompts)",
                    figures_dir / "ttft_short.png")
        plot_metric(fw_short, "latency",
                    "Avg Latency (ms)", "Average Latency vs Concurrency (Short Prompts)",
                    figures_dir / "latency_short.png")

    # --- Shared prefix prompts ---
    fw_prefix = extract_metrics(data, "shared_prefix_prompts")
    if fw_prefix:
        plot_metric(fw_prefix, "throughput",
                    "Throughput (tokens/sec)", "Throughput vs Concurrency (Shared Prefix)",
                    figures_dir / "throughput_prefix.png")
        plot_metric(fw_prefix, "ttft",
                    "TTFT (ms)", "Time to First Token vs Concurrency (Shared Prefix)",
                    figures_dir / "ttft_prefix.png")
        plot_metric(fw_prefix, "latency",
                    "Avg Latency (ms)", "Average Latency vs Concurrency (Shared Prefix)",
                    figures_dir / "latency_prefix.png")

    # --- GPU Memory ---
    plot_gpu_memory(data, figures_dir / "gpu_memory.png")

    # --- WandB upload ---
    if not args.no_wandb:
        upload_to_wandb(data, figures_dir)

    print("\nDone! Figures saved to:", figures_dir)


if __name__ == "__main__":
    main()
