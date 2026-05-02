"""
PyTorch Profiler — kernel-level analysis of the HF baseline path.

We only profile the baseline HF path (vLLM/SGLang run in separate server processes
which would require more invasive instrumentation). The goal is to understand
where time goes inside attention / matmul kernels for a single forward pass.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, SHARED_PREFIX, MAX_NEW_TOKENS  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Token budget for the profiled generation (kept small for trace size)")
    parser.add_argument("--output-dir", default="results/profiler")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    prompt = SHARED_PREFIX + "Focus on bug detection only."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    trace_path = out_dir / "baseline_trace.json"
    table_path = out_dir / "baseline_top_kernels.csv"

    print(f"Profiling {args.max_tokens} tokens of generation...")
    prof_schedule = schedule(wait=0, warmup=1, active=1, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(2):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace: {trace_path}")

    # Top kernels by CUDA self-time
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20)
    print("\n=== Top 20 GPU kernels (self CUDA time) ===")
    print(table)

    # Write CSV of top kernels (machine-readable).
    # Newer torch renamed self_cuda_time_total → self_device_time_total.
    def _self_dev(k):
        return getattr(k, "self_device_time_total", None) or getattr(k, "self_cuda_time_total", 0)

    def _dev_total(k):
        return getattr(k, "device_time_total", None) or getattr(k, "cuda_time_total", 0)

    keyavgs = sorted(prof.key_averages(), key=_self_dev, reverse=True)[:30]
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "kernel", "calls", "self_device_us", "device_total_us",
            "self_cpu_us", "cpu_total_us",
        ])
        for k in keyavgs:
            w.writerow([
                k.key, k.count,
                int(_self_dev(k)), int(_dev_total(k)),
                int(k.self_cpu_time_total), int(k.cpu_time_total),
            ])
    print(f"Top-kernel CSV: {table_path}")


if __name__ == "__main__":
    main()
