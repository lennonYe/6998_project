"""
Baseline benchmark: Standard HuggingFace Transformers inference.
Measures single-request and concurrent (sequential) performance.
"""

import sys
import os
import time
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    MODEL_NAME, SHORT_PROMPTS, SHARED_PREFIX, SHARED_PREFIX_SUFFIXES,
    CONCURRENCY_LEVELS, MAX_NEW_TOKENS, TEMPERATURE, DTYPE,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_max_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def generate_one(model, tokenizer, prompt, max_new_tokens):
    """Generate a single response and return timing info."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    output_tokens = outputs[0][input_len:]
    output_len = len(output_tokens)
    elapsed = t1 - t0
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    return {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_time_s": elapsed,
        "ttft_s": elapsed,  # for single request, TTFT approximates total time for first token
        "tokens_per_sec": output_len / elapsed if elapsed > 0 else 0,
        "text": text,
    }


def generate_one_streaming(model, tokenizer, prompt, max_new_tokens):
    """Generate with manual token-by-token loop to measure TTFT accurately."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    attention_mask = inputs.get("attention_mask")

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    generated_ids = input_ids.clone()
    ttft = None
    past_key_values = None

    for i in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                out = model(
                    input_ids=generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                out = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            past_key_values = out.past_key_values

        torch.cuda.synchronize()

        if ttft is None:
            ttft = time.perf_counter() - t_start

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=model.device, dtype=attention_mask.dtype)],
            dim=-1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_ids = generated_ids[0][input_len:]
    output_len = len(output_ids)
    total_time = t_end - t_start
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_time_s": total_time,
        "ttft_s": ttft,
        "tokens_per_sec": output_len / total_time if total_time > 0 else 0,
        "text": text,
    }


def run_sequential_batch(model, tokenizer, prompts, max_new_tokens):
    """Run prompts sequentially (simulating concurrency=N with no batching)."""
    results = []
    batch_start = time.perf_counter()
    for prompt in prompts:
        r = generate_one_streaming(model, tokenizer, prompt, max_new_tokens)
        results.append(r)
    batch_end = time.perf_counter()

    total_output_tokens = sum(r["output_tokens"] for r in results)
    batch_time = batch_end - batch_start

    return {
        "num_requests": len(prompts),
        "total_time_s": batch_time,
        "avg_ttft_s": sum(r["ttft_s"] for r in results) / len(results),
        "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in results) / len(results),
        "total_throughput_tps": total_output_tokens / batch_time if batch_time > 0 else 0,
        "avg_latency_s": batch_time / len(prompts),
        "per_request": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output", default="results/baseline_results.json")
    parser.add_argument("--use-generate", action="store_true",
                        help="Use model.generate() instead of manual loop")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    mem_after_load = get_gpu_memory_mb()
    print(f"GPU memory after model load: {mem_after_load:.1f} MB")

    gen_fn = generate_one if args.use_generate else generate_one_streaming
    all_results = {
        "framework": "huggingface_baseline",
        "model": args.model,
        "dtype": DTYPE,
        "gpu_memory_model_mb": mem_after_load,
        "short_prompts": {},
        "shared_prefix_prompts": {},
    }

    # --- Short prompts benchmark ---
    print("\n=== Short Prompts Benchmark ===")
    for n in CONCURRENCY_LEVELS:
        prompts = (SHORT_PROMPTS * ((n // len(SHORT_PROMPTS)) + 1))[:n]
        print(f"\nConcurrency={n} ({len(prompts)} requests, sequential)...")
        torch.cuda.reset_peak_memory_stats()

        result = run_sequential_batch(model, tokenizer, prompts, args.max_tokens)
        result["peak_gpu_memory_mb"] = get_gpu_max_memory_mb()

        # Remove full text from summary
        for r in result["per_request"]:
            r.pop("text", None)

        all_results["short_prompts"][str(n)] = result
        print(f"  Total time: {result['total_time_s']:.2f}s")
        print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
        print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")
        print(f"  Peak GPU mem: {result['peak_gpu_memory_mb']:.1f} MB")

    # --- Shared prefix prompts benchmark ---
    print("\n=== Shared Prefix Prompts Benchmark ===")
    for n in CONCURRENCY_LEVELS:
        suffixes = (SHARED_PREFIX_SUFFIXES * ((n // len(SHARED_PREFIX_SUFFIXES)) + 1))[:n]
        prompts = [SHARED_PREFIX + s for s in suffixes]
        print(f"\nConcurrency={n} ({len(prompts)} requests, sequential)...")
        torch.cuda.reset_peak_memory_stats()

        result = run_sequential_batch(model, tokenizer, prompts, args.max_tokens)
        result["peak_gpu_memory_mb"] = get_gpu_max_memory_mb()

        for r in result["per_request"]:
            r.pop("text", None)

        all_results["shared_prefix_prompts"][str(n)] = result
        print(f"  Total time: {result['total_time_s']:.2f}s")
        print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
        print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")
        print(f"  Peak GPU mem: {result['peak_gpu_memory_mb']:.1f} MB")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
