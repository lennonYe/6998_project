"""
SGLang benchmark: Launch SGLang server and send concurrent requests.
Tests RadixAttention's prefix caching advantage on shared-prefix prompts.
"""

import sys
import os
import time
import json
import asyncio
import argparse
import subprocess
import signal
from pathlib import Path

import aiohttp
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    MODEL_NAME, SHORT_PROMPTS, SHARED_PREFIX, SHARED_PREFIX_SUFFIXES,
    CONCURRENCY_LEVELS, MAX_NEW_TOKENS, SGLANG_PORT, DTYPE,
)

API_BASE = f"http://localhost:{SGLANG_PORT}"


def wait_for_server(url, timeout=900, proc=None):
    start = time.time()
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            log_path = getattr(proc, '_log_path', None)
            if log_path and log_path.exists():
                print(f"\n=== Server crashed! Log ({log_path}): ===")
                print(log_path.read_text()[-3000:])
            raise RuntimeError(f"Server process died with code {proc.returncode}")
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print("SGLang server is ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    if proc:
        log_path = getattr(proc, '_log_path', None)
        if log_path and log_path.exists():
            print(f"\n=== Server timeout! Last log ({log_path}): ===")
            print(log_path.read_text()[-3000:])
    raise TimeoutError(f"Server not ready after {timeout}s")


def _launch(cmd, log_name):
    log_path = Path(__file__).resolve().parent.parent / "results" / f"{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    print(f"Starting server: {' '.join(cmd)}")
    print(f"Server log: {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._log_file = log_file
    proc._log_path = log_path
    return proc


def launch_server(model, port, mem_fraction=0.80, extra_args=None):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--dtype", DTYPE,
        "--mem-fraction-static", str(mem_fraction),
        "--disable-radix-cache",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _launch(cmd, "sglang_noradix_server")


def launch_server_with_radix(model, port, mem_fraction=0.80, extra_args=None):
    """Launch SGLang with RadixAttention enabled (default behavior)."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--dtype", DTYPE,
        "--mem-fraction-static", str(mem_fraction),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _launch(cmd, "sglang_radix_server")


def kill_server(proc):
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("SGLang server stopped.")
    log_file = getattr(proc, '_log_file', None)
    if log_file:
        log_file.close()


async def send_request(session, prompt, request_id, api_base, model_name=None):
    """Send request via OpenAI-compatible API."""
    payload = {
        "model": model_name or MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft = None
    chunk_count = 0
    usage_tokens = None
    input_tokens = None

    async with session.post(
        f"{api_base}/v1/chat/completions",
        json=payload,
    ) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                    input_tokens = chunk["usage"].get("prompt_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    chunk_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    t_end = time.perf_counter()
    total_time = t_end - t_start
    output_tokens = usage_tokens if usage_tokens is not None else chunk_count

    return {
        "request_id": request_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": total_time,
        "ttft_s": ttft if ttft else total_time,
        "tokens_per_sec": output_tokens / total_time if total_time > 0 else 0,
    }


async def run_concurrent_batch(prompts, concurrency_label, api_base, model_name=None):
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        batch_start = time.perf_counter()
        tasks = [
            send_request(session, prompt, i, api_base, model_name)
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        batch_end = time.perf_counter()

    total_output_tokens = sum(r["output_tokens"] for r in results)
    batch_time = batch_end - batch_start

    return {
        "num_requests": len(prompts),
        "concurrency": concurrency_label,
        "total_time_s": batch_time,
        "avg_ttft_s": sum(r["ttft_s"] for r in results) / len(results),
        "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in results) / len(results),
        "total_throughput_tps": total_output_tokens / batch_time if batch_time > 0 else 0,
        "avg_latency_s": sum(r["total_time_s"] for r in results) / len(results),
        "per_request": results,
    }


def get_gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def run_benchmark_suite(api_base, label, model_name=None):
    """Run full benchmark suite against a given server endpoint."""
    results = {"short_prompts": {}, "shared_prefix_prompts": {}}

    print(f"\n=== [{label}] Short Prompts Benchmark ===")
    for n in CONCURRENCY_LEVELS:
        prompts = (SHORT_PROMPTS * ((n // len(SHORT_PROMPTS)) + 1))[:n]
        print(f"\nConcurrency={n}...")
        result = asyncio.run(run_concurrent_batch(prompts, n, api_base, model_name))
        result["gpu_memory_mb"] = get_gpu_memory()
        results["short_prompts"][str(n)] = result
        print(f"  Total time: {result['total_time_s']:.2f}s")
        print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
        print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")

    print(f"\n=== [{label}] Shared Prefix Prompts Benchmark ===")
    for n in CONCURRENCY_LEVELS:
        suffixes = (SHARED_PREFIX_SUFFIXES * ((n // len(SHARED_PREFIX_SUFFIXES)) + 1))[:n]
        prompts = [SHARED_PREFIX + s for s in suffixes]
        print(f"\nConcurrency={n}...")
        result = asyncio.run(run_concurrent_batch(prompts, n, api_base, model_name))
        result["gpu_memory_mb"] = get_gpu_memory()
        results["shared_prefix_prompts"][str(n)] = result
        print(f"  Total time: {result['total_time_s']:.2f}s")
        print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
        print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--output", default="results/sglang_results.json")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--radix-only", action="store_true",
                        help="Only test with RadixAttention enabled")
    parser.add_argument("--no-radix-only", action="store_true",
                        help="Only test without RadixAttention")
    parser.add_argument("--mem-fraction", type=float, default=0.80,
                        help="SGLang --mem-fraction-static (lowered for L4 24GB)")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Pass --disable-cuda-graph to SGLang server (faster boot, slightly slower inference; needed for AWQ on L4)")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "framework": "sglang",
        "model": args.model,
        "dtype": DTYPE,
    }

    extra_server_args = ["--disable-cuda-graph"] if args.disable_cuda_graph else None

    # --- Test WITHOUT RadixAttention (for fair comparison with vLLM) ---
    if not args.radix_only:
        proc = None
        if not args.no_launch:
            proc = launch_server(args.model, args.port,
                                 mem_fraction=args.mem_fraction,
                                 extra_args=extra_server_args)
        try:
            api_base = f"http://localhost:{args.port}"
            wait_for_server(api_base, proc=proc)
            all_results["gpu_memory_serving_mb"] = get_gpu_memory()
            all_results["no_radix"] = run_benchmark_suite(
                api_base, "SGLang (no RadixCache)", model_name=args.model)
        finally:
            if proc:
                kill_server(proc)
                time.sleep(5)  # wait for GPU memory to free

    # --- Test WITH RadixAttention ---
    if not args.no_radix_only:
        proc = None
        if not args.no_launch:
            proc = launch_server_with_radix(args.model, args.port,
                                            mem_fraction=args.mem_fraction,
                                            extra_args=extra_server_args)
        try:
            api_base = f"http://localhost:{args.port}"
            wait_for_server(api_base, proc=proc)
            all_results["gpu_memory_serving_radix_mb"] = get_gpu_memory()
            all_results["with_radix"] = run_benchmark_suite(
                api_base, "SGLang (RadixAttention)", model_name=args.model)
        finally:
            if proc:
                kill_server(proc)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
