"""
vLLM benchmark: Launch vLLM server and send concurrent requests via OpenAI-compatible API.
Measures TTFT, throughput, and GPU memory under different concurrency levels.
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
from config import (
    MODEL_NAME, SHORT_PROMPTS, SHARED_PREFIX, SHARED_PREFIX_SUFFIXES,
    CONCURRENCY_LEVELS, MAX_NEW_TOKENS, VLLM_PORT, DTYPE,
)

API_BASE = f"http://localhost:{VLLM_PORT}"


def wait_for_server(url, timeout=300, proc=None):
    """Wait for the server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        # Check if server process died
        if proc and proc.poll() is not None:
            log_path = getattr(proc, '_log_path', None)
            if log_path and log_path.exists():
                print(f"\n=== Server crashed! Log ({log_path}): ===")
                print(log_path.read_text()[-3000:])
            raise RuntimeError(f"Server process died with code {proc.returncode}")
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print("vLLM server is ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    # Timeout - print logs
    if proc:
        log_path = getattr(proc, '_log_path', None)
        if log_path and log_path.exists():
            print(f"\n=== Server timeout! Last log ({log_path}): ===")
            print(log_path.read_text()[-3000:])
    raise TimeoutError(f"Server not ready after {timeout}s")


def launch_server(model, port, max_model_len=4096, gpu_mem_util=0.85,
                  quantization=None, extra_args=None):
    """Launch vLLM server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--no-enable-log-requests",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    if extra_args:
        cmd.extend(extra_args)

    log_path = Path(__file__).resolve().parent.parent / "results" / "vllm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    print(f"Starting vLLM server: {' '.join(cmd)}")
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


def kill_server(proc):
    """Kill the server process group."""
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("vLLM server stopped.")
    log_file = getattr(proc, '_log_file', None)
    if log_file:
        log_file.close()


async def send_request(session, prompt, request_id, model_name=None):
    """Send a single chat completion request and measure timing."""
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
        f"{API_BASE}/v1/chat/completions",
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
                # Final usage chunk has empty choices
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


async def run_concurrent_batch(prompts, concurrency_label, model_name=None):
    """Send all prompts concurrently and gather results."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        batch_start = time.perf_counter()
        tasks = [
            send_request(session, prompt, i, model_name)
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


def get_vllm_gpu_memory():
    """Query GPU memory from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--port", type=int, default=VLLM_PORT)
    parser.add_argument("--output", default="results/vllm_results.json")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't launch server (assume already running)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--quantization", default=None,
                        help="vLLM quantization arg, e.g. 'awq', 'fp8', 'gptq'")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global API_BASE
    API_BASE = f"http://localhost:{args.port}"

    proc = None
    if not args.no_launch:
        proc = launch_server(
            args.model, args.port,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_memory_utilization,
            quantization=args.quantization,
        )

    try:
        wait_for_server(API_BASE, proc=proc)
        gpu_mem_serving = get_vllm_gpu_memory()
        print(f"GPU memory (serving): {gpu_mem_serving:.1f} MB")

        all_results = {
            "framework": "vllm",
            "model": args.model,
            "dtype": DTYPE,
            "quantization": args.quantization,
            "gpu_memory_serving_mb": gpu_mem_serving,
            "short_prompts": {},
            "shared_prefix_prompts": {},
        }

        # --- Short prompts ---
        print("\n=== Short Prompts Benchmark ===")
        for n in CONCURRENCY_LEVELS:
            prompts = (SHORT_PROMPTS * ((n // len(SHORT_PROMPTS)) + 1))[:n]
            print(f"\nConcurrency={n}...")
            result = asyncio.run(run_concurrent_batch(prompts, n, args.model))
            result["gpu_memory_mb"] = get_vllm_gpu_memory()

            all_results["short_prompts"][str(n)] = result
            print(f"  Total time: {result['total_time_s']:.2f}s")
            print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
            print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")

        # --- Shared prefix prompts ---
        print("\n=== Shared Prefix Prompts Benchmark ===")
        for n in CONCURRENCY_LEVELS:
            suffixes = (SHARED_PREFIX_SUFFIXES * ((n // len(SHARED_PREFIX_SUFFIXES)) + 1))[:n]
            prompts = [SHARED_PREFIX + s for s in suffixes]
            print(f"\nConcurrency={n}...")
            result = asyncio.run(run_concurrent_batch(prompts, n, args.model))
            result["gpu_memory_mb"] = get_vllm_gpu_memory()

            all_results["shared_prefix_prompts"][str(n)] = result
            print(f"  Total time: {result['total_time_s']:.2f}s")
            print(f"  Avg TTFT: {result['avg_ttft_s']*1000:.1f}ms")
            print(f"  Throughput: {result['total_throughput_tps']:.1f} tok/s")

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    finally:
        if proc:
            kill_server(proc)


if __name__ == "__main__":
    main()
