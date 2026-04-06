# LLM Serving Framework Benchmark

Benchmarking HuggingFace Baseline vs vLLM (PagedAttention) vs SGLang (RadixAttention) on Qwen2.5-1.5B-Instruct.

## Project Structure

```
benchmark/
├── config.py                 # Shared config (model, prompts, params)
├── run_all.sh                # Run full pipeline
├── run_single.sh             # Run individual benchmarks
├── scripts/
│   ├── bench_baseline.py     # HuggingFace baseline (sequential)
│   ├── bench_vllm.py         # vLLM server + concurrent requests
│   ├── bench_sglang.py       # SGLang server ± RadixAttention
│   └── visualize.py          # Charts + optional WandB upload
└── results/                  # Output JSON + figures
```

## Environment Setup

Three separate venvs in scratch to avoid dependency conflicts:

```
/insomnia001/depts/edu/users/yy3608/envs/
├── env_baseline/   # torch 2.11 + transformers 5.5
├── env_vllm/       # vLLM 0.19 + torch 2.10
└── env_sglang/     # SGLang 0.5.10 + torch 2.9
```

## Running Benchmarks

### On a GPU node (required)

First request an interactive GPU session:
```bash
srun --pty -t 0-04:00 --gres=gpu:l40s:1 -A edu /bin/bash
```

Then run all benchmarks:
```bash
cd /insomnia001/home/yy3608/yym_self_code/benchmark
./run_all.sh
```

Or run individually:
```bash
./run_single.sh baseline      # HuggingFace baseline
./run_single.sh vllm          # vLLM (auto-launches server)
./run_single.sh sglang        # SGLang (both with/without RadixCache)
./run_single.sh visualize     # Generate charts
./run_single.sh visualize-wandb  # Charts + upload to WandB
```

## Metrics Measured

- **TTFT** (Time to First Token): Prefill latency
- **Throughput** (tokens/sec): Total output tokens / wall time
- **Latency**: Average per-request end-to-end time
- **GPU Memory**: Peak memory usage

## Benchmark Dimensions

- **Concurrency levels**: 1, 2, 4, 8, 16
- **Short prompts**: Simple Q&A (tests raw serving overhead)
- **Shared prefix prompts**: Same long prefix + different suffixes (tests KV cache reuse / RadixAttention)

## WandB Integration

To use WandB:
```bash
wandb login 
./run_single.sh visualize-wandb
```
