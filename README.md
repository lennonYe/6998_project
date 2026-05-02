# LLM Serving Framework Benchmark

COMS 6998 HPML — Benchmarking & Profiling High-Throughput Serving Frameworks for LLMs.

Compares **HuggingFace baseline** vs **vLLM (PagedAttention)** vs **SGLang (RadixAttention)** on
latency, throughput, and GPU memory under concurrent loads.

## Team

| Name | UNI |
| --- | --- |
| Yimeng Ye | yy3608 |
| Can Yang | cy2811 |
| Yuxuan Wang | yw4609 |
| Jiawei Wang | jw4807 |
| Zhijing Wu | zw3155 |

## WandB Dashboard

- Project: https://wandb.ai/shvpz8ctzd-1/llm-serving-benchmark
- Qwen2.5-1.5B-FP16 run: https://wandb.ai/shvpz8ctzd-1/llm-serving-benchmark/runs/oscrou8i

Each (model × quantization) combo produces its own WandB run, tagged with the label.

## Hardware Note (L4 vs L40S)

The mid-report numbers were collected on **NVIDIA L40S (48 GB)**. This branch (`l4-rerun`)
re-runs every experiment on **NVIDIA L4 (24 GB)**. Throughput numbers are lower
than the L40S baseline (L4 has roughly one-third the FP16 compute) — that is expected and
discussed in the final report.

## Project Structure

```
benchmark/
├── config.py                 # Shared config (models, prompts, params)
├── run_all.sh                # Run full pipeline
├── run_single.sh             # Run individual benchmarks
├── scripts/
│   ├── bench_baseline.py     # HuggingFace baseline (sequential)
│   ├── bench_vllm.py         # vLLM server + concurrent requests
│   ├── bench_sglang.py       # SGLang server ± RadixAttention
│   ├── bench_profile.py      # PyTorch Profiler (kernel-level)
│   └── visualize.py          # Charts + WandB upload
└── results/                  # Output JSON + figures
```

## Environment Setup (L4)

Three separate venvs (different frameworks have conflicting torch versions):

```
~/yym_self_project/envs/
├── env_baseline/   # torch + transformers + bitsandbytes (INT8 baseline)
├── env_vllm/       # vLLM (supports AWQ/FP8 quant)
└── env_sglang/     # SGLang
```

Create with:

```bash
mkdir -p ~/yym_self_project/envs
for env in env_baseline env_vllm env_sglang; do
    python3 -m venv ~/yym_self_project/envs/$env
done

source ~/yym_self_project/envs/env_baseline/bin/activate
pip install torch transformers accelerate bitsandbytes matplotlib wandb aiohttp requests

source ~/yym_self_project/envs/env_vllm/bin/activate
pip install vllm aiohttp requests

source ~/yym_self_project/envs/env_sglang/bin/activate
pip install "sglang[all]" aiohttp requests
```

## Secrets

Copy `.env.example` → `.env` (gitignored) and fill in:

```bash
cp .env.example .env
# edit .env to add WANDB_API_KEY and HF_TOKEN
set -a && source .env && set +a
wandb login "$WANDB_API_KEY"
huggingface-cli login --token "$HF_TOKEN"   # needed for gated models like Llama-3.2
```

## Running Benchmarks

```bash
cd benchmark
./run_all.sh                        # full pipeline
./run_single.sh baseline            # HuggingFace baseline only
./run_single.sh vllm                # vLLM only
./run_single.sh sglang              # SGLang (both with/without RadixCache)
./run_single.sh visualize           # charts only
./run_single.sh visualize-wandb     # charts + WandB upload
```

## Metrics Measured

- **TTFT** (Time to First Token): prefill latency
- **Throughput** (tokens/sec): total output tokens / wall time
- **Latency**: average per-request end-to-end time
- **GPU Memory**: peak memory usage

## Benchmark Dimensions

- **Concurrency**: 1, 2, 4, 8, 16
- **Models**: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-1B-Instruct
- **Quantization**: FP16 (baseline), INT4 (AWQ), FP8 (vLLM)
- **Prompt types**: short Q&A vs shared-prefix code-analysis (tests RadixAttention)
