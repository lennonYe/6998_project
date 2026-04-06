#!/bin/bash
# ============================================================
# LLM Serving Framework Benchmark - Full Pipeline
# Run this on a GPU node (e.g., L40S via srun)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_DIR="/insomnia001/depts/edu/users/yy3608/envs"

# Check GPU
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================"
echo "Step 1: Baseline HuggingFace Benchmark"
echo "============================================"
source "$ENV_DIR/env_baseline/bin/activate"
python scripts/bench_baseline.py --model "$MODEL" --output results/baseline_results.json
deactivate

echo ""
echo "============================================"
echo "Step 2: vLLM Benchmark"
echo "============================================"
source "$ENV_DIR/env_vllm/bin/activate"
python scripts/bench_vllm.py --model "$MODEL" --output results/vllm_results.json
deactivate

echo ""
echo "============================================"
echo "Step 3: SGLang Benchmark"
echo "============================================"
source "$ENV_DIR/env_sglang/bin/activate"
python scripts/bench_sglang.py --model "$MODEL" --output results/sglang_results.json
deactivate

echo ""
echo "============================================"
echo "Step 4: Generate Visualizations"
echo "============================================"
source "$ENV_DIR/env_baseline/bin/activate"
python scripts/visualize.py --results-dir results --no-wandb

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "Results: $SCRIPT_DIR/results/"
echo "Figures: $SCRIPT_DIR/results/figures/"
echo "============================================"
