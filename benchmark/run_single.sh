#!/bin/bash
# ============================================================
# Run a single benchmark (baseline / vllm / sglang / visualize)
# Usage: ./run_single.sh baseline|vllm|sglang|visualize
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_DIR="/insomnia001/depts/edu/users/yy3608/envs"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

case "$1" in
    baseline)
        echo "Running baseline benchmark..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python scripts/bench_baseline.py --model "$MODEL"
        ;;
    vllm)
        echo "Running vLLM benchmark..."
        source "$ENV_DIR/env_vllm/bin/activate"
        python scripts/bench_vllm.py --model "$MODEL"
        ;;
    sglang)
        echo "Running SGLang benchmark..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python scripts/bench_sglang.py --model "$MODEL"
        ;;
    sglang-radix)
        echo "Running SGLang benchmark (RadixAttention only)..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python scripts/bench_sglang.py --model "$MODEL" --radix-only
        ;;
    sglang-noradix)
        echo "Running SGLang benchmark (no RadixAttention only)..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python scripts/bench_sglang.py --model "$MODEL" --no-radix-only
        ;;
    visualize)
        echo "Generating visualizations..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python scripts/visualize.py --no-wandb
        ;;
    visualize-wandb)
        echo "Generating visualizations + WandB upload..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python scripts/visualize.py
        ;;
    *)
        echo "Usage: $0 {baseline|vllm|sglang|sglang-radix|sglang-noradix|visualize|visualize-wandb}"
        exit 1
        ;;
esac
