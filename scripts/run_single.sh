#!/bin/bash
# ============================================================
# Run a single benchmark
# Usage: ./run_single.sh baseline|vllm|sglang|sglang-radix|sglang-noradix|visualize|visualize-wandb|profile
# Override model:    MODEL=meta-llama/Llama-3.2-1B-Instruct ./run_single.sh vllm
# Override env dir:  ENV_DIR=/path/to/envs ./run_single.sh baseline
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR="${ENV_DIR:-$HOME/yym_self_project/envs}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

case "$1" in
    baseline)
        echo "Running baseline benchmark on $MODEL..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python src/bench_baseline.py --model "$MODEL"
        ;;
    vllm)
        echo "Running vLLM benchmark on $MODEL..."
        source "$ENV_DIR/env_vllm/bin/activate"
        python src/bench_vllm.py --model "$MODEL"
        ;;
    sglang)
        echo "Running SGLang benchmark on $MODEL..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python src/bench_sglang.py --model "$MODEL"
        ;;
    sglang-radix)
        echo "Running SGLang benchmark (RadixAttention only)..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python src/bench_sglang.py --model "$MODEL" --radix-only
        ;;
    sglang-noradix)
        echo "Running SGLang benchmark (no RadixAttention only)..."
        source "$ENV_DIR/env_sglang/bin/activate"
        python src/bench_sglang.py --model "$MODEL" --no-radix-only
        ;;
    profile)
        echo "Running PyTorch Profiler on baseline..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python src/bench_profile.py --model "$MODEL"
        ;;
    visualize)
        echo "Generating visualizations..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python src/visualize.py --no-wandb
        ;;
    visualize-wandb)
        echo "Generating visualizations + WandB upload..."
        source "$ENV_DIR/env_baseline/bin/activate"
        python src/visualize.py
        ;;
    *)
        echo "Usage: $0 {baseline|vllm|sglang|sglang-radix|sglang-noradix|profile|visualize|visualize-wandb}"
        exit 1
        ;;
esac
