#!/bin/bash
# ============================================================
# Run baseline + vLLM + SGLang across all models in MODELS list.
# Each (framework, model) combo writes its own results file under
# results/{label}/{framework}_results.json.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_DIR="${ENV_DIR:-$HOME/yym_self_project/envs}"

if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../.env"
    set +a
fi

# label, hf_id, quant flag (empty = none)
MODELS_TSV=$(cat <<'EOF'
Qwen2.5-1.5B-FP16	Qwen/Qwen2.5-1.5B-Instruct
Qwen2.5-3B-FP16	Qwen/Qwen2.5-3B-Instruct
Llama-3.2-1B-FP16	meta-llama/Llama-3.2-1B-Instruct
Qwen2.5-1.5B-AWQ-INT4	Qwen/Qwen2.5-1.5B-Instruct-AWQ	awq
EOF
)

while IFS=$'\t' read -r LABEL HF_ID QUANT; do
    [ -z "$LABEL" ] && continue
    OUT_DIR="results/$LABEL"
    mkdir -p "$OUT_DIR"

    echo "============================================"
    echo " Model: $LABEL ($HF_ID${QUANT:+ — quant=$QUANT})"
    echo "============================================"

    # AWQ skipped for HF baseline (bitsandbytes path is different — keep baseline FP16 only).
    if [ -z "$QUANT" ]; then
        echo "--- baseline ---"
        source "$ENV_DIR/env_baseline/bin/activate"
        python scripts/bench_baseline.py --model "$HF_ID" --output "$OUT_DIR/baseline_results.json"
        deactivate
    fi

    echo "--- vLLM ---"
    source "$ENV_DIR/env_vllm/bin/activate"
    if [ -n "$QUANT" ]; then
        python scripts/bench_vllm.py --model "$HF_ID" --quantization "$QUANT" --output "$OUT_DIR/vllm_results.json"
    else
        python scripts/bench_vllm.py --model "$HF_ID" --output "$OUT_DIR/vllm_results.json"
    fi
    deactivate

    echo "--- SGLang (with + without RadixAttention) ---"
    source "$ENV_DIR/env_sglang/bin/activate"
    python scripts/bench_sglang.py --model "$HF_ID" --output "$OUT_DIR/sglang_results.json"
    deactivate

done <<< "$MODELS_TSV"

echo ""
echo "All models complete."
echo "Per-model results in results/<label>/"
