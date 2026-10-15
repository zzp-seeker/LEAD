#!/usr/bin/env bash
set -euo pipefail
METHOD=${1:?usage: run_train.sh <grpo|sdpo|rlsd|lead> <biology|chemistry|material|physics> <qwen3-4b|qwen3-8b>}
DOMAIN=${2:?}
SCALE=${3:?}
shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="${FRAMEWORK:-$HERE/../framework}"
CONFIG_DIR="${CONFIG_DIR:-$HERE/../configs}"
CONFIG="${METHOD}_${DOMAIN}_${SCALE}"
: "${MODEL_PATH:?set MODEL_PATH to the backbone checkpoint directory}"
: "${DATA_DIR:?set DATA_DIR to the SupplementaryMaterial/data directory}"
: "${CKPT_DIR:=$PWD/ckpts/$CONFIG}"
[ -f "$CONFIG_DIR/$CONFIG.yaml" ] || { echo "no such config: $CONFIG_DIR/$CONFIG.yaml" >&2; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || { echo "model config missing: $MODEL_PATH/config.json" >&2; exit 1; }
[ -f "$DATA_DIR/curated/sciknoweval-$DOMAIN/train.parquet" ] || { echo "training data missing for $DOMAIN" >&2; exit 1; }
mkdir -p "$CKPT_DIR"
export MODEL_PATH DATA_DIR FRAMEWORK CKPT_DIR
export WORKDIR="${WORKDIR:-$CKPT_DIR}"
export PYTHONPATH="$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
python3 "$HERE/apply_method.py" --method "$METHOD" --framework "$FRAMEWORK"
mkdir -p "$CKPT_DIR/rollout_traces"
cd "$FRAMEWORK"
exec python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="['$DATA_DIR/curated/sciknoweval-$DOMAIN/train.parquet']" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$CKPT_DIR/rollout_traces" \
    custom_reward_function.path="$FRAMEWORK/verl/utils/reward_score/feedback/__init__.py" \
    trainer.logger='[console]' \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}" \
    +ray_kwargs.ray_init.include_dashboard=false \
    "$@"
