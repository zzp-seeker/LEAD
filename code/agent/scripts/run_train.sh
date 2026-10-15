#!/usr/bin/env bash
set -euo pipefail
METHOD=${1:?usage: run_train.sh <grpo|sdpo|rlsd|lead> <alfworld|webshop|search> <qwen3-4b|qwen3-8b>}
SCENARIO=${2:?}
SCALE=${3:?}
shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="${AGENT_FRAMEWORK:-$HERE/../framework}"
CONFIG_DIR="${CONFIG_DIR:-$HERE/../configs}"
CONFIG="${METHOD}_${SCENARIO}_${SCALE}"
: "${MODEL_PATH:?set MODEL_PATH to the backbone checkpoint directory}"
: "${AGENT_DATA_DIR:?set AGENT_DATA_DIR to the environment task data directory}"
: "${CKPT_DIR:=$PWD/ckpts/$CONFIG}"
[ -f "$CONFIG_DIR/$CONFIG.yaml" ] || { echo "no such config: $CONFIG_DIR/$CONFIG.yaml" >&2; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || { echo "model config missing: $MODEL_PATH/config.json" >&2; exit 1; }
[ -f "$FRAMEWORK/verl/trainer/main_a7.py" ] || { echo "agent framework missing: $FRAMEWORK" >&2; exit 1; }
mkdir -p "$CKPT_DIR"
export MODEL_PATH AGENT_DATA_DIR CKPT_DIR
export PYTHONPATH="$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"
case "$METHOD" in
  lead) ENTRY=verl.trainer.main_a7 ;;
  rlsd) ENTRY=verl.trainer.main_rlsd ;;
  *)    ENTRY=verl.trainer.main_sdar ;;
esac
mkdir -p "$CKPT_DIR"
cd "$FRAMEWORK"
exec python3 -m "$ENTRY" \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
