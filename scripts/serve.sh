#!/usr/bin/env bash
# Serve Viora Studio with your TRAINED model.
#
#   ./scripts/serve.sh /path/to/final.pt          # or viora-trained.pt
#   ./scripts/serve.sh /path/to/final.pt 8000 cpu
#
# Then open http://127.0.0.1:8000 and drop a video in. Without a checkpoint
# argument this serves the UNTRAINED tiny model (evidence-only answers, no captions).
set -euo pipefail

CKPT="${1:-}"
PORT="${2:-8000}"
DEVICE="${3:-}"

cd "$(dirname "$0")/.."

if [[ -n "$CKPT" ]]; then
  if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
  fi
  export VIORA_CHECKPOINT="$CKPT"
  # The trained model is the pragmatic recipe (SigLIP + Qwen-0.5B + LoRA); the
  # config already pins llm.name_or_path, so the frozen backbones load from HF.
  export VIORA_MODEL_CONFIG="${VIORA_MODEL_CONFIG:-configs/model/viora_pragmatic.yaml}"
  echo "serving TRAINED model: $CKPT"
  echo "  config: $VIORA_MODEL_CONFIG"
else
  echo "no checkpoint given -> serving the UNTRAINED tiny model (no captions)."
  echo "  usage: $0 /path/to/final.pt"
fi

[[ -n "$DEVICE" ]] && export VIORA_DEVICE="$DEVICE"

echo "open http://127.0.0.1:${PORT}"
exec python -m uvicorn viora.serving.api:app --host 127.0.0.1 --port "$PORT"
