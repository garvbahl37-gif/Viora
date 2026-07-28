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
# Frames sampled per clip (training used 8). Long uploads need more:
#   VIORA_NUM_FRAMES=32 ./scripts/serve.sh <ckpt>
export VIORA_NUM_FRAMES="${VIORA_NUM_FRAMES:-}"

# Resolve an interpreter: the project venv first (it has viora + uvicorn installed),
# then an active venv, then python3. Bare `python` often does not exist on macOS.
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "no python interpreter found (tried .venv/bin/python, \$VIRTUAL_ENV, python3, python)" >&2
  exit 1
fi

if ! "$PY" -c "import uvicorn" 2>/dev/null; then
  echo "uvicorn is not installed for $PY -- run:  $PY -m pip install -e '.[serve]'" >&2
  exit 1
fi

echo "open http://127.0.0.1:${PORT}   (interpreter: $PY)"
exec "$PY" -m uvicorn viora.serving.api:app --host 127.0.0.1 --port "$PORT"
