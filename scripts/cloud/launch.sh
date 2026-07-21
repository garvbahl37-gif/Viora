#!/usr/bin/env bash
# One-command cloud training launch for Viora (pragmatic path: SigLIP + Qwen + LoRA).
#
# On a fresh GPU node (e.g. Lambda / RunPod 8xA100):
#     git clone <repo> && cd viora
#     bash scripts/cloud/launch.sh 8 Qwen/Qwen2.5-7B-Instruct
#
# Args: [NGPU=8] [LLM=Qwen/Qwen2.5-7B-Instruct]
# Env:  VIORA_SHARDS (shard pattern), CUDA_VERSION (cu124), HF_TOKEN (optional)
set -euo pipefail

NGPU="${1:-8}"
LLM="${2:-Qwen/Qwen2.5-7B-Instruct}"
SHARDS="${VIORA_SHARDS:-data/shards/train-{000000..000099}.tar}"
CUDA="${CUDA_VERSION:-cu124}"

echo ">> Viora cloud launch: ${NGPU} GPUs, LLM=${LLM}"
echo ">> shards: ${SHARDS}"

# 1) environment
if ! command -v uv >/dev/null 2>&1; then pip install -q uv; fi
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q torch --index-url "https://download.pytorch.org/whl/${CUDA}"
uv pip install -q -e ".[all]" peft webdataset uvicorn

# 2) sanity check (prints GPUs, bf16, etc.)
python scripts/validate_environment.py

# 3) build shards if none exist yet (synthetic fallback so the pipeline runs end-to-end)
if ! ls data/shards/*.tar >/dev/null 2>&1; then
  echo ">> no shards found — building a synthetic set to validate the pipeline"
  python scripts/build_shards.py --synthetic 2000 --out "data/shards/train-%06d.tar"
  SHARDS="data/shards/train-{000000..000001}.tar"
fi

# 4) launch training. FSDP only helps with >1 GPU; a single cheap card runs plain.
FSDP=$([ "${NGPU}" -gt 1 ] && echo true || echo false)
BATCH="${VIORA_BATCH:-$([ "${NGPU}" -gt 1 ] && echo 8 || echo 4)}"  # smaller default on one card
torchrun --nproc_per_node="${NGPU}" scripts/train.py \
  --model configs/model/viora_pragmatic.yaml \
  --train configs/training/pragmatic_lora.yaml \
  --shards "${SHARDS}" \
  llm.name_or_path="${LLM}" \
  training.fsdp="${FSDP}" \
  training.precision=bf16 \
  training.batch_size="${BATCH}" \
  training.gradient_checkpointing=true

echo ">> done. checkpoints in runs/pragmatic/  |  serve with:"
echo "   VIORA_MODEL_CONFIG=configs/model/viora_pragmatic.yaml VIORA_CHECKPOINT=runs/pragmatic/final.pt \\"
echo "   uvicorn viora.serving.api:app --host 0.0.0.0 --port 8000"
