#!/usr/bin/env bash
set -euo pipefail

# Single-sample inference entrypoint for merged diabetic mask conditioning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

BASE_CKPT="${BASE_CKPT:-${CODES_DIR}/../checkpoints}"
ADAPTER_CKPT="${ADAPTER_CKPT:-${CODES_DIR}/../checkpoints/stageA_1k_final/adapter.pth}"
MASK_PATH="${MASK_PATH:-${CODES_DIR}/data/train/diabetic/mask/10000_left_fusion.png}"
PROMPT="${PROMPT:-COLOR_STYLE: warm orange fundus; a color fundus photograph with clear optic disc, macula, retinal vessels, and mild diabetic retinopathy signs.}"
OUT_DIR="${OUT_DIR:-${CODES_DIR}/results/infer/stageA_1k_final}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${CODES_DIR}/google_gemma-2b}"
LOCAL_VAE_PATH="${LOCAL_VAE_PATH:-${CODES_DIR}/sdxl-vae}"

cd "${CODES_DIR}"
mkdir -p "${OUT_DIR}"

if [ -z "${ADAPTER_CKPT}" ] || [ ! -e "${ADAPTER_CKPT}" ]; then
  echo "Adapter checkpoint not found. ADAPTER_CKPT=${ADAPTER_CKPT}" >&2
  exit 1
fi

python inference_mask.py \
  --model NextDiT_2B_GQA_patch2 \
  --base_ckpt "${BASE_CKPT}" \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  --prompt "${PROMPT}" \
  --mask_path "${MASK_PATH}" \
  --out_dir "${OUT_DIR}" \
  --image_size 512 \
  --num_sampling_steps 80 \
  --sampling_method euler \
  --cfg_scale 2.0 \
  --seed 42 \
  --precision bf16 \
  --qk_norm \
  --mask_scale 1.0 \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --local_diffusers_model_root "${LOCAL_VAE_PATH}"
