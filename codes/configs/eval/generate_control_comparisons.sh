#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${RUN_DIR:?set RUN_DIR to the finished train run directory}"
STEP_DIR="${STEP_DIR:-$(find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"
BASE_CKPT="${BASE_CKPT:-${CODES_DIR}/../checkpoints}"
METADATA="${METADATA:-${CODES_DIR}/data/merged_lesion_only/diabetic/autodl/metadata_example_20_nonempty.jsonl}"
ROOT_DIR="${ROOT_DIR:-${CODES_DIR}/data}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/eval/controls}"
SEED="${SEED:-42}"
CFG_SCALE="${CFG_SCALE:-2.0}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-80}"
VAE_NAME="${VAE_NAME:-sdxl}"
FUNDUS_COLOR_REFERENCE="${FUNDUS_COLOR_REFERENCE:-}"
FUNDUS_COLOR_GAMMA="${FUNDUS_COLOR_GAMMA:-1.0}"
FUNDUS_WARM_R="${FUNDUS_WARM_R:-1.0}"
FUNDUS_WARM_G="${FUNDUS_WARM_G:-1.0}"
FUNDUS_WARM_B="${FUNDUS_WARM_B:-1.0}"
FUNDUS_SATURATION="${FUNDUS_SATURATION:-1.0}"
FUNDUS_CONTRAST="${FUNDUS_CONTRAST:-1.0}"
FUNDUS_SHADOW_LIFT="${FUNDUS_SHADOW_LIFT:-0.0}"
FUNDUS_EDGE_LIFT="${FUNDUS_EDGE_LIFT:-0.0}"
FUNDUS_FLATTEN_STRENGTH="${FUNDUS_FLATTEN_STRENGTH:-0.0}"
FUNDUS_PROFILE_JSON="${FUNDUS_PROFILE_JSON:-}"
FUNDUS_SHADE_CORRECTION_STRENGTH="${FUNDUS_SHADE_CORRECTION_STRENGTH:-0.0}"
FUNDUS_SHADE_BLUR_RADIUS="${FUNDUS_SHADE_BLUR_RADIUS:-41.0}"

# Fixed ids chosen from the example_20 split we already validated.
LESION_ID="${LESION_ID:-12544_left}"
ALT_MASK_ID="${ALT_MASK_ID:-13168_left}"
HEALTHY_PROMPT_ID="${HEALTHY_PROMPT_ID:-13447_left}"

if [ ! -d "${STEP_DIR}" ]; then
  echo "checkpoint step dir not found: ${STEP_DIR}" >&2
  exit 1
fi

ADAPTER_CKPT="${STEP_DIR}"
mkdir -p "${OUT_DIR}"

lookup_field() {
  local record_id="$1"
  local field="$2"
  python - "$METADATA" "$record_id" "$field" <<'PY'
import json, sys
meta, record_id, field = sys.argv[1:]
with open(meta, "r", encoding="utf-8-sig") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record["id"]) == record_id:
            print(record[field])
            sys.exit(0)
raise SystemExit(f"record_id not found: {record_id}")
PY
}

resolve_path() {
  python - "$ROOT_DIR" "$1" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
value = Path(sys.argv[2])
print(str(value if value.is_absolute() else (root / value).resolve()))
PY
}

LESION_PROMPT="$(lookup_field "${LESION_ID}" caption)"
HEALTHY_PROMPT="$(lookup_field "${HEALTHY_PROMPT_ID}" caption)"
LESION_MASK="$(resolve_path "$(lookup_field "${LESION_ID}" mask)")"
ALT_MASK="$(resolve_path "$(lookup_field "${ALT_MASK_ID}" mask)")"

cd "${CODES_DIR}"

common_args=(
  --model NextDiT_2B_GQA_patch2
  --base_ckpt "${BASE_CKPT}"
  --adapter_ckpt "${ADAPTER_CKPT}"
  --out_dir "${OUT_DIR}"
  --image_size 512
  --num_sampling_steps "${NUM_SAMPLING_STEPS}"
  --sampling_method euler
  --cfg_scale "${CFG_SCALE}"
  --seed "${SEED}"
  --precision bf16
  --vae "${VAE_NAME}"
  --qk_norm
  --mask_scale 1.0
)

if [ -n "${FUNDUS_COLOR_REFERENCE}" ]; then
  common_args+=(
    --fundus_color_reference "${FUNDUS_COLOR_REFERENCE}"
    --fundus_color_gamma "${FUNDUS_COLOR_GAMMA}"
    --fundus_warm_r "${FUNDUS_WARM_R}"
    --fundus_warm_g "${FUNDUS_WARM_G}"
    --fundus_warm_b "${FUNDUS_WARM_B}"
    --fundus_saturation "${FUNDUS_SATURATION}"
    --fundus_contrast "${FUNDUS_CONTRAST}"
    --fundus_shadow_lift "${FUNDUS_SHADOW_LIFT}"
    --fundus_edge_lift "${FUNDUS_EDGE_LIFT}"
    --fundus_flatten_strength "${FUNDUS_FLATTEN_STRENGTH}"
    --fundus_shade_correction_strength "${FUNDUS_SHADE_CORRECTION_STRENGTH}"
    --fundus_shade_blur_radius "${FUNDUS_SHADE_BLUR_RADIUS}"
  )
  if [ -n "${FUNDUS_PROFILE_JSON}" ]; then
    common_args+=(--fundus_profile_json "${FUNDUS_PROFILE_JSON}")
  fi
fi

# 1. zero mask
python inference_mask.py \
  "${common_args[@]}" \
  --prompt "${LESION_PROMPT}" \
  --mask_path "${LESION_MASK}" \
  --disable_mask_condition \
  --output_name "control_zero_mask"

# 2. same prompt + different masks
python inference_mask.py \
  "${common_args[@]}" \
  --prompt "${LESION_PROMPT}" \
  --mask_path "${LESION_MASK}" \
  --output_name "control_same_prompt_mask_a"

python inference_mask.py \
  "${common_args[@]}" \
  --prompt "${LESION_PROMPT}" \
  --mask_path "${ALT_MASK}" \
  --output_name "control_same_prompt_mask_b"

# 3. same mask + different prompts
python inference_mask.py \
  "${common_args[@]}" \
  --prompt "${LESION_PROMPT}" \
  --mask_path "${LESION_MASK}" \
  --output_name "control_same_mask_prompt_lesion"

python inference_mask.py \
  "${common_args[@]}" \
  --prompt "${HEALTHY_PROMPT}" \
  --mask_path "${LESION_MASK}" \
  --output_name "control_same_mask_prompt_healthy"

echo "[ok] control comparisons saved to ${OUT_DIR}"
