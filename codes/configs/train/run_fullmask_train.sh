#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_DIR="${CODES_DIR}/results/setup/4090"

sanitize_positive_int() {
  local value="${1:-}"
  local fallback="${2:-1}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
    echo "${fallback}"
  else
    echo "${value}"
  fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="$(sanitize_positive_int "${OMP_NUM_THREADS:-1}" 1)"
export MKL_NUM_THREADS="$(sanitize_positive_int "${MKL_NUM_THREADS:-1}" 1)"
export OPENBLAS_NUM_THREADS="$(sanitize_positive_int "${OPENBLAS_NUM_THREADS:-1}" 1)"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

MASTER_PORT="${MASTER_PORT:-29531}"
RUN_NAME="${RUN_NAME:-diabetic_merged_fullmask_4090_strict}"
DATA_CONFIG="${DATA_CONFIG:-${CODES_DIR}/configs/train/diabetic_merged_fullmask_clean_colorstyle8_compact.yaml}"
BASE_CKPT="${BASE_CKPT:-${CODES_DIR}/../checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${CODES_DIR}/results/train/${RUN_NAME}}"
MASK_SCALE="${MASK_SCALE:-2.0}"
STRUCT_MASK_CHANNELS="${STRUCT_MASK_CHANNELS:-6}"
FUNDUS_GEOMETRY_ALIGN="${FUNDUS_GEOMETRY_ALIGN:-1}"
FUNDUS_PADDING_RATIO="${FUNDUS_PADDING_RATIO:-0.01}"
MAX_STEPS="${MAX_STEPS:-10000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_CHECKPOINTING="${USE_CHECKPOINTING:-0}"
MASK_ADAPTER_LR="${MASK_ADAPTER_LR:-1e-4}"
CAP_EMBEDDER_LR="${CAP_EMBEDDER_LR:-}"
LR="${LR:-5e-6}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.1}"
TRAINABLE_STRATEGY="${TRAINABLE_STRATEGY:-struct_strict}"
BALANCE_FIELD="${BALANCE_FIELD:-}"
PARSER_CKPT="${PARSER_CKPT:-}"
RESUME="${RESUME:-}"
RESUME_ADAPTER_ONLY="${RESUME_ADAPTER_ONLY:-0}"
PARSER_BASE_CHANNELS="${PARSER_BASE_CHANNELS:-32}"
PARSER_LOSS_WEIGHT="${PARSER_LOSS_WEIGHT:-0.0}"
PARSER_DICE_WEIGHT="${PARSER_DICE_WEIGHT:-0.0}"
OPTIC_DISC_WEIGHT="${OPTIC_DISC_WEIGHT:-0.0}"
VESSEL_WEIGHT="${VESSEL_WEIGHT:-0.0}"
LESION_WEIGHT="${LESION_WEIGHT:-0.0}"
COLOR_L1_WEIGHT="${COLOR_L1_WEIGHT:-0.0}"
LUMA_LOSS_WEIGHT="${LUMA_LOSS_WEIGHT:-0.0}"
COLOR_STAT_WEIGHT="${COLOR_STAT_WEIGHT:-0.0}"
COLOR_CHROMA_WEIGHT="${COLOR_CHROMA_WEIGHT:-0.0}"
RB_GAP_WEIGHT="${RB_GAP_WEIGHT:-0.0}"
DETACH="${DETACH:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
TRAIN_SCREEN="${TRAIN_SCREEN:-train_fullmask4090}"
TRAIN_LOG="${TRAIN_LOG:-${SETUP_DIR}/train_fullmask_clean.log}"
SHUTDOWN_LOG="${SHUTDOWN_LOG:-${SETUP_DIR}/shutdown_watch_fullmask.log}"

TRAIN_ARGS=(
  --id "${RUN_NAME}"
  --data_path "${DATA_CONFIG}"
  --results_dir "${RESULTS_DIR}"
  --image_size 512
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --max_steps "${MAX_STEPS}"
  --ckpt_every "${CKPT_EVERY}"
  --log_every 20
  --precision bf16
  --grad_precision fp32
  --qk_norm
  --vae sdxl
  --mask_scale "${MASK_SCALE}"
  --struct_mask_channels "${STRUCT_MASK_CHANNELS}"
  --fundus_padding_ratio "${FUNDUS_PADDING_RATIO}"
  --caption_dropout_prob "${CAPTION_DROPOUT_PROB}"
  --lr "${LR}"
  --mask_adapter_lr "${MASK_ADAPTER_LR}"
  --num_workers "${NUM_WORKERS}"
  --data_parallel fsdp
  --global_seed 42
  --init_from "${BASE_CKPT}"
  --grad_clip 0.5
  --wd 0.00001
  --snr_type uniform
  --trainable_strategy "${TRAINABLE_STRATEGY}"
  --parser_base_channels "${PARSER_BASE_CHANNELS}"
  --parser_loss_weight "${PARSER_LOSS_WEIGHT}"
  --parser_dice_weight "${PARSER_DICE_WEIGHT}"
  --optic_disc_weight "${OPTIC_DISC_WEIGHT}"
  --vessel_weight "${VESSEL_WEIGHT}"
  --lesion_weight "${LESION_WEIGHT}"
  --color_l1_weight "${COLOR_L1_WEIGHT}"
  --luma_loss_weight "${LUMA_LOSS_WEIGHT}"
  --color_stat_weight "${COLOR_STAT_WEIGHT}"
  --color_chroma_weight "${COLOR_CHROMA_WEIGHT}"
  --rb_gap_weight "${RB_GAP_WEIGHT}"
)

if [ -n "${BALANCE_FIELD}" ]; then
  TRAIN_ARGS+=(--balance_field "${BALANCE_FIELD}")
fi

if [ -n "${CAP_EMBEDDER_LR}" ]; then
  TRAIN_ARGS+=(--cap_embedder_lr "${CAP_EMBEDDER_LR}")
fi

if [ "${FUNDUS_GEOMETRY_ALIGN}" = "1" ]; then
  TRAIN_ARGS+=(--fundus_geometry_align)
fi

if [ -n "${PARSER_CKPT}" ]; then
  TRAIN_ARGS+=(--parser_ckpt "${PARSER_CKPT}")
fi

if [ -n "${RESUME}" ]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

if [ "${RESUME_ADAPTER_ONLY}" = "1" ]; then
  TRAIN_ARGS+=(--resume_adapter_only)
fi

if [ "${USE_CHECKPOINTING}" = "1" ]; then
  TRAIN_ARGS+=(--checkpointing)
fi

mkdir -p "${RESULTS_DIR}" "${SETUP_DIR}"

run_train() {
  cd "${CODES_DIR}"
  torchrun --nproc-per-node=1 --master_port="${MASTER_PORT}" train.py "${TRAIN_ARGS[@]}"
}

if [ "${DETACH}" = "1" ]; then
  screen -S "${TRAIN_SCREEN}" -X quit >/dev/null 2>&1 || true

  screen -dmS "${TRAIN_SCREEN}" bash -lc "
    set -euo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    unset all_proxy ALL_PROXY no_proxy NO_PROXY
    set +u
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate Retina
    set -u
    cd '${CODES_DIR}'
    export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
    export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
    export MKL_NUM_THREADS='${MKL_NUM_THREADS}'
    export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}'
    export TORCH_COMPILE_DISABLE='${TORCH_COMPILE_DISABLE}'
    export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
    export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'
    export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}'
    export HF_HUB_DISABLE_TELEMETRY='${HF_HUB_DISABLE_TELEMETRY}'
    set -o pipefail
    torchrun --nproc-per-node=1 --master_port='${MASTER_PORT}' train.py ${TRAIN_ARGS[*]} 2>&1 | tee '${TRAIN_LOG}'
    status=\${PIPESTATUS[0]}
    if [ \"\${status}\" -eq 0 ] && [ '${AUTO_SHUTDOWN}' = '1' ]; then
      echo \"[\$(date -u '+%Y-%m-%dT%H:%M:%SZ')] training finished; syncing before shutdown\" | tee -a '${SHUTDOWN_LOG}'
      sync
      echo \"[\$(date -u '+%Y-%m-%dT%H:%M:%SZ')] issuing shutdown -h now\" | tee -a '${SHUTDOWN_LOG}'
      shutdown -h now 'training finished successfully' >>'${SHUTDOWN_LOG}' 2>&1 || poweroff >>'${SHUTDOWN_LOG}' 2>&1 || halt >>'${SHUTDOWN_LOG}' 2>&1
    fi
    exit \"\${status}\"
  "

  echo "[launch] train screen=${TRAIN_SCREEN}"
  echo "[launch] train log=${TRAIN_LOG}"
  if [ "${AUTO_SHUTDOWN}" = "1" ]; then
    echo "[launch] shutdown log=${SHUTDOWN_LOG}"
  fi
else
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate Retina
  set -u
  run_train
  if [ "${AUTO_SHUTDOWN}" = "1" ]; then
    sync
    shutdown -h now "training finished successfully" || poweroff || halt
  fi
fi
