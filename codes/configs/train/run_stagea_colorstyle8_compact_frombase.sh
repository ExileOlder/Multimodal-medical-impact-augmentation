#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_DIR="${CODES_DIR}/results/setup/4090"

RUN_NAME="${RUN_NAME:-mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix}"
RESULTS_DIR="${RESULTS_DIR:-${CODES_DIR}/results/train/${RUN_NAME}}"
TRAIN_SCREEN="${TRAIN_SCREEN:-train_stageA_1k_colorfix_frombase}"
TRAIN_LOG="${TRAIN_LOG:-${SETUP_DIR}/${RUN_NAME}.log}"
MASTER_PORT="${MASTER_PORT:-29582}"

PARSER_CKPT="${PARSER_CKPT:-${CODES_DIR}/results/train/fusion_parser_maskqc_clean_canonical_5k_bs32/2026-04-22_11-27-34_fusion_parser/parser_best.pt}"
DATA_CONFIG="${DATA_CONFIG:-${CODES_DIR}/configs/train/diabetic_merged_fullmask_clean_colorstyle8_compact.yaml}"

export RUN_NAME
export RESULTS_DIR
export TRAIN_SCREEN
export TRAIN_LOG
export MASTER_PORT
export PARSER_CKPT
export DATA_CONFIG
export MAX_STEPS="${MAX_STEPS:-1000}"
export AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
export DETACH="${DETACH:-1}"
export FUNDUS_GEOMETRY_ALIGN="${FUNDUS_GEOMETRY_ALIGN:-1}"
export FUNDUS_PADDING_RATIO="${FUNDUS_PADDING_RATIO:-0.01}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export USE_CHECKPOINTING="${USE_CHECKPOINTING:-1}"
export LR="${LR:-5e-6}"
export MASK_ADAPTER_LR="${MASK_ADAPTER_LR:-1e-4}"
export CAP_EMBEDDER_LR="${CAP_EMBEDDER_LR:-2e-5}"
export CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
export COLOR_L1_WEIGHT="${COLOR_L1_WEIGHT:-0.4}"
export LUMA_LOSS_WEIGHT="${LUMA_LOSS_WEIGHT:-0.2}"
export COLOR_STAT_WEIGHT="${COLOR_STAT_WEIGHT:-0.1}"

exec bash "${CODES_DIR}/configs/train/run_fullmask_train_strict_alignment.sh"
