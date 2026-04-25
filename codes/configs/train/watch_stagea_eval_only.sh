#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_DIR="${CODES_DIR}/results/setup/4090"

STAGEA_RESULTS_DIR="${STAGEA_RESULTS_DIR:?set STAGEA_RESULTS_DIR to the stage A results root}"
STAGEA_LOG="${STAGEA_LOG:?set STAGEA_LOG to the train log path}"
STAGEA_SCREEN="${STAGEA_SCREEN:-}"
STAGEA_RUN_NAME="${STAGEA_RUN_NAME:-$(basename "${STAGEA_RESULTS_DIR}")}"
STAGEA_RUN_DIR="${STAGEA_RUN_DIR:-}"
STAGEA_POLL_SECONDS="${STAGEA_POLL_SECONDS:-30}"
STAGEA_MAX_MISSING_POLLS="${STAGEA_MAX_MISSING_POLLS:-10}"

WATCH_LOG="${WATCH_LOG:-${SETUP_DIR}/watch_${STAGEA_RUN_NAME}.log}"
BASE_CKPT="${BASE_CKPT:-${CODES_DIR}/../checkpoints}"
EVAL_METADATA="${EVAL_METADATA:-${CODES_DIR}/data/merged/diabetic/autodl/metadata_example_20_maskqc_clean_colorstyle8_compact.jsonl}"
EVAL_ROOT_DIR="${EVAL_ROOT_DIR:-${CODES_DIR}/data}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
EVAL_TOPK="${EVAL_TOPK:-20}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-canonical_square}"
FUNDUS_PADDING_RATIO="${FUNDUS_PADDING_RATIO:-0.01}"
TRIPTYCH_MAX_ITEMS="${TRIPTYCH_MAX_ITEMS:-10}"
GENERATED_LABEL="${GENERATED_LABEL:-Generated}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "${SETUP_DIR}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${WATCH_LOG}"
}

latest_subdir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

latest_checkpoint_dir() {
  local run_dir="$1"
  find "${run_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' | \
    awk '$1 ~ /^[0-9]+$/ {print $2}' | sort | tail -n 1
}

screen_exists() {
  local screen_name="$1"
  if [ -z "${screen_name}" ]; then
    return 1
  fi
  screen -ls 2>/dev/null | grep -q "[.]${screen_name}[[:space:]]"
}

extract_stagea_run_dir_from_log() {
  if [ ! -f "${STAGEA_LOG}" ]; then
    return 0
  fi
  grep "Experiment directory:" "${STAGEA_LOG}" | tail -n 1 | sed 's/^.*Experiment directory: //'
}

resolve_stagea_run_dir() {
  local attempts=0
  while true; do
    local run_dir=""
    if [ -n "${STAGEA_RUN_DIR}" ] && [ -f "${STAGEA_RUN_DIR}/run_manifest.json" ]; then
      echo "${STAGEA_RUN_DIR}"
      return 0
    fi
    run_dir="$(extract_stagea_run_dir_from_log)"
    if [ -n "${run_dir}" ] && [ -f "${run_dir}/run_manifest.json" ]; then
      echo "${run_dir}"
      return 0
    fi
    run_dir="$(latest_subdir "${STAGEA_RESULTS_DIR}")"
    if [ -n "${run_dir}" ] && [ -f "${run_dir}/run_manifest.json" ]; then
      echo "${run_dir}"
      return 0
    fi
    attempts=$((attempts + 1))
    if [ "${attempts}" -ge 20 ]; then
      log "stage A run directory did not appear in time"
      return 1
    fi
    sleep 5
  done
}

wait_for_stagea_done() {
  local missing_polls=0
  log "waiting for stage A completion: log=${STAGEA_LOG}"
  while true; do
    if [ -f "${STAGEA_LOG}" ] && grep -q "Done!" "${STAGEA_LOG}"; then
      log "stage A finished successfully"
      return 0
    fi

    if pgrep -f "train.py.*${STAGEA_RESULTS_DIR}" >/dev/null 2>&1 || screen_exists "${STAGEA_SCREEN}"; then
      missing_polls=0
    else
      missing_polls=$((missing_polls + 1))
      log "stage A process not found (miss ${missing_polls}/${STAGEA_MAX_MISSING_POLLS}); waiting"
      if [ "${missing_polls}" -ge "${STAGEA_MAX_MISSING_POLLS}" ]; then
        log "stage A missing for too long without success marker; aborting"
        return 1
      fi
    fi

    sleep "${STAGEA_POLL_SECONDS}"
  done
}

run_eval_and_triptych() {
  local run_dir="$1"
  local marker="${run_dir}/checkpoints/eval/.posttrain_eval_only_done"
  local step_dir
  local step_name
  local eval_dir
  local generated_dir
  local output_sheet

  if [ -f "${marker}" ]; then
    log "post-train eval already finished earlier; skip"
    return 0
  fi

  step_dir="$(latest_checkpoint_dir "${run_dir}")"
  if [ -z "${step_dir}" ]; then
    log "no numeric checkpoints found under ${run_dir}"
    return 1
  fi

  step_name="$(basename "${step_dir}")"
  eval_dir="${run_dir}/checkpoints/eval/${step_name}"
  generated_dir="${eval_dir}/generated"
  output_sheet="${eval_dir}/${STAGEA_RUN_NAME}_triptych_sheet_canonical.png"

  log "running structural eval for ${run_dir}"
  (
    cd "${CODES_DIR}"
    python ./run_structural_eval.py \
      --run_dir "${run_dir}" \
      --base_ckpt "${BASE_CKPT}" \
      --metadata "${EVAL_METADATA}" \
      --root_dir "${EVAL_ROOT_DIR}" \
      --limit "${EVAL_LIMIT}" \
      --topk "${EVAL_TOPK}" \
      --alignment_mode "${ALIGNMENT_MODE}" \
      --fundus_padding_ratio "${FUNDUS_PADDING_RATIO}"
  ) >>"${WATCH_LOG}" 2>&1

  log "rendering triptych sheet for ${generated_dir}"
  (
    cd "${CODES_DIR}"
    python ./render_generation_comparisons.py \
      --metadata "${EVAL_METADATA}" \
      --root_dir "${EVAL_ROOT_DIR}" \
      --generated_dir "${generated_dir}" \
      --output_path "${output_sheet}" \
      --max_items "${TRIPTYCH_MAX_ITEMS}" \
      --generated_label "${GENERATED_LABEL}" \
      --align_mode "${ALIGNMENT_MODE}" \
      --fundus_padding_ratio "${FUNDUS_PADDING_RATIO}"
  ) >>"${WATCH_LOG}" 2>&1

  touch "${marker}"
  log "eval artifacts ready: ${eval_dir}"
}

maybe_shutdown() {
  if [ "${AUTO_SHUTDOWN}" != "1" ]; then
    log "AUTO_SHUTDOWN=0; skip shutdown"
    return 0
  fi
  sync
  log "issuing shutdown -h now after stage A eval"
  shutdown -h now "stage A training and eval finished successfully" >>"${WATCH_LOG}" 2>&1 || \
    poweroff >>"${WATCH_LOG}" 2>&1 || \
    halt >>"${WATCH_LOG}" 2>&1
}

main() {
  log "eval-only watcher start"
  log "stagea_results_dir=${STAGEA_RESULTS_DIR}"

  wait_for_stagea_done

  local run_dir
  run_dir="$(resolve_stagea_run_dir)"
  log "stage A run dir: ${run_dir}"

  run_eval_and_triptych "${run_dir}"
  maybe_shutdown
  log "watch completed"
}

main "$@"
