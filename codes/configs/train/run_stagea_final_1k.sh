#!/usr/bin/env bash
set -euo pipefail

# Final Stage A scheme selected for the thesis/system demo.
# This reproduces the 1K colorfix adapter that passed the structure/color guard.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export RUN_NAME="${RUN_NAME:-mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export COLOR_L1_WEIGHT="${COLOR_L1_WEIGHT:-0.4}"
export LUMA_LOSS_WEIGHT="${LUMA_LOSS_WEIGHT:-0.2}"
export COLOR_STAT_WEIGHT="${COLOR_STAT_WEIGHT:-0.1}"
export DETACH="${DETACH:-1}"
export AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

exec bash "${CODES_DIR}/configs/train/run_stagea_colorstyle8_compact_frombase.sh"
