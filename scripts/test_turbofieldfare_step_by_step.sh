#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
SETUP_SCRIPT="${REPO_ROOT}/scripts/setup_turbofieldfare_env.sh"
STAGE_SCRIPT="${REPO_ROOT}/scripts/stage_turbofieldfare_dropin.sh"
SMOKE_SCRIPT="${REPO_ROOT}/scripts/smoke_turbofieldfare_local_process.sh"
TIMEOUT_MS="${CGC_TURBOFIELDFARE_SMOKE_TIMEOUT_MS:-15000}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0")
  $(basename "$0") <TurboFieldfareServer> <gemma4.gturbo-dir>

Modes:
  1) No arguments:
     Use whatever is already present at the fixed staging paths.
  2) Two arguments:
     First stage the provided server + model, then run the full test flow.

The flow is:
  Step 1. Stage artifacts if paths were provided
  Step 2. Resolve env from fixed staging
  Step 3. Check staged files and active processes
  Step 4. Run local_process smoke
EOF
}

step() {
  echo
  echo "== $1 =="
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 0 && $# -ne 2 ]]; then
  usage
  exit 2
fi

if [[ $# -eq 2 ]]; then
  step "Step 1: Stage External Artifacts"
  "${STAGE_SCRIPT}" "$1" "$2"
else
  step "Step 1: Use Existing Staging"
  echo "No source paths provided; using current fixed staging paths."
fi

step "Step 2: Resolve Environment"
ENV_OUTPUT="$(source "${SETUP_SCRIPT}")"
echo "${ENV_OUTPUT}"

SERVER_BIN="$(printf '%s\n' "${ENV_OUTPUT}" | awk -F= '/^CGC_TURBOFIELDFARE_SERVER_BIN=/{print $2}')"
MODEL_DIR="$(printf '%s\n' "${ENV_OUTPUT}" | awk -F= '/^CGC_TURBOFIELDFARE_MODEL=/{print $2}')"
READY_FLAG="$(printf '%s\n' "${ENV_OUTPUT}" | awk -F= '/^CGC_TURBOFIELDFARE_READY=/{print $2}')"

step "Step 3: Preflight Checks"
if [[ -n "${SERVER_BIN}" ]]; then
  ls -l "${SERVER_BIN}"
else
  echo "server: missing"
fi
if [[ -n "${MODEL_DIR}" ]]; then
  ls -ld "${MODEL_DIR}"
  if [[ -f "${MODEL_DIR}/manifest.json" ]]; then
    echo "manifest: ${MODEL_DIR}/manifest.json"
  else
    echo "manifest: missing"
  fi
else
  echo "model: missing"
fi

BUSY_PROCESSES="$(pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' || true)"
if [[ -n "${BUSY_PROCESSES}" ]]; then
  echo "active TurboFieldfare-related processes detected:"
  echo "${BUSY_PROCESSES}"
else
  echo "active processes: none"
fi

step "Step 4: local_process Smoke"
set +e
"${SMOKE_SCRIPT}" "${TIMEOUT_MS}"
SMOKE_RC=$?
set -e

echo
if [[ "${READY_FLAG}" == "1" && "${SMOKE_RC}" -eq 0 ]]; then
  echo "PASS: local_process is ready."
  exit 0
fi

echo "NOT READY: local_process is still blocked or failed."
echo "Hint:"
echo "  - server ready flag: ${READY_FLAG}"
echo "  - smoke exit code: ${SMOKE_RC}"
exit "${SMOKE_RC}"
