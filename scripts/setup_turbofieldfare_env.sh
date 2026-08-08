#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
TF_REPO="${CGC_TURBOFIELDFARE_REPO:-/Users/alexchuang/Documents/turbo-fieldfare}"
TF_STAGING_ROOT="${CGC_TURBOFIELDFARE_STAGING_ROOT:-${REPO_ROOT}/var/external/turbofieldfare}"
TF_STAGE_BIN="${TF_STAGING_ROOT}/bin/TurboFieldfareServer"
TF_STAGE_MODEL="${TF_STAGING_ROOT}/models/gemma4.gturbo"
TF_BIN_DEFAULT="${TF_REPO}/.build/release/TurboFieldfareServer"
TF_RESTORED_MODEL_DEFAULT="${CGC_TURBOFIELDFARE_RESTORED_MODEL:-/tmp/gemma4-restored-local.gturbo}"
TF_MODEL_DEFAULT="${TF_REPO}/scratch/gemma4.gturbo"
TF_LOCAL_MODEL_DEFAULT="${REPO_ROOT}/scratch/gemma4.gturbo"

mkdir -p "${TF_STAGING_ROOT}/bin" "${TF_STAGING_ROOT}/models"

pick_first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

pick_first_directory() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

is_complete_gturbo_directory() {
  local candidate="$1"
  [[ -n "${candidate}" && -d "${candidate}" ]] || return 1
  [[ -f "${candidate}/manifest.json" ]] || return 1
  [[ -f "${candidate}/verified-install.json" ]] || return 1
  [[ -f "${candidate}/model_weights.bin" ]] || return 1
  [[ -f "${candidate}/packed_experts/layout.json" ]] || return 1
  [[ -f "${candidate}/packed_experts/layer_29.bin" ]] || return 1
}

pick_first_complete_gturbo_directory() {
  local candidate
  for candidate in "$@"; do
    if is_complete_gturbo_directory "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

export CGC_TURBOFIELDFARE_REPO="${TF_REPO}"
export CGC_TURBOFIELDFARE_STAGING_ROOT="${TF_STAGING_ROOT}"

TF_SERVER_BIN="$(
  pick_first_executable \
    "${CGC_TURBOFIELDFARE_SERVER_BIN:-}" \
    "${CGC_TURBOFIELDFARE_BIN:-}" \
    "${CGC_TURBOFIELDFARE_PATH:-}" \
    "${TF_STAGE_BIN}" \
    "${TF_BIN_DEFAULT}" \
    "$(command -v TurboFieldfareServer 2>/dev/null || true)" \
    "$(command -v turbofieldfare 2>/dev/null || true)" \
  || true
)"

TF_MODEL_DIR="$(
  pick_first_complete_gturbo_directory \
    "${CGC_TURBOFIELDFARE_MODEL:-}" \
    "${TURBOFIELDFARE_MODEL:-}" \
    "${CGC_TURBOFIELDFARE_MODEL_DIR:-}" \
    "${TF_RESTORED_MODEL_DEFAULT}" \
    "${TF_STAGE_MODEL}" \
    "${TF_MODEL_DEFAULT}" \
    "${TF_LOCAL_MODEL_DEFAULT}" \
  || true
)"

if [[ -n "${TF_SERVER_BIN}" ]]; then
  export CGC_TURBOFIELDFARE_SERVER_BIN="${TF_SERVER_BIN}"
  export CGC_TURBOFIELDFARE_BIN="${TF_SERVER_BIN}"
  export CGC_TURBOFIELDFARE_PATH="${TF_SERVER_BIN}"
fi

if [[ -n "${TF_MODEL_DIR}" ]]; then
  export CGC_TURBOFIELDFARE_MODEL="${TF_MODEL_DIR}"
  export CGC_TURBOFIELDFARE_MODEL_DIR="${TF_MODEL_DIR}"
fi

echo "CGC_TURBOFIELDFARE_REPO=${CGC_TURBOFIELDFARE_REPO}"
echo "CGC_TURBOFIELDFARE_STAGING_ROOT=${CGC_TURBOFIELDFARE_STAGING_ROOT}"
echo "CGC_TURBOFIELDFARE_SERVER_BIN=${CGC_TURBOFIELDFARE_SERVER_BIN:-}"
echo "CGC_TURBOFIELDFARE_MODEL=${CGC_TURBOFIELDFARE_MODEL:-}"
if [[ -n "${CGC_TURBOFIELDFARE_SERVER_BIN:-}" && -n "${CGC_TURBOFIELDFARE_MODEL:-}" ]]; then
  echo "CGC_TURBOFIELDFARE_READY=1"
else
  echo "CGC_TURBOFIELDFARE_READY=0"
fi
echo "Drop-in binary target: ${TF_STAGE_BIN}"
echo "Drop-in model target: ${TF_STAGE_MODEL}"
echo "Use with: source \"${REPO_ROOT}/scripts/setup_turbofieldfare_env.sh\""
