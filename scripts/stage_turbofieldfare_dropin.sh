#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
STAGING_ROOT="${CGC_TURBOFIELDFARE_STAGING_ROOT:-${REPO_ROOT}/var/external/turbofieldfare}"
BIN_TARGET="${STAGING_ROOT}/bin/TurboFieldfareServer"
MODEL_TARGET="${STAGING_ROOT}/models/gemma4.gturbo"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <TurboFieldfareServer> <gemma4.gturbo-dir>

Example:
  $(basename "$0") \\
    "/path/to/TurboFieldfareServer" \\
    "/path/to/gemma4.gturbo"

This stages the external artifacts into:
  ${BIN_TARGET}
  ${MODEL_TARGET}

The script uses symlinks so the original artifacts stay in place.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

SERVER_SRC="$(python3 - <<'PY' "$1"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
MODEL_SRC="$(python3 - <<'PY' "$2"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ! -x "${SERVER_SRC}" ]]; then
  echo "error: TurboFieldfareServer is not executable: ${SERVER_SRC}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_SRC}" ]]; then
  echo "error: gemma4.gturbo directory not found: ${MODEL_SRC}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_SRC}/manifest.json" ]]; then
  echo "error: gemma4.gturbo is missing manifest.json: ${MODEL_SRC}" >&2
  exit 1
fi

mkdir -p "$(dirname "${BIN_TARGET}")" "$(dirname "${MODEL_TARGET}")"
ln -sfn "${SERVER_SRC}" "${BIN_TARGET}"
ln -sfn "${MODEL_SRC}" "${MODEL_TARGET}"

echo "Staged TurboFieldfare artifacts:"
echo "  server -> ${BIN_TARGET}"
echo "  model  -> ${MODEL_TARGET}"
echo
echo "Next:"
echo "  source \"${REPO_ROOT}/scripts/setup_turbofieldfare_env.sh\""
echo "  \"${REPO_ROOT}/scripts/smoke_turbofieldfare_local_process.sh\""
