#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/docs/technical_whitepapers/examples/host2_deepseek_inst1_inst2_minicpm5_fusionroute_runtime.example.json}"

cd "${REPO_ROOT}"
python3 cgc_engine/tools/scripts/demo/generate_system_execution_manifest_example.py --config "${CONFIG_PATH}"
