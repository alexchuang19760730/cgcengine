#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
TIMEOUT_MS="${1:-15000}"
PYTHON_BIN="${REPO_ROOT}/.venv-cgc/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "error: python3 not found" >&2
  exit 1
fi

source "${REPO_ROOT}/scripts/setup_turbofieldfare_env.sh" >/dev/null

BUSY_PROCESSES="$(pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' || true)"
if [[ -n "${BUSY_PROCESSES}" ]]; then
  echo "error: detected existing TurboFieldfare-related process; smoke expects an idle host" >&2
  echo "${BUSY_PROCESSES}" >&2
  exit 3
fi

"${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${TIMEOUT_MS}"
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
timeout_ms = int(sys.argv[2])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.shared.colibri_backend import ColibriAdapterBackend
from app.shared.colibri_backend import build_unified_runtime_ir_v0

request_id = "tf-smoke"
backend = ColibriAdapterBackend(model_roots={
    "gturbo": str(os.environ.get("CGC_TURBOFIELDFARE_MODEL") or ""),
    "turbofieldfare": str(os.environ.get("CGC_TURBOFIELDFARE_MODEL") or ""),
    "gemma4_gturbo": str(os.environ.get("CGC_TURBOFIELDFARE_MODEL") or ""),
})
ir = build_unified_runtime_ir_v0(
    request_id=request_id,
    runtime_unit_plan={
        "enabled": True,
        "mode": "local_process",
        "reason": "turbofieldfare_local_process_smoke",
        "model": "gemma-4-26b-a4b-it",
        "family": "gemma4",
        "route_mode": "local_process",
        "frontier_key": request_id,
    },
    model_id="gemma-4-26b-a4b-it",
    model_family="gemma4",
    model_format="safetensors",
    runtime_mode="local_process",
    execution_intent="local_process_smoke",
    backend_family="mlx",
    runtime_backend="turbofieldfare",
    adapter_name="gemma4_a4b",
    platform="macos",
    device_class="apple_silicon",
    strategy_family="standard",
    speculative_mode="none",
    max_tokens=16,
)
snapshot = backend.begin_request(ir)
result = backend.submit_to_engine(
    transport="local_process",
    target="TurboFieldfareServer",
    timeout_ms=timeout_ms,
)

launch_contract = dict(snapshot.get("backend_launch_contract") or {})
submission = dict(result.get("submission") or {})
response_contract = dict(submission.get("response_contract") or {})
raw = dict(response_contract.get("raw") or {})
receipt = dict(raw.get("receipt") or {})

summary = {
    "ready": str(submission.get("delivery") or "") == "local_process_ready",
    "delivery": str(submission.get("delivery") or ""),
    "session_state": str(submission.get("session_state") or ""),
    "message": str(submission.get("message") or ""),
    "reason": str(launch_contract.get("reason") or ""),
    "missing_dependencies": list(launch_contract.get("missing_dependencies") or []),
    "server_bin": str(launch_contract.get("executable") or ""),
    "model_dir": str(launch_contract.get("model_dir") or ""),
    "base_url": str(launch_contract.get("base_url") or ""),
    "health_url": str(launch_contract.get("health_url") or ""),
    "receipt_path": str(launch_contract.get("receipt_path") or ""),
    "log_path": str(launch_contract.get("log_path") or ""),
    "request_contract_path": str(launch_contract.get("request_contract_path") or ""),
    "worker_id": str(submission.get("worker_id") or ""),
    "receipt_status": str(receipt.get("status") or ""),
    "receipt_message": str(receipt.get("message") or ""),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["ready"] else 4)
PY
