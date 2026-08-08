#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
TF_REPO="${CGC_TURBOFIELDFARE_REPO:-/Users/alexchuang/Documents/turbo-fieldfare}"
TF_STAGING_ROOT="${CGC_TURBOFIELDFARE_STAGING_ROOT:-${REPO_ROOT}/var/external/turbofieldfare}"
TF_STAGE_BIN="${TF_STAGING_ROOT}/bin/TurboFieldfareServer"
TF_SWIFT_BREW="/opt/homebrew/opt/swift/bin/swift"
TF_MANIFEST="${TF_REPO}/Package.swift"

usage() {
  cat <<EOF
Usage:
  $(basename "$0")

Build TurboFieldfareServer with the best available Swift toolchain and stage it to:
  ${TF_STAGE_BIN}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${TF_REPO}" ]]; then
  echo "error: missing TurboFieldfare repo: ${TF_REPO}" >&2
  exit 1
fi

if [[ ! -f "${TF_MANIFEST}" ]]; then
  echo "error: missing Package.swift: ${TF_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${TF_STAGING_ROOT}/bin"

if [[ -x "${TF_SWIFT_BREW}" ]]; then
  SWIFT_BIN="${TF_SWIFT_BREW}"
  export PATH="/opt/homebrew/opt/swift/bin:${PATH}"
else
  SWIFT_BIN="$(command -v swift)"
fi

SDKROOT_VALUE="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
if [[ -n "${SDKROOT_VALUE}" ]]; then
  export SDKROOT="${SDKROOT_VALUE}"
fi

export SWIFT_EXEC="${SWIFT_BIN}"
export SWIFTPM_ENABLE_PLUGINS=0
export SWIFTCI_USE_LOCAL_DEPS=1

"${SWIFT_BIN}" --version

perl -0pi -e 's#// swift-tools-version:\s*6\.2#// swift-tools-version: 6.1#g' "${TF_MANIFEST}"
perl -0pi -e 's#\.macOS\(\.v26\)#.macOS("15.0")#g; s#\.iOS\(\.v26\)#.iOS("18.0")#g' "${TF_MANIFEST}"

(
  cd "${TF_REPO}"
  "${SWIFT_BIN}" build -c release --product TurboFieldfareServer --disable-sandbox
)

BUILD_BIN="${TF_REPO}/.build/release/TurboFieldfareServer"
if [[ ! -x "${BUILD_BIN}" ]]; then
  echo "error: build finished but binary is missing: ${BUILD_BIN}" >&2
  exit 1
fi

cp -f "${BUILD_BIN}" "${TF_STAGE_BIN}"
chmod +x "${TF_STAGE_BIN}"

echo "staged_server=${TF_STAGE_BIN}"
