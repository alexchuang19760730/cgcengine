#!/usr/bin/env bash
# 把 prime-agent 的 release 鏡像抓到本機快取 —— 給容器**離線**安裝用。
#
# 為什麼需要它（2026-09-17 實測，數字都是當場量的）：
#   同一個 URL，host 是 ~12.5 MB/s（59,607,437 B / 4.55s），容器內只有 ~164 KB/s
#   （49,090,240 B / 300s）—— 差 80 倍，瓶頸是 colima 的 NAT，不是上游。
#   而官方安裝器把 `--max-time 300` **寫死**（`prime_agent_curl_download … --max-time 300`，
#   install.sh 沒有可調該值的環境變數）⇒ 59.6 MB 的 release **永遠抓不完**。
#   後果：整段安裝 421.89s 後逾時，`INSTALL_FAIL_STATUS`，prime-agent 從未被啟動，
#   於是 tb 回報 `agent_timeout` ＋ `total_input_tokens = 0` —— 那個 Unresolved
#   **完全不是**關於模型／agent 路徑的證據。
#
# 做法：在 host 上抓一次（十幾秒），存成與上游相同的目錄佈局；adapter 再把產出的
#   tarball 用 `session.copy_to_container` 送進容器（走 Docker API，不經容器網路）；
#   容器內由 `prime-agent-setup.sh` 起一個 **loopback** http server，讓**官方安裝器**
#   從 127.0.0.1 取檔 —— 它明文支援這種 feed，不是 hack：
#     install.sh 的 prime_agent_validate_download_base_url：
#       `http://*` 只有在 PRIME_AGENT_ALLOW_INSECURE_HTTP_FOR_TESTS=1 **且**
#       prime_agent_is_loopback_test_base_url（只接受 http://127.0.0.1:<數字埠>）時放行。
#   實測同一顆容器：安裝 421.89s → **4s**（rc=0，`prime-agent --version` = 0.9.5）。
#
# 佈局（照上游 R2 桶；★ 目錄帶 `v`、檔名**不帶** —— 寫錯會拿到 404，我踩過）：
#   <mirror>/stable                                           內容如 `v0.9.5`
#   <mirror>/releases/v<ver>/SHA256SUMS
#   <mirror>/releases/v<ver>/prime-agent-<ver>-linux-arm64.tar.gz
#   <mirror>/releases/v<ver>/prime-agent-<ver>-linux-arm64-musl.tar.gz
#
# 用法:
#   bash agent_harness/tb_loop/scripts/fetch-prime-agent-mirror.sh              # 缺才抓
#   bash agent_harness/tb_loop/scripts/fetch-prime-agent-mirror.sh --refresh    # 重抓
#
# 可覆寫: TB_PA_BASE（上游 base）、TB_PA_MIRROR（快取位置）、TB_PA_PLATFORMS（平台後綴清單）
set -u

BASE="${TB_PA_BASE:-https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev}"
ROOT="${TB_PA_MIRROR:-${HOME}/.cache/prime-agent-mirror}"
TARBALL="${ROOT}.tar.gz"
PLATFORMS="${TB_PA_PLATFORMS:-linux-arm64 linux-arm64-musl}"
REFRESH=0
if [ "${1:-}" = "--refresh" ]; then REFRESH=1; fi

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    else sha256sum "$1" | awk '{print $1}'; fi
}

echo "[mirror] base   = ${BASE}"
echo "[mirror] 快取   = ${ROOT}"
echo "[mirror] tarball= ${TARBALL}"

# ---- 1) channel 檔 --------------------------------------------------------
mkdir -p "${ROOT}/releases"
if [ "${REFRESH}" = 1 ] || [ ! -s "${ROOT}/stable" ]; then
    echo "[mirror] 抓 channel: ${BASE}/stable"
    curl -fsSL --max-time 30 "${BASE}/stable" -o "${ROOT}/stable" || {
        echo "[mirror] 失敗: 抓不到 channel 檔" >&2; exit 1; }
fi
CHANNEL="$(tr -d '\r\n' < "${ROOT}/stable")"
VER="${CHANNEL#v}"
if [ -z "${VER}" ]; then
    echo "[mirror] 失敗: channel 檔是空的" >&2; exit 1
fi
DIR="${ROOT}/releases/v${VER}"
mkdir -p "${DIR}"
echo "[mirror] channel = ${CHANNEL}  ⇒ 目錄 releases/v${VER}，檔名不含 v"

# ---- 2) SHA256SUMS --------------------------------------------------------
if [ "${REFRESH}" = 1 ] || [ ! -s "${DIR}/SHA256SUMS" ]; then
    echo "[mirror] 抓 checksums"
    curl -fsSL --max-time 30 "${BASE}/releases/v${VER}/SHA256SUMS" -o "${DIR}/SHA256SUMS" || {
        echo "[mirror] 失敗: 抓不到 SHA256SUMS" >&2; exit 1; }
fi

# ---- 3) 各平台的 tarball ＋ 逐檔驗 sha256 ---------------------------------
FAILED=0
for PLAT in ${PLATFORMS}; do
    F="prime-agent-${VER}-${PLAT}.tar.gz"
    EXPECT="$(awk -v f="${F}" '$2 == f {print $1}' "${DIR}/SHA256SUMS")"
    if [ -z "${EXPECT}" ]; then
        echo "[mirror] 跳過 ${F}（SHA256SUMS 裡沒有這一個）"
        continue
    fi
    if [ "${REFRESH}" = 0 ] && [ -s "${DIR}/${F}" ] && [ "$(sha256_of "${DIR}/${F}")" = "${EXPECT}" ]; then
        echo "[mirror] 已快取且相符: ${F}  ($(wc -c < "${DIR}/${F}" | tr -d ' ') B)"
        continue
    fi
    printf "[mirror] 抓 %-42s " "${F}"
    curl -fsSL --max-time 180 -o "${DIR}/${F}" \
        -w '%{size_download}B %{time_total}s %{speed_download}B/s\n' \
        "${BASE}/releases/v${VER}/${F}" || {
        echo "[mirror] 失敗: 下載 ${F}" >&2; FAILED=1; continue; }
    GOT="$(sha256_of "${DIR}/${F}")"
    if [ "${GOT}" != "${EXPECT}" ]; then
        echo "[mirror] 失敗: ${F} 的 sha256 不符（exp ${EXPECT} got ${GOT}）" >&2
        FAILED=1
    else
        echo "[mirror]       sha256 ✔"
    fi
done
if [ "${FAILED}" != 0 ]; then
    echo "[mirror] 有檔案失敗 —— 不產生 tarball，讓安裝腳本退回直接下載" >&2
    exit 1
fi

# ---- 4) 打包成單一 tarball（adapter 直接送這一個檔）----------------------
echo "[mirror] 打包 ${TARBALL}"
rm -f "${TARBALL}"
tar -czf "${TARBALL}" -C "${ROOT}" . || { echo "[mirror] 打包失敗" >&2; exit 1; }
echo "[mirror] 完成: $(du -sh "${ROOT}" | awk '{print $1}') 目錄 / $(du -sh "${TARBALL}" | awk '{print $1}') tarball"
echo "[mirror] 內容:"
tar -tzf "${TARBALL}" | sed 's/^/  /'
