#!/usr/bin/env bash
# 產生 prime-agent 的 **uv bundle**（給容器離線建 Python kernel 用）。
#
# 為什麼需要它：`--autonomous` 的 `bash()` 是**透過 Python REPL 暴露的**，沒有 Python kernel
# 就等於**沒有任何工具**。而 prime-agent 自己的 `--prime-agent-bootstrap` 會去下
# uv ＋ Python ＋ 一批 wheel —— 實測容器對 PyPI 的吞吐只有 **~0.7 MB/min**，20 分鐘都跑不完，
# 而且 `UV_OFFLINE=1` 也擋不住那個停頓。
#
# ★ 所以不要用它的 bootstrap，改成「host 把料備齊 → 注入 → 容器內離線建」。
#
# ============================ 2026-09-17 第二版（重要改動）============================
# 第一版只有 19 MB（uv ＋ 幾個 pure-python wheel），理由是「不想帶 managed Python」。
# 那個設計**被實驗推翻**了：prime-agent 對 `PRIME_AGENT_KERNEL_PYTHON` 有一道**硬檢查**，
# 缺 default Python packages 直接拒絕（不是警告）。實測（smoke9）agent 拿到的原文：
#
#   PRIME_AGENT_KERNEL_PYTHON points to a Python missing default Python packages
#   (requests, httpx, yaml, tomli, dotenv, pandas, numpy, scipy, bs4, lxml, pydantic, tyro):
#   /root/.prime/agent/kernel-venv/bin/python
#
# 而那份清單裡有 numpy / scipy / pandas / lxml / pydantic-core —— **都是原生擴充**，
# 所以「只裝 pure-python wheel」在這個要求下不成立：native wheel 綁 ABI。
# ⇒ 為了同時滿足「清單齊全」與「與任務映像無關」，這一版**自帶一份 linux-aarch64 的
#   CPython 3.11**（wheel 全部是 cp311 ⇒ ABI 固定），也就是回到「image-independent」的
#   做法，但**不用 prime-agent 的 bootstrap**（那條會卡死）。
#   附帶好處：3.11 正是 prime-agent 自己期望的版本（執行檔裡 `pPn = "3.11"`）。
#
# ★ 仍然刻意不裝 `mcp`：prime-agent-runtime 宣告了它，但
#   ① `rlm/mcp.py` 是「kernel-owned generic MCP **client** registry」，只 import 標準庫
#      與 `.mcp_base`；mcp_base 對 `mcp` SDK 的 import 全在**函式內部**（實測
#      `import rlm.mcp` 在沒有 mcp 的情況下 **OK**）。
#   ② 裝了反而壞：`mcp` → `pyjwt[crypto]` → `cryptography`，而它的原生擴充在這台 VM 會
#      SIGILL（`import 'cryptography.hazmat.bindings'` → Illegal instruction）。
#   ⇒ 本線（跑 tb 任務）不需要 MCP 工具伺服器。（kernel shim 把 `import rlm.mcp` 放在
#      `bash` 的那個 try 裡，所以它必須 import 得過 —— 上面已證它會過。）
#
# 實測：bundle 約 160 MB；容器內離線建 venv ＋ 裝完 15 個 import **1 秒**。
#
# 用法:
#   bash agent_harness/tb_loop/scripts/fetch-prime-agent-uvbundle.sh            # 缺才做
#   bash agent_harness/tb_loop/scripts/fetch-prime-agent-uvbundle.sh --refresh  # 重做
# 可覆寫: TB_PA_UVBUNDLE（輸出目錄）、TB_PA_UV_URL（uv 二進位來源）、
#         TB_PA_PY_URL（linux CPython 來源）、TB_PA_PYTHON（抓 wheel 用的直譯器）
set -u

OUT="${TB_PA_UVBUNDLE:-${HOME}/.cache/prime-agent-uvbundle}"
TAR="${OUT}/prime-agent-uvbundle.tar.gz"
WHEELS="${OUT}/wheels"
UV_URL="${TB_PA_UV_URL:-https://github.com/astral-sh/uv/releases/download/0.12.15/uv-aarch64-unknown-linux-gnu.tar.gz}"
# linux-aarch64 的 standalone CPython。來源就是 `uv python list --all-platforms --show-urls`
# 印出來的那個（python-build-standalone）。換版本時一起改這裡即可。
PY_URL="${TB_PA_PY_URL:-https://releases.astral.sh/github/python-build-standalone/releases/download/20260807/cpython-3.11.15%2B20260807-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz}"
PY_TAR="${OUT}/cpython-3.11.15-linux-aarch64.tar.gz"
PY="${TB_PA_PYTHON:-$(command -v python3)}"
REFRESH=0
if [ "${1:-}" = "--refresh" ]; then REFRESH=1; fi

# 必須齊全的 wheel 名（prime-agent 的硬檢查清單 ＋ dill ＋ hatchling）。
#   hatchling：`uv pip install --no-deps <prime-agent-runtime 目錄>` 要它的建置後端。
#   dill：prime-agent 自己 managed 環境的成員。
PA_REQUIRED_WHEELS="requests httpx pyyaml tomli python_dotenv pandas numpy scipy
beautifulsoup4 lxml pydantic tyro dill hatchling"

echo "[uvbundle] 輸出     = ${TAR}"
echo "[uvbundle] uv 來源   = ${UV_URL}"
echo "[uvbundle] py 來源   = ${PY_URL}"
echo "[uvbundle] 抓 wheel 用 = ${PY}"

mkdir -p "${OUT}" "${WHEELS}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

# ---- 1) linux 的 uv 二進位 -------------------------------------------------
if [ "${REFRESH}" = 1 ] || [ ! -s "${OUT}/uv-linux.tar.gz" ]; then
    echo "[uvbundle] 抓 uv（linux-aarch64）"
    curl -fsSL --max-time 180 -o "${OUT}/uv-linux.tar.gz" \
        -w '[uvbundle]   %{size_download}B %{time_total}s %{speed_download}B/s\n' "${UV_URL}" || {
        echo "[uvbundle] 失敗: 抓不到 uv" >&2; exit 1; }
fi
mkdir -p "${STAGE}/.local/bin"
tar -xzf "${OUT}/uv-linux.tar.gz" -C "${STAGE}" 2>/dev/null
UVBIN="$(find "${STAGE}" -maxdepth 2 -type f -name uv | head -1)"
if [ -z "${UVBIN}" ]; then echo "[uvbundle] 失敗: tarball 裡沒有 uv" >&2; exit 1; fi
cp "${UVBIN}" "${STAGE}/.local/bin/uv"
UVX="$(find "${STAGE}" -maxdepth 2 -type f -name uvx | head -1)"
[ -n "${UVX}" ] && cp "${UVX}" "${STAGE}/.local/bin/uvx"
rm -rf "${STAGE}/uv-aarch64-unknown-linux-gnu"
echo "[uvbundle] uv 就位: $(du -h "${STAGE}/.local/bin/uv" | awk '{print $1}')"

# ---- 2) linux 的 standalone CPython 3.11 -----------------------------------
if [ "${REFRESH}" = 1 ] || [ ! -s "${PY_TAR}" ]; then
    echo "[uvbundle] 抓 CPython 3.11（linux-aarch64）"
    curl -fsSL --max-time 300 -o "${PY_TAR}" \
        -w '[uvbundle]   %{size_download}B %{time_total}s %{speed_download}B/s\n' "${PY_URL}" || {
        echo "[uvbundle] 失敗: 抓不到 CPython" >&2; exit 1; }
fi
mkdir -p "${STAGE}/py311"
tar -xzf "${PY_TAR}" -C "${STAGE}/py311" --strip-components=1 2>/dev/null || {
    echo "[uvbundle] 失敗: 解不開 CPython" >&2; exit 1; }
if [ ! -x "${STAGE}/py311/bin/python3.11" ]; then
    echo "[uvbundle] 失敗: 解出來的目錄裡沒有 bin/python3.11" >&2; exit 1; fi
# ★ 不要在這裡執行它 —— 那是 **linux-aarch64** 的二進位，host 是 macOS，
#   跑起來只會得到 `cannot execute binary file`（我第一版就是這樣，白印一行像錯誤的東西）。
#   版本由檔名帶出即可；真正能不能跑的驗收在容器裡（prime-agent-setup.sh 的步骤 3）。
echo "[uvbundle] CPython 就位: py311/bin/python3.11（linux-aarch64，$(du -sh "${STAGE}/py311" | awk '{print $1}')）"

# ---- 3) cp311 的 wheelhouse -------------------------------------------------
#   ★ 平台必須與上面那份 CPython 相符（aarch64 / cp311）；這也是「與任務映像無關」的來源。
#   ★ 快取判準是「**清單裡該有的在不在**」，不是「目錄空不空」—— 目錄裡只躺著一個別的
#     wheel 也算非空，那樣會靜默跳過下載、產出一個缺料的 bundle（實測踩過一次）。
pa_missing_wheel=0
for pa_name in ${PA_REQUIRED_WHEELS}; do
    ls "${WHEELS}/${pa_name}"-*.whl >/dev/null 2>&1 || pa_missing_wheel=1
done
if [ "${REFRESH}" = 1 ] || [ "${pa_missing_wheel}" = 1 ]; then
    echo "[uvbundle] 抓 wheels（cp311 / manylinux aarch64）"
    # shellcheck disable=SC2086
    "${PY}" -m pip download -q -d "${WHEELS}" \
        --only-binary=:all: \
        --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64 \
        --python-version 3.11 --implementation cp --abi cp311 \
        ${PA_REQUIRED_WHEELS} || {
        echo "[uvbundle] 失敗: pip download" >&2; exit 1; }
fi
echo "[uvbundle] wheels（$(ls "${WHEELS}" | wc -l | tr -d ' ') 個 / $(du -sh "${WHEELS}" | awk '{print $1}')）"
mkdir -p "${STAGE}/wheels"
cp "${WHEELS}"/*.whl "${STAGE}/wheels/"

# ---- 4) 打包 ---------------------------------------------------------------
rm -f "${TAR}"
tar czf "${TAR}" -C "${STAGE}" . || { echo "[uvbundle] 失敗: 打包" >&2; exit 1; }
echo "[uvbundle] 完成: $(du -sh "${TAR}" | awk '{print $1}')  （解開約 $(du -sh "${STAGE}" | awk '{print $1}')）"
echo "[uvbundle] 內容（前 12 項）:"
tar -tzf "${TAR}" | head -12 | sed 's/^/[uvbundle]   /'
