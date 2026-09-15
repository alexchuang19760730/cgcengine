#!/bin/bash
# check_build_tracked.sh — pre-commit 檢查：build/bin 的 dylib 與 rpath 是否被 git 追蹤
#
# 背景（見 docs/檢討報告_2026-08-26_git與build追蹤疏失.md）：
#   疏失 B: 版本化 dylib（libllama.0.0.15 等）被 .gitignore 的 /build* 一刀切忽略
#            → git 完全看不到、rebuild 一覆寫就永久遺失（29.85 t/s build 一度救不回來）
#   疏失 D: exe 的 @rpath 被改到別的 build 目錄 / symlink 指到不相容 dylib
#            → production 載入錯誤 dylib，batch cap / garbled output
# 本 script 在每次 commit 前自動驗證這兩個洞沒有再發生：
#   1. build/bin 裡每個 *.dylib、必要 dylib 前綴、關鍵 exe 都必須被 git 追蹤（不得 untracked / 被忽略）
#   2. symlink（如 libllama.0.dylib）必須解析到「已追蹤」的目標
#   3. 關鍵 exe 的 @rpath 必須指向本 repo 自己的 build/bin（不得被改到別處）
#
# 檢查 6（2026-08-30 新增）：main⊆dev 分支包含性
#   「main 有的 dev 都要有」— 在 dev 上 commit 時，本地 main 與 github0823/main
#   必須已是 HEAD 的祖先；在 main 上直接 commit 會使 main 領先 dev → 擋下。
#   修復方式：git merge main（在 dev 上）整合後再 commit；晉升一律走
#   git switch main && git merge --ff-only dev（fast-forward 不觸發本 hook）。
#
# 檢查 7（2026-08-30 新增，commit 4a746c724 死鎖教訓）：CGC 專家快取死鎖防護（靜態）
#   事件：CGC_PREFETCH_SRC=hist 非 MTP prefill 100% 掛死於 ensure_batch(il=1)。
#   根因：pick_slot 用「last_use >= batch_tick」啟發式近似「本 batch 持有的 slot」，
#         bg 洪泛填充完成的 ++tick 把全部 slot 推進保護集 → pick_slot 永遠 -1 →
#         裸 bg_cv.wait 死鎖。教訓：近似狀態會隨並發腐爛，必須用精確集合（mask）。
#   7a pick_slot 必須用 batch_mask（精確集合），不得回到 min_tick（啟發式）
#   7b ensure_batch 的 batch_owned mask 必須在位（宣告 + hit 標記 + 分配標記）
#   7c ensure_batch/ensure_slot 的掛死看門狗（FATAL）必須在位（等待必須可證偽）
#   7d 裸 bg_cv.wait(lk)（無述詞）數量 ≤ in_flight 守衛數量（新等待點必須掛守衛）
#
# 檢查 8（2026-08-30 新增）：原始碼 ↔ build 產物同步
#   專案硬性要求：build 產物隨原始碼一起 commit（checkout 免重建可驗證）。
#   改了 llama 原始碼的 commit：(a) build/bin 產物必須一起 staged、
#   (b) binary mtime 不得老於 staged 原始碼（= 有重建過）。
#   ALLOW_STALE_BIN=1 可跳過 (b)。
#
# 檢查 11（2026-09-05 新增）：replay benchmark regression
#   三個應用 profile (qa-zh / longform-zh / coding) 跑 quality / prefill_tps /
#   decode_tps / peak_rss_mb 12 個指標，跟 HEAD 的 .replay_bench_baseline.json
#   比較 (= 上一個 commit 的「已知良好」狀態)。
#   Verdict: 至少 1 個指標改善 (improvement) 且不超過 1 個指標退化 (regression) → PASS
#   否則 → FAIL（precommit hook 拒絕 commit）。
#   只對「代碼」commit 觸發（src/ / scripts/ / *.cpp / *.h / *.py 等），純文件
#   commit (docs/ / *.md / *.html) 不跑。ALLOW_REPLAY_BENCH_BASELINE=1 在 HEAD
#   沒有 baseline 時（bootstrap 階段）改成 SKIP 而非 FAIL。RUN_REPLAY_BENCH=0 可完全關閉。
#
# 用法：
#   scripts/check_build_tracked.sh                 # 檢查目前 cwd 所在的 repo
#   scripts/check_build_tracked.sh --repo PATH     # 檢查指定 repo
#   scripts/check_build_tracked.sh --install-hook  # 安裝到 .git/hooks/pre-commit（每次 commit 自動跑）
#   scripts/check_build_tracked.sh --list-rpaths   # 只列出關鍵 exe 的 @rpath（快速診斷）
#
# env：
#   BIN_DIR          build 目錄相對路徑（預設 build/bin）
#   REQUIRED_EXES    空白分隔的 exe 清單（預設 "llama-simple llama-speculative-simple llama-bench"）
#   REQUIRED_DYLIB   空白分隔的 dylib 前綴（預設 "libllama. libllama-common. libmtmd. libggml."）
#   ALLOW_MISSING=1  允許 build/bin 或檔案不存在（不擋 commit；預設為擋）
#   ALLOW_STALE_BIN=1 允許 binary 比 staged 原始碼舊（不建議；預設為擋）
#   RUN_CGC_PROD_ACCEPT=1  啟用重型生產驗收（短/長 prompt no-0000 + 指標摘錄）
#   RUN_DEPLOY_HARMONYOS_ACCEPT=1  啟用 deploy-harmonyos/macOS 重型驗收（重建 bundle + 啟動檢查）
#   CGC_ACCEPT_SHORT_MIN_TPS / CGC_ACCEPT_LONG_BASE_MIN_TPS / CGC_ACCEPT_LONG_DENSE_MIN_TPS
#                    可選：為重型生產驗收加 t/s 下限（未設 = 僅回報，不擋）

set -uo pipefail

REPO_ROOT=""
INSTALL_HOOK=0
LIST_RPATHS=0
HOOK_BIN_DIR=""
BIN_DIR="${BIN_DIR:-build/bin}"
REQUIRED_EXES=(${REQUIRED_EXES:-llama-simple llama-speculative-simple llama-bench})
REQUIRED_DYLIB=(${REQUIRED_DYLIB:-libllama. libllama-common. libmtmd. libggml.})

usage() {
    sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO_ROOT="$2"; shift 2 ;;
        --install-hook)
            INSTALL_HOOK=1; shift
            # 可選擇性接 BIN_DIR：--install-hook src/llama.cpp/build/bin
            case "${1:-}" in ''|--*) : ;; *) HOOK_BIN_DIR="$1"; shift ;; esac
            ;;
        --list-rpaths) LIST_RPATHS=1; shift ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done

if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "error: not inside a git repo（用 --repo PATH 指定）"; exit 2; }
fi
# 真正 repo root：.git 可能在上層（如 llama.cpp-master 只是 llama-src repo 的子目錄）
REPO_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || echo "$REPO_ROOT")"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)" || { echo "error: repo 不存在: $REPO_ROOT"; exit 2; }
# pre-commit hook 的落點必須是「git 真正會去讀的目錄」。
#   --absolute-git-dir 給的是 worktree 私有目錄（linked worktree 下為
#   <common>/.git/worktrees/<name>）；寫進去 git 根本不讀 ⇒ 印「已安裝」但等於沒裝。
#   --git-path hooks 才是 git 讀的那個（common dir 的 hooks）。
# 但 --git-path 在 main worktree 下會回「相對」路徑（".git/hooks"），那是相對 git 的
#   cwd（= $REPO_ROOT），不是相對本 script 的 cwd ⇒ 必須錨定成絕對路徑，
#   否則下面沒 cd 就直接寫 $HOOK_DIR/pre-commit 會落到錯的地方（甚至靜默失敗）。
GIT_HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks 2>/dev/null)"
case "$GIT_HOOKS_DIR" in
    /*) : ;;                                              # 已是絕對路徑
    '') : ;;                                              # 空 = 不是 git repo，留給下面擋
    *)  GIT_HOOKS_DIR="$REPO_ROOT/$GIT_HOOKS_DIR" ;;       # 相對 ⇒ 錨定到 REPO_ROOT
esac
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir 2>/dev/null)"  # 僅用於死雙胞胎偵測
BIN_ABS="$REPO_ROOT/$BIN_DIR"

# 解析 symlink / 正規化絕對路徑（macOS 的 readlink 沒有 -f，用 python3）
realpath_() { python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

fail_count=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; fail_count=$((fail_count + 1)); }
info() { printf '  INFO  %s\n' "$1"; }

echo "repo: $REPO_ROOT"
echo "build: $BIN_ABS"

# --- install-hook 模式：把自己複製成 pre-commit hook（內嵌 repo 與 build 路徑） ---
if [ "$INSTALL_HOOK" = 1 ]; then
    [ -n "$GIT_HOOKS_DIR" ] || { echo "error: $REPO_ROOT 不是 git repo"; exit 2; }
    HOOK_DIR="$GIT_HOOKS_DIR"
    [ -d "$HOOK_DIR" ] || mkdir -p "$HOOK_DIR"
    HOOK_PATH="$HOOK_DIR/pre-commit"
    # 這是「活」的共用 hook（全部 worktree 共用一顆）⇒ 蓋掉前先留一份
    if [ -f "$HOOK_PATH" ]; then
        cp -p "$HOOK_PATH" "$HOOK_PATH.bak.$(date '+%Y%m%d_%H%M%S')"
    fi
    hook_bin="${HOOK_BIN_DIR:-$BIN_DIR}"
    cat > "$HOOK_PATH" <<EOF
#!/bin/bash
# generated by check_build_tracked.sh --install-hook ($(date '+%Y-%m-%d %H:%M'))
# 落點是 git 真正會讀的 hooks 目錄（common dir）⇒ 同一 repo 的所有 worktree 共用這一顆。
# 因此 REPO_ROOT 必須在「執行時」解析、不能寫死安裝當下的那一個：否則從別的 worktree
# commit 時會去驗錯的 repo（偽 PASS），這正是本檔案之前落到死路徑的原因之一。
# 每次 commit 前驗證 build/bin 的 dylib 與 rpath 都被 git 追蹤 + main⊆dev 包含性
set -euo pipefail
export BIN_DIR='$hook_bin'
REPO_ROOT="\$(git rev-parse --show-toplevel 2>/dev/null)"
exec "\$REPO_ROOT/scripts/check_build_tracked.sh" --repo "\$REPO_ROOT"
EOF
    chmod +x "$HOOK_PATH"
    echo "已安裝 pre-commit hook: $HOOK_PATH (BIN_DIR=$hook_bin)"
    echo "  git 認定的 hooks 目錄: $(git -C "$REPO_ROOT" rev-parse --git-path hooks 2>/dev/null)"
    # 死雙胞胎偵測：git 不讀的 worktree 私有 hooks 裡若還有 pre-commit，
    # 那顆是地雷（看起來像活的）。2026-09-09 那次就是這樣靜默裝錯地方。
    # 注意：在 main worktree 下 --absolute-git-dir 就是 common dir 本身，
    # 此時兩者同一顆、不是雙胞胎 ⇒ 必須先比對目錄是否真的不同，否則會誤報。
    PRIV_HOOKS="$(cd "$GIT_DIR/hooks" 2>/dev/null && pwd -P)"
    REAL_HOOKS="$(cd "$HOOK_DIR" 2>/dev/null && pwd -P)"
    if [ -f "$GIT_DIR/hooks/pre-commit" ] && [ "$PRIV_HOOKS" != "$REAL_HOOKS" ]; then
        echo "  WARN  偵測到 git 不讀的死雙胞胎 hook: $GIT_DIR/hooks/pre-commit" >&2
    fi
    exit 0
fi

# ============ 檢查 6：main⊆dev 分支包含性（main 有的 dev 都要有）============
# 放在 build/bin 存在性檢查之前：無論本 repo 有無 build 目錄都要驗分支紀律。
echo "--- 檢查 main⊆dev 包含性（防兩線分岔） ---"
CUR_BRANCH="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true)"
for ref in refs/heads/main refs/remotes/github0823/main; do
    git -C "$REPO_ROOT" show-ref --verify --quiet "$ref" || continue
    ref_name="${ref#refs/}"
    if [ "$CUR_BRANCH" = "main" ] && [ "$ref" = "refs/heads/main" ]; then
        fail "正在 main 上直接 commit - 這會使 main 領先 dev（違反 main⊆dev）。請改在 dev commit，再 git switch main && git merge --ff-only dev 晉升"
        break
    fi
    if [ -n "$CUR_BRANCH" ] && [ "$CUR_BRANCH" != "dev" ]; then
        pass "非 dev 分支 (${CUR_BRANCH}) - 跳過 main⊆dev 檢查"
        break
    fi
    if git -C "$REPO_ROOT" merge-base --is-ancestor "$ref" HEAD 2>/dev/null; then
        pass "main⊆dev: $ref_name 已包含於 HEAD"
    else
        fail "dev 缺少 $ref_name 的 commit（main 有的 dev 沒有）。修復：git merge $ref_name 整合後再 commit（需要時先 git fetch github0823）"
    fi
done

SKIP_RPATH_CHECK=0
if [ -n "$CUR_BRANCH" ] && [ "$CUR_BRANCH" != "dev" ]; then
    SKIP_RPATH_CHECK=1
fi

HAVE_BIN_DIR=1
if [ ! -d "$BIN_ABS" ]; then
    HAVE_BIN_DIR=0
    if [ "${ALLOW_MISSING:-0}" = 1 ]; then
        echo "SKIP  $BIN_DIR 不存在，ALLOW_MISSING=1 → 跳過 build/bin 追蹤檢查"
    else
        echo "SKIP  $BIN_DIR 不存在（本 repo 若無生產 build 屬正常；仍會繼續跑其他驗收）"
    fi
fi

# ============ 檢查 1：build/bin 內每個 *.dylib 都必須被 git 追蹤 ============
# A build artifact that is not tracked has one of exactly two causes, and the two need
# opposite fixes. diagnose_untracked names which one it is, because the confusing case is the
# first: the project's own .gitignore once had a bare `build/` rule that silently swallowed
# build/bin, so this gate reported "untracked dylib" for every artifact and the real cause
# (an ignore rule) was nowhere in the output.
#   * ignored  -> find the LAST matching rule. The build/bin carve-out already exists at the
#                 TOP of src/llama.cpp/.gitignore (`/build*` -> `!/build/` -> `/build/*` ->
#                 `!/build/bin/` -> `!/build/bin/*`); a plain `build/` further down overrides
#                 it via last-match-wins. Fix = remove/narrow that rule -- NOT `git add -f`,
#                 which tracks the artifact once and lets the next rebuild vanish silently.
#   * not staged -> plain `git add` of the artifact is the fix.
diagnose_untracked() {
    local rel="$1" rule
    # `-q` is the only reliable "is it ignored" test: `check-ignore -v` also prints NEGATION
    # rules (`!/build/bin/*.dylib`), i.e. the patterns that make a path NOT ignored. Testing
    # only for non-empty output therefore misreported a merely-unstaged file as ignored
    # (caught by the branch test for exactly that case).
    if git -C "$REPO_ROOT" check-ignore -q -- "$rel" 2>/dev/null; then
        rule="$(git -C "$REPO_ROOT" check-ignore -v -- "$rel" 2>/dev/null | head -1)"
        echo "      原因: 被 .gitignore 靜默忽略 -> $rule"
        echo "      修法: 找出那條規則並縮小它（build/bin 的 carve-out 已在 .gitignore 頂部；"
        echo "            最後一條匹配的規則勝出，所以後面的 build/ 會覆蓋前面的 !/build/bin/）。"
        echo "            不要用 git add -f —— 那只會追蹤一次，下次重建又會靜默消失。"
    else
        echo "      原因: 檔案未被納入 index（沒有 ignore 規則擋它）-> git add \"$rel\""
    fi
}

if [ "$HAVE_BIN_DIR" = 1 ]; then
    echo "--- 檢查 dylib 被追蹤（洞 B） ---"
    DYLIBS=()
    while IFS= read -r f; do DYLIBS+=("$f"); done < <(find "$BIN_ABS" -maxdepth 1 -name '*.dylib' -type f 2>/dev/null | sort)
    if [ "${#DYLIBS[@]}" -eq 0 ]; then
        fail "build/bin 沒有任何 *.dylib"
    else
        for f in "${DYLIBS[@]}"; do
            rel="${f#"$REPO_ROOT"/}"
            if git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
                pass "追蹤 OK: $rel"
            else
                fail "未追蹤 dylib（會被 git 靜默忽略/遺失）: $rel"
                diagnose_untracked "$rel"
            fi
        done
    fi

    # ============ 檢查 2：必要 dylib 前綴至少要有一個已追蹤的檔案 ============
    for prefix in "${REQUIRED_DYLIB[@]}"; do
        found=0
        for f in "${DYLIBS[@]}"; do
            base="$(basename "$f")"
            case "$base" in
                "$prefix"*)
                    rel="${f#"$REPO_ROOT"/}"
                    if git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then found=1; fi
                    ;;
            esac
        done
        if [ "$found" = 1 ]; then
            pass "必要 dylib 前綴已追蹤: $prefix"
        else
            fail "缺少已追蹤的必要 dylib 前綴: $prefix"
        fi
    done

    # ============ 檢查 3：symlink 必須解析到已追蹤的目標 ============
    echo "--- 檢查 symlink 目標被追蹤 ---"
    LINKS=()
    while IFS= read -r l; do LINKS+=("$l"); done < <(find "$BIN_ABS" -maxdepth 1 -type l 2>/dev/null | sort)
    for l in "${LINKS[@]}"; do
        target="$(realpath_ "$l")"
        rel_target="${target#"$REPO_ROOT"/}"
        if [ "$rel_target" = "$target" ]; then
            fail "symlink 指向 repo 外: $l -> $target"
            continue
        fi
        if git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel_target" >/dev/null 2>&1; then
            pass "symlink 目標已追蹤: $(basename "$l") -> $rel_target"
        else
            fail "symlink 目標未追蹤（洞 B）: $l -> $rel_target"
        fi
    done

    # ============ 檢查 4：關鍵 exe 存在且被追蹤 ============
    echo "--- 檢查關鍵 exe 被追蹤 ---"
    for exe in "${REQUIRED_EXES[@]}"; do
        p="$BIN_ABS/$exe"
        if [ ! -x "$p" ]; then
            fail "關鍵 exe 不存在: $exe"
            continue
        fi
        rel="${p#"$REPO_ROOT"/}"
        if git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
            pass "exe 追蹤 OK: $rel"
        else
            fail "未追蹤 exe: $rel"
            diagnose_untracked "$rel"
        fi
    done

    # ============ 檢查 5：關鍵 exe 的 @rpath 必須指向本 repo 自己的 build/bin ============
    echo "--- 檢查 @rpath（洞 D） ---"
    if [ "$SKIP_RPATH_CHECK" = 1 ]; then
        echo "SKIP  非 dev 分支，worktree 可共用既有 binary；跳過 @rpath 驗收"
    else
        for exe in "${REQUIRED_EXES[@]}"; do
            p="$BIN_ABS/$exe"
            [ -x "$p" ] || continue
            # otool: LC_RPATH 下的 path 行 → 取第 2 欄（絕對路徑）
            rpaths=($(otool -l "$p" 2>/dev/null | awk '/LC_RPATH/{f=1} f && /path /{print $2; f=0}'))
            if [ "${#rpaths[@]}" -eq 0 ]; then
                fail "@rpath 不存在（可能載入不相容 dylib）: $exe"
                continue
            fi
            ok=0
            for rp in "${rpaths[@]}"; do
                rp_real="$(realpath_ "$rp")"
                if [ "$rp_real" = "$BIN_ABS" ]; then ok=1; fi
            done
            if [ "$ok" = 1 ]; then
                pass "@rpath 指向本 repo build/bin: $exe -> ${rpaths[*]}"
            else
                fail "@rpath 被改到別處 (洞 D): $exe -> ${rpaths[*]} (預期 ${BIN_ABS})"
            fi
        done
    fi

    # ============ 檢查 7：CGC 專家快取死鎖防護（2026-08-30 hist-prefetch 死鎖，4a746c724 修復） ============
    # 教訓一（近似狀態隨並發腐爛）：「用啟發式近似某狀態集合」的保護邏輯，只要新增一個
    #   併發寫入者（bg 填充的 ++tick）就會失效 → 必須用精確集合（batch_owned mask）。
    # 教訓二（等待必須可證偽）：條件不可能再滿足的 wait 必須大聲 FATAL，不得無聲掛死。
    echo "--- 檢查 CGC 死鎖防護（hist-prefetch 教訓） ---"
    # llama 原始碼根：從 build/bin 往上兩層（src/llama.cpp/build/bin → src/llama.cpp；build/bin → repo 根）
    LLAMA_ROOT="$(dirname "$(dirname "$BIN_ABS")")"
    EC_SRC="$LLAMA_ROOT/src/llama-expert-cache.cpp"
    if [ ! -f "$EC_SRC" ]; then
        echo "SKIP  $EC_SRC 不存在（非 CGC fork repo）"
    else
        # 7a：pick_slot 用精確 batch_mask，不得回到 min_tick 啟發式
        if grep -q 'uint64_t min_tick' "$EC_SRC"; then
            fail "7a pick_slot 回到 min_tick 啟發式（近似狀態隨並發腐爛 — 2026-08-30 死鎖根因）"
        elif grep -qF 'const uint8_t * batch_mask' "$EC_SRC"; then
            pass "7a pick_slot 用精確 batch_mask（非啟發式近似）"
        else
            fail "7a pick_slot 簽名找不到 batch_mask（ensure_batch 的 in-batch 保護被改壞？）"
        fi
        # 7b：batch_owned mask 在位（宣告 + hit 標記 + miss 分配標記）
        n_mask="$(grep -cF 'batch_owned' "$EC_SRC" || true)"
        if [ "${n_mask:-0}" -ge 3 ]; then
            pass "7b batch_owned mask 在位 (${n_mask} 處 >= 3)"
        else
            fail "7b batch_owned mask 不完整 (${n_mask} 處 < 3: 宣告/hit 標記/分配標記)"
        fi
        # 7c：掛死看門狗在位（等待可證偽）
        if grep -qF 'FATAL ensure_batch' "$EC_SRC" && grep -qF 'FATAL ensure_slot' "$EC_SRC"; then
            pass "7c 掛死看門狗在位（ensure_batch + ensure_slot FATAL）"
        else
            fail "7c 掛死看門狗被移除（pick_slot -1 且無 in-flight 時的等待不可證偽 = 無聲掛死）"
        fi
        # 7d：每個「等待填充完成」的裸 wait 都要有 in_flight 守衛
        n_bare="$(grep -cF 'cache->bg_cv.wait(lk);' "$EC_SRC" || true)"
        n_guard="$(grep -cF 'in_flight = false' "$EC_SRC" || true)"
        if [ "${n_bare:-0}" -le "${n_guard:-0}" ]; then
            pass "7d 裸 bg_cv.wait ${n_bare} 個 ≤ in_flight 守衛 ${n_guard} 個"
        else
            fail "7d 裸 bg_cv.wait(lk) ${n_bare} 個 > 守衛 ${n_guard} 個 — 新等待點必須先證明有 in-flight 填充可等"
        fi
    fi

    # ============ 檢查 8：原始碼 ↔ build 產物同步（checkout 免重建可驗證不變量） ============
    echo "--- 檢查 原始碼↔binary 同步 ---"
    STAGED_FILES="$(git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)"
    if [ -z "$STAGED_FILES" ]; then
        echo "SKIP  無 staged 檔案（手動跑 hook 時屬正常；commit 時必有 staged）"
    else
        # staged 的 llama 原始碼（src/ 與 examples/ 下的 .cpp/.h/.c/.mm/.metal — 都編進 binary）
        staged_src=()
        while IFS= read -r f; do
            case "$f" in
                *src/*.cpp|*src/*.h|*src/*.c|*src/*.mm|*src/*.metal|*examples/*.cpp|*examples/*.h)
                    [ -f "$REPO_ROOT/$f" ] && staged_src+=("$f") ;;
            esac
        done <<< "$STAGED_FILES"
        if [ "${#staged_src[@]}" -eq 0 ]; then
            pass "8 無 llama 原始碼變更（僅 doc/腳本/產物）"
        else
            # (a) build/bin 產物必須在同一個 commit staged
            #     例外: 若所有 dylib byte 跟 HEAD 相同 (= source 改的是 env-gated
            #     dead code, 編譯器優化掉, dylib 內容無 functional change),
            #     允許 source 跟 dylib 在不同 commit (env-gated-safe), 用
            #     ALLOW_ENV_GATED_BIN=1 顯式開啟。
            staged_bin="$(grep -F 'build/bin/' <<< "$STAGED_FILES" | head -1 || true)"
            if [ -n "$staged_bin" ]; then
                pass "8 build 產物隨原始碼 staged (${staged_bin##*/})"
            else
                # 檢查所有 dylib 是否都跟 HEAD 相同
                env_gated_safe=0
                if [ "${ALLOW_ENV_GATED_BIN:-0}" = 1 ] && [ -d "$BIN_ABS" ]; then
                    diff_count=0
                    same_count=0
                    # 用 cd + find 讓輸出是相對路徑, 再用 basename 拿檔名
                    # (macOS BSD find 沒有 -printf, 用 sed 去前綴 ./ 拿 basename)
                    while IFS= read -r path; do
                        if [ -z "$path" ]; then continue; fi
                        rel="${path##*/}"  # basename 等效
                        if [ -z "$rel" ] || [ "$rel" = "." ]; then continue; fi
                        full_rel="$BIN_DIR/$rel"
                        if git -C "$REPO_ROOT" show "HEAD:$full_rel" 2>/dev/null | diff -q - "$BIN_ABS/$rel" >/dev/null 2>&1; then
                            same_count=$((same_count + 1))
                        else
                            diff_count=$((diff_count + 1))
                        fi
                    done < <(cd "$BIN_ABS" && find . -maxdepth 1 -name '*.dylib' -type f 2>/dev/null)
                    if [ "$diff_count" = 0 ] && [ "$same_count" -gt 0 ]; then
                        info "8 所有 ${same_count} 個 dylib 跟 HEAD byte-identical + ALLOW_ENV_GATED_BIN=1 → 視為 env-gated no-op commit (不需 stage build/bin)"
                        env_gated_safe=1
                    fi
                fi
                if [ "$env_gated_safe" = 1 ]; then
                    pass "8 env-gated no-op commit (ALLOW_ENV_GATED_BIN=1, ${same_count} dylib byte-identical)"
                else
                    fail "8 stage 了 ${#staged_src[@]} 個原始碼但沒 stage build/bin 產物 — git add src/llama.cpp/build/bin/ 一起進 commit（如果 source 改的是 env-gated dead code, dylib byte-identical, 可用 ALLOW_ENV_GATED_BIN=1 跳過）"
                fi
            fi
            # (b) binary mtime 不得老於最新 staged 原始碼（有重建過；ALLOW_STALE_BIN=1 跳過）
            if [ "${ALLOW_STALE_BIN:-0}" != 1 ]; then
                newest_src=0
                for f in "${staged_src[@]}"; do
                    m="$(stat -f %m "$REPO_ROOT/$f" 2>/dev/null || echo 0)"
                    [ "$m" -gt "$newest_src" ] && newest_src="$m"
                done
                newest_bin=0
                for f in "$BIN_ABS"/*.dylib; do
                    [ -f "$f" ] || continue
                    m="$(stat -f %m "$f" 2>/dev/null || echo 0)"
                    [ "$m" -gt "$newest_bin" ] && newest_bin="$m"
                done
                if [ "$newest_bin" -ge "$newest_src" ]; then
                    pass "8 binary 比 staged 原始碼新（已重建）"
                else
                    fail "8 binary 比 staged 原始碼舊 — 先 cmake --build src/llama.cpp/build --target llama-simple llama-speculative-simple -j 8 再 commit（ALLOW_STALE_BIN=1 可跳過）"
                fi
            fi
        fi
    fi
else
    echo "--- 檢查 dylib 被追蹤（洞 B） ---"
    echo "SKIP  $BIN_DIR 不存在 → 跳過 build/bin 追蹤 / rpath / deadlock / binary sync 檢查"
fi

# ============ 檢查 9：重型生產驗收（短/長 prompt no-0000 + 指標摘錄） ============
echo "--- 檢查 生產 MTP 驗收（短/長 prompt） ---"
RUN_N30="$REPO_ROOT/scripts/run_n30cache.sh"
if [ "${RUN_CGC_PROD_ACCEPT:-0}" != 1 ]; then
    echo "SKIP  RUN_CGC_PROD_ACCEPT=1 未啟用（重型驗收保留在本 script 內，需手動開）"
elif [ ! -x "$RUN_N30" ]; then
    fail "9 找不到生產腳本: $RUN_N30"
else
    prod_accept_check() {
        local label="$1"; shift
        local expect="$1"; shift
        local min_tps="${1:-}"; shift || true
        local log="/tmp/cgc_precommit_${label}.log"
        local out="/tmp/cgc_precommit_${label}.out"

        if "$RUN_N30" "$@" >"$log" 2>&1; then
            :
        else
            fail "9 ${label} 執行失敗 (見 ${log})"
            return
        fi

        cp /tmp/n30cache.out "$out" 2>/dev/null || true

        if [ ! -f "$out" ]; then
            fail "9 ${label} 沒有產出 /tmp/n30cache.out"
            return
        fi

        if grep -q '0000' "$out"; then
            fail "9 ${label} 輸出含 0000 退化"
        else
            pass "9 ${label} 無 0000 退化"
        fi

        if [ -n "$expect" ]; then
            if grep -qF "$expect" "$out"; then
                pass "9 ${label} 命中關鍵文本：$expect"
            else
                fail "9 ${label} 未命中關鍵文本：$expect"
            fi
        fi

        local tps accept hit tpot
        tps="$(sed -n 's/.*speed: *\([0-9.][0-9.]*\) t\/s.*/\1/p' "$log" | tail -1)"
        accept="$(sed -n 's/.*accept *= *\([0-9.][0-9.]*%\).*/\1/p' "$log" | tail -1)"
        hit="$(sed -n 's/.*hit rate \([0-9.][0-9.]*%\).*/\1/p' "$log" | tail -1)"
        if [ -n "${tps:-}" ]; then
            tpot="$(python3 - <<PY
tps = float("${tps}")
print(f"{1000.0/tps:.2f}")
PY
)"
            info "9 ${label} 指標：t/s=${tps} TPOT=${tpot}ms/tok accept=${accept:-n/a} hit=${hit:-n/a}"
            if [ -n "$min_tps" ]; then
                if python3 - <<PY
import sys
sys.exit(0 if float("${tps}") >= float("${min_tps}") else 1)
PY
                then
                    pass "9 ${label} t/s ${tps} >= 門檻 ${min_tps}"
                else
                    fail "9 ${label} t/s ${tps} < 門檻 ${min_tps}"
                fi
            fi
        else
            fail "9 ${label} 無法從 log 解析 t/s (見 ${log})"
        fi
    }

    prod_accept_check short_base "The capital of France is Paris" "${CGC_ACCEPT_SHORT_MIN_TPS:-}" \
        -m qwen36 --mtp --seed 42 -n 48 -p "The capital of France is"
    prod_accept_check long_base "" "${CGC_ACCEPT_LONG_BASE_MIN_TPS:-}" \
        -m qwen36 --mtp --steady
    prod_accept_check long_dense "" "${CGC_ACCEPT_LONG_DENSE_MIN_TPS:-}" \
        -m qwen36 --mtp --dense-iq4x --steady
fi

# ============ 檢查 10：deploy-harmonyos macOS bundle 可重建 / 可啟動 ============
echo "--- 檢查 deploy-harmonyos macOS bundle 驗收 ---"
DEPLOY_MACOS_CHECK="$REPO_ROOT/deploy-harmonyos/macos/check-macos-bundle.sh"
if [ "${RUN_DEPLOY_HARMONYOS_ACCEPT:-0}" != 1 ]; then
    echo "SKIP  RUN_DEPLOY_HARMONYOS_ACCEPT=1 未啟用（deploy-harmonyos 驗收保留在本 script 內，需手動開）"
elif [ ! -x "$DEPLOY_MACOS_CHECK" ]; then
    fail "10 找不到 deploy 驗收腳本: $DEPLOY_MACOS_CHECK"
else
    if "$DEPLOY_MACOS_CHECK" > /tmp/cgc_precommit_deploy_harmonyos.log 2>&1; then
        pass "10 deploy-harmonyos macOS bundle 可重建且 llama-server/llama-simple 可啟動"
        info "10 deploy 驗收記錄：/tmp/cgc_precommit_deploy_harmonyos.log"
    else
        fail "10 deploy-harmonyos 驗收失敗（見 /tmp/cgc_precommit_deploy_harmonyos.log）"
    fi
fi

# ============ 檢查 11：replay benchmark regression（代碼 commit 才觸發）============
# 三個應用 profile (qa-zh / longform-zh / coding) 跑 12 個指標（3 profile × 4 aspect:
# quality / prefill_tps / decode_tps / peak_rss_mb），跟 HEAD 的 .replay_bench_baseline.json
# 比較 (= 上一個 commit 的「已知良好」狀態)。Verdict: 至少 1 個指標改善 + 不超過
# 1 個指標退化 → PASS，否則 FAIL 拒絕 commit。
echo "--- 檢查 replay benchmark 不退化（代碼 commit 才觸發） ---"
if [ "${RUN_REPLAY_BENCH:-1}" != 1 ]; then
    echo "SKIP  RUN_REPLAY_BENCH=0 → 跳過 replay benchmark regression 檢查"
elif [ ! -x "$REPO_ROOT/scripts/check/replay_server_profile.py" ] || [ ! -x "$REPO_ROOT/scripts/check/replay_bench_compare.py" ]; then
    fail "11 缺 replay benchmark 腳本（scripts/check/replay_server_profile.py + replay_bench_compare.py）"
else
    # 1) 判斷本次 commit 是否為「代碼」(非純文件)
    STAGED_FOR_BENCH="$(git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)"
    is_code_path() {
        # 是「代碼」路徑: src/ scripts/ *.cpp *.h *.c *.mm *.metal *.py *.sh
        # Makefile / CMakeLists.txt / *.proto / deploy-harmonyos/* 也算（會影響行為）
        case "$1" in
            src/*|scripts/*|deploy-harmonyos/*) return 0 ;;
            *.cpp|*.h|*.c|*.cc|*.mm|*.metal|*.py|*.sh|*.proto) return 0 ;;
            Makefile|CMakeLists.txt|*.mk) return 0 ;;
        esac
        return 1
    }
    is_doc_path() {
        # 是「文件」路徑: docs/ *.md *.html *.txt（root 級）
        case "$1" in
            docs/*|*.md|*.html) return 0 ;;
        esac
        return 1
    }
    has_code=0
    has_doc=0
    if [ -n "$STAGED_FOR_BENCH" ]; then
        while IFS= read -r f; do
            [ -z "$f" ] && continue
            if is_code_path "$f"; then
                has_code=1
            elif is_doc_path "$f"; then
                has_doc=1
            else
                # 未知類別（如 models/、Backup/）保守視為文件，不觸發
                has_doc=1
            fi
        done <<< "$STAGED_FOR_BENCH"
    fi

    if [ "$has_code" != 1 ]; then
        echo "SKIP  本次 commit 為純文件變更（無 src/ scripts/ *.cpp *.h *.py 等代碼）→ 不觸發 replay benchmark"
    else
        info "11 本次 commit 含代碼改動 (has_code=1, has_doc=${has_doc}) -> 觸發 replay benchmark"
        # 2) 偵測 server PID（CGC_SERVER_PID env > pgrep llama-server --port 8080）
        REPLAY_BENCH="$REPO_ROOT/scripts/check/replay_server_profile.py"
        REPLAY_COMPARE="$REPO_ROOT/scripts/check/replay_bench_compare.py"
        REPLAY_REF="$REPO_ROOT/scripts/check/replay_bench_reference.json"
        CURRENT_OUT="/tmp/.replay_bench_current.json"
        BASELINE_OUT="/tmp/.replay_bench_baseline.json"
        REPORT_OUT="/tmp/.replay_bench_report.json"

        SERVER_PID="${CGC_SERVER_PID:-}"
        if [ -z "$SERVER_PID" ]; then
            SERVER_PID="$(pgrep -f 'llama-server.*--port[[:space:]]*8080' 2>/dev/null | head -1 || true)"
        fi
        if [ -z "$SERVER_PID" ]; then
            fail "11 找不到 llama-server PID（設 CGC_SERVER_PID=... 或啟動 server，RUN_REPLAY_BENCH=0 可跳過）"
        else
            info "11 使用 server PID=$SERVER_PID"
            # 3) 跑 replay benchmark
            #    --warmup --runs 3 --seed 0: 暖 expert pool + median-of-3 + 固定 seed,
            #    消除採樣/ pool 狀態噪聲造成的品質抖動 (0.667↔1.0 誤擋問題)
            if python3 "$REPLAY_BENCH" \
                --all-profiles \
                --server-pid "$SERVER_PID" \
                --reference "$REPLAY_REF" \
                --bench-output "$CURRENT_OUT" \
                --warmup \
                --runs "${REPLAY_RUNS:-3}" \
                --seed 0 \
                --commit "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
                > /tmp/.replay_bench_run.log 2>&1; then
                pass "11 replay benchmark 跑完 ($CURRENT_OUT)"
            else
                fail "11 replay benchmark 跑失敗（見 /tmp/.replay_bench_run.log）"
            fi

            # 4) 拿 baseline: 優先用 staged 版本 (`:path` = index 中的版本), 退回 HEAD
            #    staged 優先的原因: 本次 commit 可能在 baseline 跟 current 兩端都
            #    改 .replay_bench_baseline.json (refresh bootstrap), 比較時要拿「這次
            #    commit 預期的 baseline」, 不是「上一次 commit 的 baseline」。
            #    若 index 也沒有 (純 source 改動, baseline 沒重抓), 退回 HEAD 版本。
            BASELINE_SRC=""
            if git -C "$REPO_ROOT" show ":.replay_bench_baseline.json" > "$BASELINE_OUT" 2>/dev/null; then
                BASELINE_SRC="staged"
            elif git -C "$REPO_ROOT" show "HEAD:.replay_bench_baseline.json" > "$BASELINE_OUT" 2>/dev/null; then
                BASELINE_SRC="HEAD"
            fi

            if [ -z "$BASELINE_SRC" ]; then
                if [ "${ALLOW_REPLAY_BENCH_BASELINE:-0}" = 1 ]; then
                    echo "SKIP  找不到 baseline（HEAD 沒有 .replay_bench_baseline.json）但 ALLOW_REPLAY_BENCH_BASELINE=1 → 跳過比較"
                else
                    fail "11 找不到 baseline：HEAD 沒有 .replay_bench_baseline.json。需先 bootstrap: ALLOW_REPLAY_BENCH_BASELINE=1 git commit 一份 .replay_bench_baseline.json 進版（先單獨 commit 避免被擋）"
                fi
            else
                info "11 baseline 來源: $BASELINE_SRC"
                if python3 "$REPLAY_COMPARE" \
                    --current "$CURRENT_OUT" \
                    --baseline "$BASELINE_OUT" \
                    --reference "$REPLAY_REF" \
                    --report "$REPORT_OUT" \
                    > /tmp/.replay_bench_compare.log 2>&1; then
                    pass "11 replay benchmark 比較 PASS ($BASELINE_SRC)"
                    info "11 詳細 report: $REPORT_OUT (commit 這份新 baseline 可選)"
                else
                    rc=$?
                    if [ "$rc" = 1 ]; then
                        fail "11 replay benchmark 退化 (見 $REPORT_OUT 與 /tmp/.replay_bench_compare.log)"
                    else
                        fail "11 replay benchmark 比較 ERROR (見 /tmp/.replay_bench_compare.log)"
                    fi
                    info "11 此次 replay 輸出: $CURRENT_OUT (不變好即不允許 commit)"
                fi
            fi
        fi
    fi
fi

# ============ 總結 ============
if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "FAIL: $fail_count 項未通過。先修好再 commit（build 產物追蹤 / rpath / main⊆dev / 死鎖防護 / 原始碼↔binary 同步 / 生產驗收 / deploy 驗收 / replay benchmark 不退化）。"
    exit 1
fi
echo ""
echo "OK: build/bin 追蹤與 rpath 正常、main⊆dev 成立、CGC 死鎖防護在位、原始碼↔binary 同步、生產驗收與 deploy 驗收通過/未啟用、replay benchmark regression 通過/未啟用。"
exit 0
