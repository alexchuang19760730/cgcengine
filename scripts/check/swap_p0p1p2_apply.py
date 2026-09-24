#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CGC 2026-09-24 swap-miss] P0 / P1 / P2 的定點補丁器 —— `docs/SWAP_MISS_LINK_2026-09-24.md` §4.4 / §7。

為什麼是一支「補丁器」而不是直接手改：
  P0 落在 `llama-model-loader.cpp`（乾淨），P1/P2 落在 `llama-expert-cache.cpp`
  （**線 I 有 218 insertions 未提交**）。統一 diff 對這種會漂移的檔沒有用，
  所以用**錨點字串替換**：每個 hunk 都 `assert count == 1`，命中數不對就**整批不寫入**
  （不留半套修改）。錨點選的是線 I 那批改動沒有碰的程式碼，所以他們 commit 之後也還在。

三件事各自做什麼（**預設全部關閉 ⇒ 不改任何既有行為，可以安全躺在別人的髒檔裡**）：

  P0  `CGC_EXPERT_SKIP_READRAW=1`
      skip-load 的 expert tensor 不 `read_raw`（CPU buffer 保持佔位，位元組由 pool fill 的
      pread 供給）。這 10960 MiB 匿名記憶體就是 13030 + 8192 = 21222 > 16384 的那一半。
      三個 guard：① 只對 loader 自己放上 skip-load 路徑的 tensor（skip_no_read，
      含 `!buft` 前提）；② `L4_SKIP_LAYER0` 的 blk.0 排除（它的 FFN 真的讀這塊 buffer，
      跳過 ⇒ layer-0 吃 zeros、整網變垃圾還不報錯）；③ 連 `ggml_validate_row_data` 一起跳
      （否則 `--check-tensors` 會 throw invalid data）。

  P1  `CGC_POOL_MADVISE=1`
      fill 前對目標 slot 的頁 `madvise(MADV_DONTNEED)`：那塊馬上要被 pread 整塊覆寫，
      核心卻得先把它物化（若在 swap 上就是一次 swap-in 讀，讀進來下一刻就丟）。

  P2  `CGC_POOL_MADVISE=2`
      evict 時對該 slot 做同樣的事：內容本來就作廢，不該被寫出去。

⚠ P1/P2 共同的兩個陷阱（helper 存在的理由，寫錯就是**靜默毀掉隔壁 slot**）：
  ① `madvise` 要求頁對齊，否則 EINVAL；
  ② expert stride = 1.0703 MiB = 68.5 個 16 KiB 頁，**不是頁的整數倍** ⇒
     範圍不能往外取整（會清掉前後 slot 的頭尾）。只丟**完整落在範圍內**的頁。
  Metal 的 `pool_ext` 不是匿名 malloc，**永遠不碰**。

用法：
    python3 scripts/check/swap_p0p1p2_apply.py --self-test     # 離線自檢（不碰 repo）
    python3 scripts/check/swap_p0p1p2_apply.py --check          # 錨點漂移報告（只讀）
    python3 scripts/check/swap_p0p1p2_apply.py --apply          # 真的寫入（先自動 --check）
    python3 scripts/check/swap_p0p1p2_apply.py --revert         # 從 .bak 還原
"""
import argparse
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 備份放在 .workbuddy/（gitignored）：放在原始檔旁邊會變成 untracked 新檔，污染別人的 git status。
BAK_DIR = os.path.join(REPO, ".workbuddy", "patch_bak")
L = "src/llama.cpp/src/llama-model-loader.cpp"
LH = "src/llama.cpp/src/llama-model-loader.h"
E = "src/llama.cpp/src/llama-expert-cache.cpp"

# ─────────────────────────────────────────────────────────────────────────────
# P0 —— llama-model-loader.h
# ─────────────────────────────────────────────────────────────────────────────
H_INC = ("P0.a", LH,
         "#include <unordered_map>\n",
         "#include <unordered_map>\n#include <unordered_set>\n",
         "header: <unordered_set>")

H_MEMBER = ("P0.b", LH,
            "    bool expert_cache_skip_load = false;\n"
            "    std::vector<llama_expert_index_entry> expert_index;\n",
            "    bool expert_cache_skip_load = false;\n"
            "    std::vector<llama_expert_index_entry> expert_index;\n"
            "    // [CGC 2026-09-24 swap-miss P0] tensors the loader itself put on the CPU skip-load\n"
            "    // path (llama-model-loader.cpp: the `expert_cache_skip_load && _exps && blk.` branch,\n"
            "    // i.e. the `!buft` precondition already held). load_data_for must NOT read_raw these:\n"
            "    // the bytes come from the pool fill's pread and the CPU buffer is only a placeholder.\n"
            "    // Filled at tensor-creation time, read at load-data time (load_data_for is const).\n"
            "    std::unordered_set<std::string> skip_no_read;\n",
            "header: skip_no_read 成員")

H_DECL = ("P0.c", LH,
          "    // for backwards compatibility, does not support ggml-backend\n"
          "    void load_data_for(struct ggml_tensor * cur) const;\n",
          "    // for backwards compatibility, does not support ggml-backend\n"
          "    void load_data_for(struct ggml_tensor * cur) const;\n"
          "\n"
          "    // [CGC 2026-09-24 swap-miss P0] true => load_data_for must leave this tensor's buffer\n"
          "    // UNREAD (placeholder). See skip_no_read. Layer 0 under L4_SKIP_LAYER0 is excluded.\n"
          "    bool cgc_skip_readraw(const char * name) const;\n",
          "header: cgc_skip_readraw 宣告")

# ─────────────────────────────────────────────────────────────────────────────
# P0 —— llama-model-loader.cpp
# ─────────────────────────────────────────────────────────────────────────────
C_DEF = ("P0.d", L,
         "void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {\n",
         "// [CGC 2026-09-24 swap-miss P0] Skip the read_raw for CPU skip-load expert tensors.\n"
         "//\n"
         "// The 8 GiB pool + a fully-read 13.65 GB model is 21222 MiB on a 16384 MiB box: the\n"
         "// process is structurally over-subscribed and lives on swap (docs/SWAP_MISS_LINK §1).\n"
         "// ~10.9 GiB of that is expert weights that are read once here and never read again —\n"
         "// every byte that is ever needed is supplied by the pool fill's pread.\n"
         "//\n"
         "// Three guards, all of them load-bearing:\n"
         "//   (1) skip_no_read: only tensors the loader ITSELF placed on the skip-load path. The\n"
         "//       predicate at :1364 lives behind `!buft`, and an explicit tensor_buft_overrides\n"
         "//       entry can also put an expert on the CPU — re-deriving the rule from the name here\n"
         "//       would skip reads for tensors that were never meant to be skipped.\n"
         "//   (2) L4_SKIP_LAYER0's blk.0 is EXCLUDED: that tensor is deliberately demoted back to a\n"
         "//       plain CPU tensor (l4_kind = -1) and its FFN reads this buffer for real\n"
         "//       (llama-context.cpp:6437). Skipping it feeds zeros into layer 0 => the whole net\n"
         "//       outputs garbage with no error. Triggered by CGC_SERVER_SKIP0 (default 0).\n"
         "//   (3) CGC_EXPERT_SKIP_READRAW: default OFF, so landing this changes nothing until the\n"
         "//       A/B arm turns it on.\n"
         "bool llama_model_loader::cgc_skip_readraw(const char * name) const {\n"
         "    static const bool on = getenv(\"CGC_EXPERT_SKIP_READRAW\") != nullptr;\n"
         "    if (!on) {\n"
         "        return false;\n"
         "    }\n"
         "    if (skip_no_read.find(name) == skip_no_read.end()) {\n"
         "        return false; // not a loader-placed skip-load tensor (see guard 1)\n"
         "    }\n"
         "    // guard 2: layer 0 under L4_SKIP_LAYER0 reads its own buffer.\n"
         "    if (expert_cache_l4_skip_layer0 && strstr(name, \"blk.0.\") != nullptr) {\n"
         "        LLAMA_LOG_INFO(\"llama_model_loader: P0 keeps reading %s (L4_SKIP_LAYER0: its FFN \"\n"
         "                       \"reads this buffer directly)\\n\", name);\n"
         "        return false;\n"
         "    }\n"
         "    return true;\n"
         "}\n"
         "\n"
         "void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {\n",
         "cpp: cgc_skip_readraw 定義")

C_MARK = ("P0.e", L,
          "                buft = ggml_backend_cpu_buffer_type();\n"
          "                LLAMA_LOG_INFO(\"llama_model_loader: keeping %s out of GPU buffers (skip-load expert streaming)\\n\", t_meta->name);\n",
          "                buft = ggml_backend_cpu_buffer_type();\n"
          "                LLAMA_LOG_INFO(\"llama_model_loader: keeping %s out of GPU buffers (skip-load expert streaming)\\n\", t_meta->name);\n"
          "                // [CGC 2026-09-24 swap-miss P0] remember it: load_data_for will leave the\n"
          "                // buffer unread (placeholder). The layer-0 exclusion is applied there, not\n"
          "                // here, because l4_il is not in scope at this point.\n"
          "                skip_no_read.insert(t_meta->name);\n",
          "cpp: skip-load 分支記錄 tensor 名")

C_GUARD = ("P0.f", L,
           "        const auto & file = files.at(w.idx);\n"
           "        file->seek(w.offs, SEEK_SET);\n"
           "        file->read_raw(cur->data, ggml_nbytes(cur));\n",
           "        // [CGC 2026-09-24 swap-miss P0] placeholder expert tensor: return early so the\n"
           "        // ~10.9 GiB never becomes anonymous resident memory. Returning also skips the\n"
           "        // ggml_validate_row_data below — which MUST be skipped, since an unread buffer\n"
           "        // would otherwise throw \"invalid data\" under --check-tensors.\n"
           "        if (cgc_skip_readraw(ggml_get_name(cur))) {\n"
           "            return;\n"
           "        }\n"
           "        const auto & file = files.at(w.idx);\n"
           "        file->seek(w.offs, SEEK_SET);\n"
           "        file->read_raw(cur->data, ggml_nbytes(cur));\n",
           "cpp: read_raw 加 guard")

# [CGC 2026-09-24 11:1x] P0.g —— 開關判定從「存在即開」改成「取值判定」。
# 原寫法 `getenv(...) != nullptr` 讓 **任何值** 都把 P0 打開，包括 `=0`；
# 而把「顯式預設 0」寫進 launcher profile 是最自然的做法 ⇒ 那會讓 prod-new 靜默開著 P0 跑。
C_FIX = ("P0.g", L,
         "    static const bool on = getenv(\"CGC_EXPERT_SKIP_READRAW\") != nullptr;\n",
         "    // [CGC 2026-09-24] Value-based, NOT presence-based: a launcher profile that writes an\n"
         "    // explicit `CGC_EXPERT_SKIP_READRAW=0` default has to mean OFF. The first version\n"
         "    // tested `getenv() != nullptr`, which turns P0 ON for EVERY value including \"0\" --\n"
         "    // so wiring a documented default into prod-new would have silently enabled it.\n"
         "    static const bool on = []() -> bool {\n"
         "        const char * v = getenv(\"CGC_EXPERT_SKIP_READRAW\");\n"
         "        return v != nullptr && v[0] != '\\0' && v[0] != '0';\n"
         "    }();\n",
         "cpp: SKIP_READRAW 改取值判定（0/空 = 關）")

# ─────────────────────────────────────────────────────────────────────────────
# P1 / P2 —— llama-expert-cache.cpp
# ─────────────────────────────────────────────────────────────────────────────
E_INC = ("P12.a", E,
         "#include <sys/uio.h> // struct iovec / preadv (merge-read jobs)\n",
         "#include <sys/uio.h> // struct iovec / preadv (merge-read jobs)\n"
         "#include <sys/mman.h> // madvise/MADV_DONTNEED ([CGC 2026-09-24 swap-miss P1/P2])\n",
         "cpp: <sys/mman.h>")

E_FWD = ("P12.b", E,
         "static uint64_t make_key(uint32_t layer, uint32_t expert) {\n"
         "    return ((uint64_t) layer << 32) | expert;\n"
         "}\n",
         "static uint64_t make_key(uint32_t layer, uint32_t expert) {\n"
         "    return ((uint64_t) layer << 32) | expert;\n"
         "}\n"
         "\n"
         "// [CGC 2026-09-24 swap-miss P2] forward decls: both are defined after pool_region (they\n"
         "// need slots_l), but pick_slot — which sits above that — calls them. (Getting this wrong\n"
         "// is a compile error, and a compile error in this file blocks EVERY other line's build.)\n"
         "static int  cgc_pool_madvise_mode();\n"
         "static void cgc_discard_pool_slot(llama_expert_cache * cache, uint32_t layer, int32_t slot);\n",
         "cpp: P2 前向宣告")

E_HELP = ("P12.c", E,
          "        return cache->pool[layer][kind].data();\n"
          "    }\n"
          "    return nullptr;\n"
          "}\n",
          "        return cache->pool[layer][kind].data();\n"
          "    }\n"
          "    return nullptr;\n"
          "}\n"
          "\n"
          "// ─────────────────────────────────────────────────────────────────────────────\n"
          "// [CGC 2026-09-24 swap-miss P1/P2] Drop the physical pages behind pool bytes that are\n"
          "// about to be overwritten (P1) or are already garbage (P2).\n"
          "//\n"
          "// The 8 GiB pool is malloc'd ANONYMOUS memory. Once the box is over-subscribed\n"
          "// (model 13030 MiB + pool 8192 MiB = 21222 MiB on 16384 MiB) every one of those pages is\n"
          "// a swap candidate, and the kernel writes them out even when the very next instruction\n"
          "// overwrites them. madvise(MADV_DONTNEED) says \"just drop them\".\n"
          "//\n"
          "// ⚠ TWO hazards this helper exists for — both corrupt a NEIGHBOUR slot silently:\n"
          "//   (1) madvise() needs a PAGE-ALIGNED address (EINVAL otherwise);\n"
          "//   (2) an expert stride is 1.0703 MiB = 68.5 pages of 16 KiB, i.e. NOT a page multiple.\n"
          "//       Rounding the range OUTWARD would zero the tail of the previous slot and the head\n"
          "//       of the next one. So: round the start UP, the end DOWN, and drop only the pages\n"
          "//       FULLY INSIDE the range. (If the range holds no whole page, drop nothing.)\n"
          "// ⚠ Metal's pool_ext is NOT anonymous malloc — never pass those pointers here.\n"
          "// ─────────────────────────────────────────────────────────────────────────────\n"
          "static void cgc_discard_pages(uint8_t * base, size_t len) {\n"
          "    if (base == nullptr || len == 0) {\n"
          "        return;\n"
          "    }\n"
          "    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);\n"
          "    const uintptr_t b  = (uintptr_t) base;\n"
          "    const uintptr_t e  = b + len;\n"
          "    const uintptr_t sb = (b + page - 1) & ~(uintptr_t)(page - 1); // round UP\n"
          "    const uintptr_t se = e & ~(uintptr_t)(page - 1);              // round DOWN\n"
          "    if (se <= sb) {\n"
          "        return; // no page lies wholly inside the range: nothing may be dropped\n"
          "    }\n"
          "    ::madvise((void *) sb, (size_t)(se - sb), MADV_DONTNEED);\n"
          "}\n"
          "\n"
          "// Only the malloc'd pool may be discarded. pool_ext is a Metal allocation; discarding its\n"
          "// pages would pull the bytes out from under the GPU.\n"
          "static bool cgc_in_malloc_pool(const llama_expert_cache * cache, const uint8_t * p) {\n"
          "    for (size_t l = 0; l < cache->pool.size(); ++l) {\n"
          "        for (size_t k = 0; k < cache->pool[l].size(); ++k) {\n"
          "            const auto & v = cache->pool[l][k];\n"
          "            if (v.empty()) {\n"
          "                continue;\n"
          "            }\n"
          "            const uint8_t * base = v.data();\n"
          "            if (p >= base && p < base + v.size()) {\n"
          "                return true;\n"
          "            }\n"
          "        }\n"
          "    }\n"
          "    return false;\n"
          "}\n"
          "\n"
          "// CGC_POOL_MADVISE: unset/0 = off (byte-identical to the old path); 1 = P1 only;\n"
          "// 2 = P1 + P2. Kept env-gated so this can land in a shared file without changing any\n"
          "// running measurement.\n"
          "static int cgc_pool_madvise_mode() {\n"
          "    static const int mode = []() -> int {\n"
          "        const char * v = getenv(\"CGC_POOL_MADVISE\");\n"
          "        if (v == nullptr || v[0] == '\\0' || v[0] == '0') { return 0; }\n"
          "        return (v[0] == '2') ? 2 : 1;\n"
          "    }();\n"
          "    return mode;\n"
          "}\n"
          "\n"
          "static void cgc_discard_pool_slot(llama_expert_cache * cache, uint32_t layer, int32_t slot) {\n"
          "    if (slot < 0 || layer >= cache->pool.size()) {\n"
          "        return;\n"
          "    }\n"
          "    const uint32_t nslots = slots_l(cache, layer);\n"
          "    if (nslots == 0) {\n"
          "        return;\n"
          "    }\n"
          "    for (size_t k = 0; k < cache->pool[layer].size(); ++k) {\n"
          "        auto & v = cache->pool[layer][k];\n"
          "        if (v.empty()) {\n"
          "            continue;\n"
          "        }\n"
          "        const size_t stride = v.size() / nslots;\n"
          "        cgc_discard_pages(v.data() + (size_t) slot * stride, stride);\n"
          "    }\n"
          "}\n",
          "cpp: madvise helper 群")

E_P1 = ("P1.d", E,
        "    const size_t n = segs.size();\n"
        "    ok.assign(n, 0);\n"
        "    if (n == 0) {\n"
        "        return;\n"
        "    }\n",
        "    const size_t n = segs.size();\n"
        "    ok.assign(n, 0);\n"
        "    if (n == 0) {\n"
        "        return;\n"
        "    }\n"
        "    // [CGC 2026-09-24 swap-miss P1] Every dst below is about to be overwritten in full by\n"
        "    // pread. Without this, a page that happens to sit on swap must first be read back in\n"
        "    // (swap-in) only to be discarded one instruction later — pure, repeated IO. Drop the\n"
        "    // pages first. Interior-pages-only + malloc-pool-only: see cgc_discard_pages.\n"
        "    if (cgc_pool_madvise_mode() >= 1) {\n"
        "        for (size_t i = 0; i < n; ++i) {\n"
        "            if (cgc_in_malloc_pool(cache, dsts[i])) {\n"
        "                cgc_discard_pages(dsts[i], segs[i].bytes);\n"
        "            }\n"
        "        }\n"
        "    }\n",
        "cpp: fill 前 discard（P1）")

E_P2 = ("P2.e", E,
        "            const int32_t evicted = owner[best_slot];\n"
        "            if (evicted >= 0) {\n"
        "                cache->n_evictions++; // [CGC miss attribution] a resident expert is losing its slot\n"
        "            }\n",
        "            const int32_t evicted = owner[best_slot];\n"
        "            if (evicted >= 0) {\n"
        "                cache->n_evictions++; // [CGC miss attribution] a resident expert is losing its slot\n"
        "                // [CGC 2026-09-24 swap-miss P2] the slot's bytes are dead from here on: keep\n"
        "                // them from ever being written out to swap. (Interior pages only, malloc pool\n"
        "                // only — the two partial pages at the ends belong to the neighbouring slots.)\n"
        "                if (cgc_pool_madvise_mode() >= 2) {\n"
        "                    cgc_discard_pool_slot(cache, layer, best_slot);\n"
        "                }\n"
        "            }\n",
        "cpp: evict 時 discard（P2）")

HUNKS = [H_INC, H_MEMBER, H_DECL, C_DEF, C_MARK, C_GUARD, C_FIX, E_INC, E_FWD, E_HELP, E_P1, E_P2]

# 「已套用」的判據不能用 anchor 本身：P2.e 是把程式碼插進 anchor 中間，
# 套用後 anchor 不再連續出現（但那不代表沒套用）。所以每個 hunk 自備一個 marker
# —— 只存在於新增程式碼裡的獨特字串。
MARKER = {
    "P0.a":  "#include <unordered_set>",
    "P0.b":  "std::unordered_set<std::string> skip_no_read;",
    "P0.c":  "bool cgc_skip_readraw(const char * name) const;",
    "P0.d":  "bool llama_model_loader::cgc_skip_readraw",
    "P0.e":  "skip_no_read.insert(t_meta->name);",
    "P0.f":  "if (cgc_skip_readraw(ggml_get_name(cur))) {",
    # P0.g 是「改寫」型 hunk（replacement 不含 anchor），所以 marker 必須是只出現在新碼裡的字串
    "P0.g":  "v[0] != '\\0' && v[0] != '0'",
    "P12.a": "#include <sys/mman.h>",
    "P12.b": "static void cgc_discard_pool_slot(llama_expert_cache * cache, uint32_t layer, int32_t slot);",
    "P12.c": "static void cgc_discard_pages(uint8_t * base, size_t len) {",
    "P1.d":  "if (cgc_pool_madvise_mode() >= 1) {",
    "P2.e":  "cgc_discard_pool_slot(cache, layer, best_slot);",
}
assert {h[0] for h in HUNKS} == set(MARKER), "marker 表與 hunk 表不一致"


def read(rel):
    with io.open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def check(verbose=True):
    """錨點漂移報告。回傳 (ok, rows)；ok=False 代表至少一個 hunk 不能安全套用。"""
    ok = True
    rows = []
    for name, rel, anchor, repl, descr in HUNKS:
        s = read(rel)
        na, nr = s.count(anchor), s.count(MARKER[name])
        # ⚠ 順序要緊：marker 存在就代表已套用。anchor 在套用後通常**仍然存在**
        #   （replacement 含 anchor），所以先判 anchor 會把已套用誤報成 BOTH。
        if nr >= 1:
            state, good = "ALREADY", True
        elif na == 1:
            state, good = "APPLY", True
        else:
            state, good = "DRIFT(錨點命中 %d)" % na, False
        ok = ok and good
        rows.append((name, rel, state, descr))
        if verbose:
            print("  %-6s %-8s %-38s %s" % (name, state, descr, rel.split("/")[-1]))
    return ok, rows


def apply(force=False):
    ok, _ = check(verbose=False)
    if not ok and not force:
        print("FAIL: 有 hunk 的錨點不唯一或已套用（先跑 --check）；整批未寫入。")
        return 1
    by_file = {}
    for name, rel, anchor, repl, descr in HUNKS:
        by_file.setdefault(rel, []).append((name, anchor, repl, MARKER[name]))
    for rel, hs in by_file.items():
        p = os.path.join(REPO, rel)
        s = read(rel)
        bak = os.path.join(BAK_DIR, os.path.basename(rel))
        if not os.path.exists(bak):
            os.makedirs(BAK_DIR, exist_ok=True)
            with io.open(bak, "w", encoding="utf-8") as f:
                f.write(s)
        for name, anchor, repl, marker in hs:
            if s.count(anchor) == 1 and s.count(marker) == 0:
                s = s.replace(anchor, repl)
                print("  applied %-6s %s" % (name, rel.split("/")[-1]))
            else:
                print("  skipped %-6s %s (already/anchor missing)" % (name, rel.split("/")[-1]))
        with io.open(p, "w", encoding="utf-8") as f:
            f.write(s)
    print("done. 還原： python3 %s --revert" % os.path.basename(__file__))
    return 0


def revert():
    n = 0
    for rel in sorted({h[1] for h in HUNKS}):
        p = os.path.join(REPO, rel)
        bak = os.path.join(BAK_DIR, os.path.basename(rel))
        if os.path.exists(bak):
            with io.open(bak, encoding="utf-8") as f:
                s = f.read()
            with io.open(p, "w", encoding="utf-8") as f:
                f.write(s)
            os.remove(bak)
            n += 1
            print("  reverted %s" % rel)
    print("reverted %d file(s)" % n)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# self-test：不碰 repo。把 C++ 的兩個判斷邏輯用 Python 重寫一遍並驗證。
# ─────────────────────────────────────────────────────────────────────────────
def _discard_range(base, length, page):
    """cgc_discard_pages 的對齊決策（回傳 (start,end) 或 None）。"""
    if length == 0:
        return None
    sb = (base + page - 1) & ~(page - 1)
    se = (base + length) & ~(page - 1)
    if se <= sb:
        return None
    return (sb, se)


def _skip_readraw(skip_no_read, name, l4_skip_layer0):
    """cgc_skip_readraw 的三個 guard（env 已開啟的前提下）。"""
    if name not in skip_no_read:
        return False
    if l4_skip_layer0 and "blk.0." in name:
        return False
    return True


def self_test():
    p = f = 0
    PAGE = 16384
    STRIDE = 1122304  # 1.0703 MiB

    # 1) 頁對齊：丟棄範圍必須頁對齊
    r = _discard_range(0, STRIDE, PAGE)
    p += 1 if r and r[0] % PAGE == 0 and r[1] % PAGE == 0 else 0
    f += 0 if (r and r[0] % PAGE == 0 and r[1] % PAGE == 0) else 1
    print("  %s 丟棄範圍頁對齊" % ("ok  " if r and r[0] % PAGE == 0 else "FAIL"))

    # 2) 不越界：slot k 的丟棄範圍必須完全落在 slot k 之內（不能碰到 k-1 / k+1）
    bad = 0
    for k in range(6):
        base = k * STRIDE
        r = _discard_range(base, STRIDE, PAGE)
        if r is None:
            continue
        if r[0] < base or r[1] > base + STRIDE:
            bad += 1
    p, f = (p + 1, f) if bad == 0 else (p, f + 1)
    print("  %s 不越界（6 個連續 slot，越界數=%d）" % ("ok  " if bad == 0 else "FAIL", bad))

    # 3) 真的有丟到東西：68.5 頁的 stride 至少要丟到 67 頁
    r = _discard_range(0, STRIDE, PAGE)
    npages = (r[1] - r[0]) // PAGE if r else 0
    good = npages >= 67
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s 丟棄量足夠（%d 頁 / stride %.1f 頁）" % ("ok  " if good else "FAIL", npages, STRIDE / PAGE))

    # 4) 小於一頁的範圍：什麼都不丟（不是往外取整）
    r = _discard_range(PAGE - 8, 16, PAGE)
    good = r is None
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s 跨頁的小範圍不丟（避免清到隔壁）" % ("ok  " if good else "FAIL"))

    # 5) 剛好頁對齊的範圍：整段丟
    r = _discard_range(0, PAGE * 4, PAGE)
    good = r == (0, PAGE * 4)
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s 頁對齊範圍整段丟" % ("ok  " if good else "FAIL"))

    # 6) P0 guard 1：不在 skip_no_read 裡的 tensor 不跳（override 到 CPU 的 expert）
    good = _skip_readraw({"blk.5.ffn_down_exps.weight"}, "blk.5.ffn_gate_up_exps.weight", False) is False
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s guard1 非 skip-load tensor 照讀" % ("ok  " if good else "FAIL"))

    # 7) P0 guard 2：L4_SKIP_LAYER0 下的 blk.0 必須照讀（否則 layer-0 吃 zeros）
    good = _skip_readraw({"blk.0.ffn_down_exps.weight"}, "blk.0.ffn_down_exps.weight", True) is False
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s guard2 L4_SKIP_LAYER0 的 blk.0 照讀" % ("ok  " if good else "FAIL"))

    # 8) 同一個名字在 SKIP0 關閉時照跳（不是無條件排除 layer 0）
    good = _skip_readraw({"blk.0.ffn_down_exps.weight"}, "blk.0.ffn_down_exps.weight", False) is True
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s guard2 只在 SKIP0 開啟時排除 blk.0" % ("ok  " if good else "FAIL"))

    # 9) 一般 skip-load expert：跳
    good = _skip_readraw({"blk.7.ffn_up_exps.weight"}, "blk.7.ffn_up_exps.weight", False) is True
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s 一般 skip-load expert 跳過 read_raw" % ("ok  " if good else "FAIL"))

    # 10) 每個 hunk 都要有 marker，且 marker 必須只存在於新增程式碼（anchor 裡沒有）
    bad = [h[0] for h in HUNKS if MARKER[h[0]] not in h[3] or MARKER[h[0]] in h[2]]
    good = not bad
    p, f = (p + 1, f) if good else (p, f + 1)
    print("  %s marker 只存在於新增程式碼%s" % ("ok  " if good else "FAIL", "" if good else " " + str(bad)))

    # 11) 冪等性：套用一次後 marker 命中 1 次；再套用會被 check 判為 ALREADY 而不是重複插入
    ok_all = True
    for name, rel, anchor, repl, descr in HUNKS:
        s = "X" + anchor + "Y"
        s2 = s.replace(anchor, repl)
        if s2.count(MARKER[name]) != 1:
            ok_all = False
        s3 = s2.replace(anchor, repl)  # 再套一次
        if s3.count(MARKER[name]) != 2 and s3.count(anchor) == 1:
            ok_all = False  # 沒被偵測到 ⇒ 會重複插入
    p, f = (p + 1, f) if ok_all else (p, f + 1)
    print("  %s 冪等（重複套用可被偵測）" % ("ok  " if ok_all else "FAIL"))

    print("\nself-test: %d passed, %d failed" % (p, f))
    return 0 if f == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="P0/P1/P2 定點補丁器")
    ap.add_argument("--check", action="store_true", help="錨點漂移報告（只讀）")
    ap.add_argument("--apply", action="store_true", help="寫入（先自動 --check）")
    ap.add_argument("--revert", action="store_true", help="從 .bak 還原")
    ap.add_argument("--self-test", action="store_true", help="離線自檢")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.revert:
        return revert()
    if a.apply:
        return apply()
    ok, _ = check()
    print("\n%s" % ("全部錨點命中唯一，可 --apply" if ok else "有 hunk 不能安全套用，見上"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
