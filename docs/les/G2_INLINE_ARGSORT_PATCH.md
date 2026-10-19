# G2 施工补丁：圖內 argsort 預測 + 在線命中率對賬（READ=0）

> Charter：`scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml`（G2，行 108–111）
> 上游：`docs/les/upstream/qwen35moe.cpp:455-466`（MiniCPM-V 4.7 LES，SHA `88bd37eb`）
> 狀態：**施工圖紙（逐字可 apply）**。commit2 探針落地（乾淨窗口 replay 通過）＋清理 7 個 affinity 工作區 hunk 後，一次性 apply 三檔 → 編譯 → 乾淨窗口跑 G2 驗收。**在那之前不得改 src**（保持 commit2 的 index 邊界）。

---

## 0. 一句話現狀

commit2 探針**已經把 G2 最難的那一步算好了**：`qwen35moe.cpp` 裡
`les_logits = build_lora_mm(model.layers[il+1].ffn_gate_inp, attn_post_norm)`
（＝上游 `prefetch_router_logits`），位置在 `attn_post_norm` 後、`build_layer_ffn` 前，與上游完全同構。

G2 只補兩件事：
1. **圖內**對 `les_logits` 跑 `ggml_argsort_top_k`（Metal/CPU 真跑、出 I32 專家 id），不是 host numpy；
2. **在線**把預測 top-k 與 A1 真實路由（`CGC-A1-RAW` 的 `ids`）按 `(pass, target=il+1)` 對拍，實時輸出 `n_hits/n_armed`。

**G2 不讀盤**（`LLAMA_MOE_PREFETCH_READ` 預設 0）、不寫 slot、不參與專家選擇 ⇒ 逐字節一致。

---

## 1. 前置閘門（未過不 apply）

- [ ] commit2 六檔已 commit（`bash ~/les_commit2_run.sh` 乾淨窗口 replay PASS），HEAD 已含探針。
- [ ] 工作區 7 個 expert-affinity 功能 hunk（`llama-context.cpp` 舊行 5394/9017/9026/9307/9324/9548/9726）已撤（已判死 −0.87%），得到「純探針」工作區。
- [ ] 純探針版已單獨編譯一次 `BUILD_EXIT=0`（分離驗證 commit2，避免 G2 與它綁在一起）。
- [ ] 乾淨 GPU 窗口：重開機 `swap used < 1024 MiB`、thermal nominal、退 WorkBuddy/Chrome/FreeBuff。

---

## 2. 三檔改動（全部 env-gated，預設 OFF ⇒ 節點不存在 ⇒ byte-identical）

### 2.1 `src/llama.cpp/src/models/qwen35moe.cpp` —— 圖內 argsort 節點

**定位**：現有 LES 探針匿名塊（`cb(attn_post_norm,...)` 之後、`[CGC rho probe] 舊位置` 之前）。

**替換前（commit2 探針現狀）**：
```cpp
        {
            static const bool cgc_les_on = getenv("CGC_LES_PROBE") != nullptr
                                         && getenv("CGC_LES_PROBE")[0] != '0';
            if (cgc_les_on && ubatch.n_tokens == 1 &&
                il + 1 < n_layer &&
                model.layers[il + 1].ffn_gate_inp != nullptr) {
                ggml_tensor * les_logits = build_lora_mm(model.layers[il + 1].ffn_gate_inp, attn_post_norm);
                cb(les_logits, "cgc_les_h", il);
                ggml_build_forward_expand(gf, les_logits);
            }
        }
```

**替換後（G2：logits 閘不變，新增 argsort 頂點，閘＝LLAMA_MOE_LOOKAHEAD_PREFETCH）**：
```cpp
        {
            static const bool cgc_les_on = getenv("CGC_LES_PROBE") != nullptr
                                         && getenv("CGC_LES_PROBE")[0] != '0';
            // [G2] 圖內預測構圖（charter e-les-router-prefetch G2）。
            //   LLAMA_MOE_LOOKAHEAD_PREFETCH=1 建「下一層 router 預測」頂點；
            //   G2 不設 LLAMA_MOE_PREFETCH_READ（=0）：只預測、不 preadv 讀盤（G3 才讀）。
            //   深度 LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH，預設 8（G1 recall@d8=0.79）。
            static const bool cgc_la_topk =
                getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr &&
                getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH")[0] != '0';
            static int cgc_la_depth = -1;
            if (cgc_la_topk && cgc_la_depth < 0) {
                const char * d = getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH");
                cgc_la_depth = (d != nullptr && d[0] != '\0') ? atoi(d) : 8;
                if (cgc_la_depth < 1)   cgc_la_depth = 1;
                if (cgc_la_depth > 256) cgc_la_depth = 256;
            }
            if ((cgc_les_on || cgc_la_topk) && ubatch.n_tokens == 1 &&
                il + 1 < n_layer &&
                model.layers[il + 1].ffn_gate_inp != nullptr) {
                ggml_tensor * les_logits = build_lora_mm(model.layers[il + 1].ffn_gate_inp, attn_post_norm);
                if (cgc_les_on) {                       // G1 離線 sidecar：256 維 logits（行為不變）
                    cb(les_logits, "cgc_les_h", il);
                    ggml_build_forward_expand(gf, les_logits);
                }
                if (cgc_la_topk) {                      // G2 圖內 argsort：I32 專家 id
                    // expand(topk) 會自動把 cont<-argsort<-les_logits 依賴鏈帶進圖，
                    // 故只開 topk 不開 cgc_les_on 時 logits 仍會被算（只是不 capture）。
                    ggml_tensor * les_topk = ggml_cont(ctx0,
                        ggml_argsort_top_k(ctx0, les_logits, cgc_la_depth));
                    cb(les_topk, "cgc_les_topk", il);
                    ggml_build_forward_expand(gf, les_topk);
                }
            }
        }
```

要點：
- `ggml_argsort_top_k` 返回 **I32 排序索引**（`ggml.c:5364` `ggml_new_tensor(GGML_TYPE_I32)`，DESC argsort + view 前 k；Metal kernel 齊全 `ggml-metal-ops.cpp:7818`）。輸出 shape `[depth,1]`，值即專家 id（0..255）。
- `ubatch.n_tokens==1` 保證只有 decode 單 token 建頂點；prefill 與 MTP verify 多 token（`n_tokens>1`）自動降級不預測（G5 行為同源）。
- 預設兩閘皆 OFF：`cgc_les_h` 與 `cgc_les_topk` 頂點都不存在 ⇒ 圖與基線逐位元相同。

---

### 2.2 `src/llama.cpp/ggml/src/ggml-backend.cpp` —— split 圖白名單放行

**定位**：async split 回調裡 force-forward 探針頂點的名字匹配（現約 2744–2763 行）。

**(a) 閘聚合**（現 `static const bool cgc_les_probe = ...` 那行附近）：
```cpp
                    static const bool cgc_td_cb     = getenv("CGC_TD_CB")     != nullptr;
                    static const bool cgc_rho_probe = getenv("CGC_RHO_PROBE") != nullptr;
                    static const bool cgc_les_probe = getenv("CGC_LES_PROBE") != nullptr;
                    static const bool cgc_la_topk   = getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr;
                    if (cgc_td_cb || cgc_rho_probe || cgc_les_probe || cgc_la_topk) {
```

**(b) 名字匹配加一支**：
```cpp
                            } else if (strncmp(tn->name, "cgc_rho_logits", 14) == 0 ||
                                       strncmp(tn->name, "cgc_bhead_h",   11) == 0 ||
                                       strncmp(tn->name, "cgc_les_h",      9) == 0 ||
                                       strncmp(tn->name, "cgc_les_topk",  12) == 0) {
```

> prod-new（20 t/s 單提交路徑）走 direct-walk（見 2.3c），split 路徑只需保證頂點不被剪掉；capture 統一在 direct-walk 做，避免雙路徑重複計數。

---

### 2.3 `src/llama.cpp/src/llama-context.cpp` —— topk capture + 真實路由 ring + 在線交集

> include 現狀（已核實）：`<atomic>` 已在第 4 行、`<unordered_map>` 已在第 37 行；**缺 `<mutex>` 與 `<array>`**，在第 37 行 include 區補這兩行（頂部、勿插在函數間）：
> ```cpp
> #include <mutex>
> #include <array>
> ```

**(a) 前向宣告**（現 `static void cgc_les_capture(ggml_tensor * t);` 約 3606 行下方）：
```cpp
static void cgc_les_gt_record(int pass, int il, const int32_t * ids, int k);
static void cgc_les_topk_capture(ggml_tensor * t);
```

**(b) 真實路由 ring + 預測對賬（緊貼 `cgc_les_capture` 定義之前/之後放，檔內 static）**：
```cpp
// [G2] 在線對賬：A1 dispatch 回調把「該層真實 top8 專家 id」寫進 ring；decode 圖跑完、
// synchronize 後的 direct-walk 讀 cgc_les_topk-<il>（對 target=il+1 的預測），與 ring 對拍。
// 整個 decode step 圖跑完後，target 層的真實路由必已在 ring（預測在 il 層、真實在 il+1 層，
// walk 在全圖之後），故直接查 (pass,target) 即可。閘＝LLAMA_MOE_LOOKAHEAD_PREFETCH，OFF=零成本。
static std::mutex g_les_gt_mu;
static std::unordered_map<uint64_t, std::array<int32_t, 8> > g_les_gt;  // key=(pass<<16)|(il&0xffff)
static long g_les_hits = 0, g_les_armed = 0;
static int  g_les_rep_il = -1;   // 每個 step 末（最大 il）打印一次累計

// 注意：A1 塊的 ids 是 const int32_t*（llama-context.cpp:7729），專家 id 0..255 正負無歧義。
static void cgc_les_gt_record(int pass, int il, const int32_t * ids, int k) {
    if (k > 8) k = 8;
    std::array<int32_t, 8> gt{};
    for (int i = 0; i < k; ++i) { gt[i] = ids[i]; }
    const uint64_t key = ((uint64_t)(uint32_t) pass << 16) | (uint16_t) il;
    bool trim = false;
    {
        std::lock_guard<std::mutex> lk(g_les_gt_mu);
        g_les_gt[key] = gt;
        if (g_les_gt.size() > 768) trim = true;   // ~3 pass * 256，週期性清舊 pass
    }
    if (trim) {
        const int keep = pass - 2;
        std::lock_guard<std::mutex> lk(g_les_gt_mu);
        for (auto it = g_les_gt.begin(); it != g_les_gt.end();) {
            const int p = (int)(it->first >> 16);
            if (p < keep) it = g_les_gt.erase(it); else ++it;
        }
    }
}

static void cgc_les_topk_capture(ggml_tensor * t) {
    if (t->type != GGML_TYPE_I32 || t->ne[1] != 1 || t->ne[0] <= 0 || t->ne[0] > 256) {
        return;
    }
    const int depth = (int) t->ne[0];
    int32_t pred[256];
    ggml_backend_tensor_get(t, pred, 0, (size_t) depth * sizeof(int32_t));
    const char * dash = strrchr(t->name, '-');
    const int il     = dash ? atoi(dash + 1) : -1;   // 物理層；target = il+1
    const int target = il + 1;
    const int pass   = g_cgc_bhead_pass.load();
    if (target < 4) return;                           // 【G1 裁定口徑】豁免前 3 層
    std::array<int32_t, 8> gt{};
    bool have = false;
    {
        std::lock_guard<std::mutex> lk(g_les_gt_mu);
        auto it = g_les_gt.find(((uint64_t)(uint32_t) pass << 16) | (uint16_t) target);
        if (it != g_les_gt.end()) { gt = it->second; have = true; }
    }
    if (!have) return;
    int hits = 0;
    for (int d = 0; d < depth; ++d) {
        for (int g = 0; g < 8; ++g) {
            if (pred[d] == gt[g]) { ++hits; break; }
        }
    }
    g_les_hits += hits;
    g_les_armed += depth;
    if (il > g_les_rep_il) {                          // 每 step 走到最大 il 時打印累計
        g_les_rep_il = il;
        const double hr = (g_les_armed > 0) ? (double) g_les_hits / (double) g_les_armed : 0.0;
        if (il >= 37) {
            fprintf(stderr, "CGC-LES-G2: pass=%d n_hits=%ld n_armed=%ld hitrate=%.4f (target>=4, depth=%d)\n",
                    pass, g_les_hits, g_les_armed, hr, depth);
        }
    }
}
```
> 註：`g_les_rep_il` 每個 decode step 需在 il 回繞時重置；最穩妥是在 A1 回調偵測到 `il <= s_a1_prev_il`（pass 遞增）時把 `g_les_rep_il = -1`（見 2.3d，已一併處理）。`il>=37` 打印條件依真實最大可預取層（n_layer−2）調整；若不想依賴層數，可改為「每 64 step 打印一次」。

**(c) direct-walk 同步點加 topk 遍歷**（現約 5703 的 `[CGC LES] ... walk gf ... cgc_les_h` 塊）：
```cpp
    static const bool cgc_les_s2c_on = getenv("CGC_LES_PROBE") != nullptr
                                    && getenv("CGC_LES_PROBE")[0] != '0';
    static const bool cgc_la_s2c_on  = getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr
                                    && getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH")[0] != '0';
    if (cgc_les_s2c_on || cgc_la_s2c_on) {
        ggml_backend_sched_synchronize(sched.get());
        const int les_n_nodes = ggml_graph_n_nodes(gf);
        for (int les_i = 0; les_i < les_n_nodes; ++les_i) {
            ggml_tensor * les_t = ggml_graph_node(gf, les_i);
            if (les_t == nullptr || les_t->data == nullptr) { continue; }
            if (cgc_les_s2c_on && strncmp(les_t->name, "cgc_les_h-", 10) == 0) {
                if (cgc_is_decode_graph((int64_t) les_t->ne[1], cgc_decode_max_tokens)) {
                    cgc_les_capture(les_t);
                }
            }
            if (cgc_la_s2c_on && strncmp(les_t->name, "cgc_les_topk-", 13) == 0) {
                if (cgc_is_decode_graph((int64_t) les_t->ne[1], cgc_decode_max_tokens)) {
                    cgc_les_topk_capture(les_t);
                }
            }
        }
    }
```

**(d) A1 真實路由回調寫 ring**（現約 8393 的 `CGC-A1-RAW` 塊，在 `if (il <= s_a1_prev_il) { ++s_a1_pass; }` 之後加重置；在每個 j 的 `ids` 迴圈後寫 ring）：
```cpp
            if (il <= s_a1_prev_il) { ++s_a1_pass; g_les_rep_il = -1; }
            s_a1_prev_il = il;
```
以及在該 j 迴圈（`fprintf(stderr, "\n");` 之後）加：
```cpp
                // [G2] 供 LES 在線對賬：記本層真實 top8（key 在 cgc_les_topk_capture 用 target=il 查）。
                static const bool g_les_gt_on =
                    getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr &&
                    getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH")[0] != '0';
                if (g_les_gt_on) {
                    cgc_les_gt_record(s_a1_pass, il, ids + j * n_expert_used, (int) n_expert_used);
                }
```

> 對齊口徑：預測 `cgc_les_topk-<il>` 用 `attn_post_norm(il)` 打 `W_gate[il+1]`，target=il+1；ring 以**真實層 il** 為 key，故查 `(pass, il+1)`。A1 對「混合層/SSM 不路由層」不發 `CGC-A1-RAW`（`n_expert_used>0` 才進），這些 target 在 ring 缺失時 `cgc_les_topk_capture` 直接 `return`（不計入分母），與 G1 離線「只統計可預取路由層」一致。

---

## 3. 編譯（commit2 落地、affinity 清理後）

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
export PATH="/opt/homebrew/bin:$PATH"
cmake --build src/llama.cpp/build --parallel 8
# 預設 OFF 下先自測：全關 env 啟動，stderr 不得出現 cgc_les_topk / CGC-LES-G2
```

---

## 4. G2 驗收（乾淨窗口，harness 唯一入口）

對照臂（同 prompt、warm-skip 64、seed 1、steady ≥128；三套 prompt 同 G1）：

| arm | env（prod-new arm 字串） | 目的 |
|---|---|---|
| OFF（基線）| （無 CGC/LLAMA_MOE 變量） | 圖無預測頂點，作 bit-exact 與 t/s 基線 |
| ON（G2）| `CGC_A1_HIST_PROBE=1;CGC_LES_PROBE=1;LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_PREFETCH_READ=0` | 建 logits+argsort、不讀盤，輸出在線命中率 |

```bash
# 每套 prompt（code/zh_chat/math）各跑 OFF/ON，warm-skip 64
python3 scripts/check/harness.py bench \
  --arm "prod-new:CGC_A1_HIST_PROBE=1;CGC_LES_PROBE=1;LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_PREFETCH_READ=0" \
  --charter scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml \
  --prompt-file scripts/prompts/les_g1_code.txt \
  --workdir /tmp/les_g2_code --json /tmp/les_g2_code.json
```

**fail-closed 判據（charter 108–111，全過才進 G3）**：
1. **預設 OFF bit-identical**：OFF arm 過 cell contract M1/M2/M3，coverage 100%、comparable=true（222/222 或該 cell 既有口徑）。
2. **在線 vs G1 離線命中率差 ≤1%**：ON arm stderr 累計 `CGC-LES-G2 hitrate`（target≥4）對比 `Backup/les_g1_hitrate_20261017/hitrate_*.json` 的 mean（0.8197/0.8205/0.8207），絕對差 ≤0.01。
3. **構圖開銷 decode t/s 退化 <2%**：ON vs OFF 同場 warm-skip64 配對 decode 中位數下降 <2%（構圖白送，否則 G3 重疊也補不回）。
4. （附加自證）Metal argsort 正確：離線對 `cgc_les_h` logits 用 numpy argsort 的 top-8 與在線 `cgc_les_topk` 逐筆一致（tie 除外）；可用 `docs/les/tools/les_hitrate.py` 擴充 `--topk-sidecar` 核對（G2 若需要可先只記 stderr 命中率，sidecar topk dump 列為可選）。

---

## 5. 風險 / 回滾

- **風險**：`ggml_argsort` 在 Metal 上對 `[256,1]` 走 argsort+merge pipeline（`ggml-metal-ops.cpp:7818/7880`），需確認單 token 下不 fallback CPU（若 fallback，構圖開銷會超 2%，驗收判據 3 會抓出來）。
- 預測頂點在 attn 之後、FFN 之前，**不回饋任何引擎狀態**（不寫 slot、不碰 `mul_mat_id`），最壞情況只是多算一個頂點被 capture，數學輸出不變。
- 回滾：三檔改動皆在獨立 env 閘內，`unset LLAMA_MOE_LOOKAHEAD_PREFETCH CGC_LES_PROBE CGC_A1_HIST_PROBE` 即回基線；`git checkout -- <三檔>` 移除補丁。
- G2 通過後才進 G3（後台 preadv worker + ping-pong，屆時 `LLAMA_MOE_PREFETCH_READ=1` 才把 `cgc_les_topk` 接到 fill 佇列）。

---

## 6. 與 G1 數字的接口（不可改）

- G1 權威結果：`Backup/les_g1_hitrate_20261017/hitrate_{code,zh_chat,math}.json`（verdict=PASS，target≥4 mean 0.8197/0.8205/0.8207；recall@d8=0.79、d10=0.86、d12=0.90）。
- 本補丁只重現「在 G1 同一 logits 上的 argsort 選擇」，不改預測器、不入口徑；任何與 G1 偏差 >1% 都表示圖內 argsort/對齊有 bug，而不是命中率變了。
