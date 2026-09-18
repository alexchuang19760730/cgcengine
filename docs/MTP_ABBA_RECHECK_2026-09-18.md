# 12.62 的重驗證：ABBA 四臂，llama-bench 口徑

2026-09-18 18:01–18:06 · 運營層（本線）· 機器狀態：8080 空、無 `llama-server`、無量測驅動
· 熱壓 launch `NOMINAL`、`Pages free` 0.11 → 3.89 GB、`swap used` 12178／13312 MB

## §0 為什麼要重驗

使用者的問題：「12.62（MTP on、nb-aware 修正後）重驗證一下，找出跟目前我們最終落地的差異，
看看是**環境**還是**本質**問題。」

背景（`.workbuddy/memory/MEMORY_PERF.md:223-240`）：decode 速度有**四個互不相容的定義**，
而 12.62 屬 **HTTP／`llama-speculative-simple`** 口徑；落地用的是 **llama-bench** 口徑
（09-17 使用者裁定的唯一 instrument of record）的 **10.78–10.98**。兩者不可直接比。

今天的 `2fcf616b9` 才把 MTP 接進 llama-bench（真因是 `test_gen_spec` 的 `n_past` 一行初始化），
所以**現在第一次可以在同一個口徑下做 MTP on/off**。

## §1 條件（照抄可重跑）

```
python3 scripts/check/llama_bench_matrix.py \
  --arms prod25-stream --prompt 0 --gen 128 --depths 512 --reps 3 \
  [--spec-type draft-mtp --spec-draft-n-max 3]
```

`--expert-cache 8589934592`（**8 GiB 生產 pool**）、`-b 512 -ub 512`、`-ngl 99`、
模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`。四臂順序 **off → on → on → off**（ABBA）。
產物 `/tmp/abba{1..4}_*.json`。臂間無人工等待（見 §4 的後果）。

## §2 結果

| 臂 | 順序 | spec | t/s | ± | thermal hist | **requests** | **miss** | miss 率 | hit% | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | off | 7.89 | 2.27 | NOM 59 / MOD 75 | **128,920** | 4,500 | 3.49% | 96.5 | 70.2 s |
| 2 | 2 | draft-mtp | 7.39 | 3.68 | MOD 97 / HEA 31 | 164,747 | 9,247 | 5.61% | 94.4 | 82.4 s |
| 3 | 3 | draft-mtp | 5.81 | 1.01 | HEA 128 | 157,006 | 9,279 | 5.91% | 94.1 | 82.0 s |
| 4 | 4 | off | 9.62 | 1.11 | HEA 107 | **128,920** | 4,201 | 3.36% | 96.7 | 54.2 s |

## §3 裁決：哪些可以引用，哪些不行

**不可引用：四臂的 t/s。** 四個 `thermal_hist` **互不相同**（NOMINAL → MODERATE → HEAVY → HEAVY），
依本 repo 的既有判準（**`t/s` 一律不引用；計數器與 digest 不受熱影響，有效**）⇒ 這四格是熱的差，
不是 treatment 的差。

**可引用（計數）**：

1. **MTP off 的負載是確定的**：兩個 off 臂的 `requests` **完全相同**（128,920）、
   `misses` 幾乎相同（4,500／4,201）。
2. **MTP 讓池請求 +27.8%**（128,920 → 164,747／157,006）——讓 draft 層的專家多跑一輪。
3. **MTP 讓 miss 率由 3.4–3.5% 升到 5.6–5.9%（×1.6–1.7）** —— 而 miss 是 IO。
4. `wall_s`：off 54.2–70.2 s，on 82.0–82.4 s。

## §4 ★ 最強的一條：環境被單獨隔離出來了

**兩個 off 臂的負載完全相同（128,920 請求）、build 相同、pool 相同，t/s 卻是 7.89 與 9.62
⇒ 環境單獨貢獻 22%。**

而且它同時說明一件事：**當前環境（free 0.11–3.89 GB、swap 91%）拿不到記錄的 10.78–10.98**
—— 兩個 off 臂都低於那個區間。

⚠️ 但這四臂是**連續跑、無人工等待**的 ⇒ 熱壓被自己從 NOMINAL 推到 HEAVY（§2 的表就看得出來）。
所以那 22% 是**「連跑造成的自我加熱」＋「背景負載」的合計**，不再是「純環境」。

## §5 對「12.62 vs 10.x」的分解

| 成分 | 量級 | 證據等級 |
|---|---|---|
| **口徑／實作**：HTTP `llama-speculative-simple` vs llama-bench `--spec-type draft-mtp` | MTP off 9.82 vs 10.62 ⇒ **約 8%** | 有記錄可查（`MTP_POOL_CONFOUND` §1 的 2×2） |
| **MTP 本身** | HTTP 口徑 **+28.5%**（12.62 vs 9.82）；llama-bench 口徑下**只量到負載增加**（§3 第 2／3 條），t/s 不可引用 | ★ **仍未裁決** |
| **環境（pool）** | 4 GiB vs 8 GiB ⇒ 一階混淆（已由 `MTP_POOL_CONFOUND` 證明） | 有記錄 |
| **環境（本輪）** | **22%**（同臂同負載）～ **65%**（四臂極差 5.81–9.62） | ★ 本輪實測 |

⇒ **12.62 既沒有複現、也沒有被證偽**：它與 10.x 的 21% 差異，**落在本輪量到的環境噪聲量級之內**
⇒ **用當前的測量方法判不出來。** 這不是「不知道就算了」——它是一個**可操作的結論**：
在噪聲降到 ~5% 之前，任何 MTP on/off 的 t/s 結論都是不可信的。

## §6 下一步（把噪聲壓下去，成本已估）

1. **臂間等待熱壓回落**（每臂前輪詢 `notifyutil -g com.apple.system.thermalpressurelevel` 直到 0），
   並**只在 thermal 匹配的對裡比**。本輪沒做 ⇒ 22% 裡包含自我加熱。
2. **優先看計數**（§3）：`requests`／`misses`／`wall_s` 不受熱影響 ⇒ 在噪聲環境下，
   它們比 t/s 更能回答「MTP 值不值得」。
3. **要 t/s 就單臂獨立跑**（一次一個 spec，中間讓機器回落），而不是四臂連跑。
4. **仍未量到的關鍵數**：**投機時每步多少 token**。`MTP_POOL_CONFOUND` §4 早已指出
   「整個問題掛在這一個數上」—— `llama-bench` 的 `CGC-MTP-PERF` 有印它（`2fcf616b9`），
   但本輪沒有取用。

## §7 踩到的坑（方法論，寫下來免得重踩）

`EXTRA="--spec-type draft-mtp --spec-draft-n-max 3"` 之後**不加引號使用**時，
**zsh 不做 word splitting**（與 bash 不同）⇒ 整串被當成**一個參數**傳給 argparse
⇒ `error: unrecognized arguments: --spec-type draft-mtp --spec-draft-n-max 3`。
第一次 ABBA 的兩臂因此白跑（失敗是**出聲的**，沒有靜默）。**在 zsh 裡傳多詞參數要用陣列，或寫全。**

## §8 檔案歸屬與邊界

- `scripts/check/llama_bench_matrix.py` 是**另一條線的未提交修改檔**（mtime 15:35:13；
  `--spec-type` 定義在 `:504`）⇒ 本輪**只讀、未改、未 stage**。
- 本檔是**原始測量記錄**，不是白皮書；它不含 `src/` 變更，未跑 D5 數值閘門（依 D5 的适用范围修訂，
  0 個 `src/` ⇒ 不適用）。
- 本輪的 t/s 全部**不可引用**（§3 的裁決）。
