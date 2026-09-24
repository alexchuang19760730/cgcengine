# FP 順序／ILP／bit-identity：他的形狀成不成立？（2026-09-22）

判的是一句話：**「鎖 FP 順序而不關 ILP（多累加器 + 固定順序合併），是唯一能同時要 ILP 與
bit-identity 的形狀」**。

結論：**診斷對、處方缺一半、「唯一」過強**。而劑量最大的那一半不在他那 24 行裡，在我們自己
樹上一行被註解掉的程式碼。

---

## 1. 他的「診斷」是對的 —— 而且對在我們這顆 build 上

他表格第一格：「要快（ILP）⇒ compiler 自己重排累加順序 ⇒ bit-exact 被破壞」。
這在一般 C/C++ 裡是**錯的**（無 `-ffast-math` 時編譯器不得重排 FP 加法）。但**在我們這裡是真的**：

| 事實 | 證據 |
|---|---|
| 本 build 走 **embed** 路徑，`.metal` 原始碼被 `incbin` 塞進 dylib，**在 runtime 才編譯** | `CMakeCache.txt:500 GGML_METAL_EMBED_LIBRARY:BOOL=ON`；`ggml-metal/CMakeLists.txt:42-52`（`.incbin`） |
| runtime 編譯用 `MTLCompileOptions`，**`fastMathEnabled` 保持預設（YES）** | `ggml-metal-device.m:229-234` |
| 關掉它的那一行**被註解掉了** | `ggml-metal-device.m:232` → `//[options setFastMathEnabled:false];` |
| 編譯期的 `GGML_METAL_SHADER_DEBUG`（= `-fno-fast-math`）**是 OFF**，且只在**非 embed** 分支生效 | `CMakeCache.txt: GGML_METAL_SHADER_DEBUG:BOOL=OFF`；`ggml-metal/CMakeLists.txt:66-78` 整個 `if/else` 在 `else()`（copy-to-bin）分支內 |

⇒ **我們現在這顆 engine 的 Metal kernel 是在 fast-math 下編譯的**，編譯器**有權**重組 FP 歸約。
⇒ 「改 kernel（unroll／多累加器）⇒ bit 變」是**這個 build 上真會發生的事**，不是教科書假設。

這一點要給他記功：他不是泛泛而談。

---

## 2. 他的「處方」缺一半 —— 而且缺的是關鍵那半

他的論證是：「最後合併的順序**寫死在程式碼裡**（`p0+p1+p2+p3`）⇒ 不管 compiler 怎麼排都是這個順序」。

**這一步在 fast-math 下不成立。** `fastMathEnabled=YES` 授權的正是「重組 FP 運算」；
寫在源碼裡的 `((p0+p1)+p2)+p3` 一樣可以被編譯器改成 `(p0+p1)+(p2+p3)` 或其他樹形。
**源碼順序只在 fast-math 關掉時才約束編譯器。**

⇒ 完整配方是**兩件事必須同時做**：

```
① sum_parts[0..3] 各自獨立累加        → 拿 ILP（互不依賴，可平行）
② MTLCompileOptions.fastMathEnabled=NO → 讓源碼裡的合併順序真的綁住 compiler
```

只做 ① 不做 ②：**拿到 ILP，沒拿到 bit-exactness**（他的 24 行單獨看就是這樣）。
只做 ② 不做 ①：**拿到 bit-exactness，歸約變單鏈、ILP 歸零**（會變慢）。
⇒ 「同時要兩者」**需要兩個一起**，他那句話把 ② 當成隱含前提了。

### 好消息：② 在我們樹上是一行現成的

`ggml-metal-device.m:232` 的 `//[options setFastMathEnabled:false];` —— 上游寫好又註解掉。
要試只需取消註解 ＋ 重建。

---

## 3. 「唯一能同時要兩者的形狀」—— 過強

- **不唯一**：k=2／4／8 個累加器都行；本質是「**多累加器 + 固定合併樹 + 禁 fast-math**」這個
  *原則*，不是某個特定形狀。
- 而且嚴格講還有第三條路：**不加累加器，就接受單鏈**（bit-exact 但慢）。它也是一個形狀，
  只是我們不要。
- 真正值得學的是這句改寫：**「把歸約形狀從『交給 compiler 決定』改成『寫死 + 讓 compiler 不准動』」**。

---

## 4. ⚠️ 一個他完全沒提的代價：樹 ≠ 鏈

就算 ①+② 都做到，結果**也不會與現在這顆 build bit-identical**：

- 現在（P2-B/C）：單鏈 `((((0+t0)+t1)+t2)+t3)`
- 他的形狀：樹 `(p0+p1)+(p2+p3)`

浮點加法不滿足結合律 ⇒ **兩者末位元不同**。

而本專案的 M1 定義是「**兩份 dump 的 row_fnv1a64 逐步相同**」
（`scripts/check/cgc_logits_oracle_compare.py` 的 `METRIC_DEFS`：
`numeric_identity = M1: same row_fnv1a64 for every step (= bit-identical logits)`），
比的是**兩顆 binary 的產出**，不是對外部 oracle 的符合性。

⇒ 一旦採用他的形狀，**M1 會一次性從 100% 掉到 0**，必須**重新 baseline**，
過渡期只能用 M2（argmax 相同）／M3（top-N 集合相同）當門檻。
⇒ 這不是「修好 bit-exactness」，是「**換一個新的、可證明的 bit-exactness 基準**」。
講成「修好」會讓人以為數字可以對照舊基準，那是錯的。

---

## 5. 「本線該學嗎」—— 該學，但學的是整套，不是那 24 行

值得學的三點（按價值排序）：

1. **把 bit-exactness 從「每次要重驗證」變成「由建置保證」**。這一條移除的是一個**閘門**，
   不是一個瓶頸 —— 它讓「改 kernel」這整類動作不必每次重跑正確性驗證。
2. **`ggml-metal-device.m:232` 這個已存在的開關**。它是我們自己樹上的資產，成本一行。
3. 多累加器本身是標準手法，沒什麼神秘。

不值得學的：他那句「唯一」，以及「這樣就能回到 25 t/s」（見 §6）。

---

## 6. 「能讓昨天失敗的方案重來嗎」—— 逐個判

| 昨天／先前判死的方案 | 這個形狀能不能救 | 理由 |
|---|---|---|
| **09-18 down-combine 融合 kernel**（正確但**慢 15–18%**） | ❌ | 它慢的原因是**平行度／grid 形狀**（expert 被放在 grid 某一維），與 FP 順序無關 |
| **G4 融合 kernel**（已結案：別寫） | ❌ | 同上 |
| **K5 小 op 群** | ❌ | 18 µs／dispatch 的**固定開銷**，撐 grid 動不了；與 FP 無關 |
| **25 t/s** | ❌ | 需要 **1.98×**。kernel 層天花板個位數 %；本線剛實測 context 那維只有 **~+20%**（§7） |
| **M-W（workers 8→2）** | ❌ | 不相干 |
| **「任何因為不能證明 bit-exact 而被擋下的 kernel 優化」這個類別** | ✅ | 它打開的是**門**，不是瓶頸 |

---

## 7. 順帶：本線剛跑完的 context-depth A/B（這一輪受污染，僅看方向）

`Backup/phase_decomp/L3/context_depth_ab.py`，一次 llama-bench、`-d 512,0,0,512`（ABBA）、reps 3：

| 順序 | depth | avg_ts |
|---|---|---|
| 1（A） | 512 | 7.512 |
| 2（B） | 0 | 9.988 |
| 3（B） | 0 | 9.300 |
| 4（A） | 512 | 8.611 |

- **4/4 分離**（兩個 B 都高於兩個 A）；中位 A 8.062 / B 9.644 ⇒ **d=0 比 d=512 快 ~+19.6%**
- 但 **A 臂絕對值 7.5–8.6 遠低於封版 12.57**，且**跑的途中散壓掉到 HEAVY**（14:47 起跑時 NOMINAL，
  14:55 讀到 HEAVY）⇒ **絕對值不可引用，只有相對比較勉強可看，需乾淨窗口重跑**。
- 另外這一輪的 binary 是 14:01 那顆（**P2-C revised**，含 `sum_parts`），不是 baseline ⇒
  絕對值本來就不能跟 12.57 比。

⇒ 就方向而言：**context 是一維真槓桿，但只有 ~20%，不是 2×**。

---

## 8. 現在真正該做的第一件事（不是他講的那件）

> **⚠ 本節標題句已被更正（同日 15:2x，0 重建）→ `docs/BITIDENTITY_AND_0907_KNOWHOW_2026-09-22.md`**
> **M1 是驗證過的：9/9 = 100%**（09-22 13:51 gate，`Backup/m123_oracle_gate/summary_p2_rebased.json`，
> comparable=true、config_diffs=[]、ref=v6_nbaware；且跑的那顆 dylib md5 `430f9d315cbcf6c4`
> ＝當前產物）。我寫本節時沒查 `Backup/m123_oracle_gate/`。
> 邊界：只證「對 09-17 v6 參考」，不證對 CPU 真值；「P2-B 相對上游有無漂移」仍無解。
> ⇒ 下面的「第一件事」降級為**待辦**，不再是「未驗證的缺陷」。

**我們從未驗證過：12.57 這顆封版 baseline 本身是不是 bit-exact。**

- 它含 P2-B/C（`#pragma unroll` + float4），而 Metal 在 fast-math 下編譯（§1）⇒
  **它有權已經不是 bit-exact，而我們沒有證據說它是**。
- 記憶裡查得到「未驗證，要 `cgc_logits_oracle_compare.py` 實跑才算數」（2026-09-22.md:361），
  和 D5 指標定義在 `METRIC_DEFS` —— **沒有任何一條寫著「12.57 baseline 已通過 M1」**。

⇒ 如果 P2-B/C 真的改了 bit，那**我們正在 ship 的基線帶著一個未證明的正確性缺陷**，
而「修成可證明的 bit-exact」會讓數字**往下走** —— 跟他設想的方向正好相反。

### 判定步驟（各一次，都要過門禁）

1. **M1 驗證 baseline**：用 `scripts/check/cgc_logits_oracle_compare.py` 比
   「含 P2-B/C」vs「去掉 P2-B/C」兩顆 binary 的 logits dump（M1 metric）。
   ⚠ 需要一顆 reference build ⇒ **一次重建**。
2. 若 M1 ≠ 100%：**先把 baseline 修乾淨**（加 `-fno-fast-math` 或他的形狀），
   **重新凍結一個新的封版數字**（預期往下，要讓使用者知道）。
3. 若 M1 = 100%：他的形狀就只是「未來改 kernel 的保險」，**沒有立即收益**，
   排在其他工作之後。

---

## 9. 門禁現況（寫這份時）

- 8080：已無 listener（14:5x 殺掉閒置的 `llama-server` PID 81094）
- 散壓：**HEAVY**（14:55 讀到）⇒ 上面任何一步現在都不能跑
- swap：~79%，usable ~10 GiB
- ⚠ **產物與源碼仍不一致**：`libggml-metal.0.19.0.dylib`（14:01 建）含 10 處 `sum_parts`，
  而源檔（demo 分支）是 0 ⇒ **量測前必須先重建 baseline 臂**

---

## 10. 一句話給他

> 「你診斷對了，我們這顆 build 的 Metal 確實在 fast-math 下跑（`ggml-metal-device.m:232`
> 那行 `setFastMathEnabled:false` 被註解掉了），所以 compiler 有權重排。
> 但你的處方少一半：**在 fast-math 下，寫在源碼裡的合併順序綁不住 compiler**，
> 必須連 `fastMathEnabled=NO` 一起做。而且做了以後結果跟現在這顆 build **不會**
> bit-identical（樹 ≠ 鏈），M1 會一次歸零、要重新 baseline。
> 所以它是『換一個可證明的基準』，不是『修好 25』。」
