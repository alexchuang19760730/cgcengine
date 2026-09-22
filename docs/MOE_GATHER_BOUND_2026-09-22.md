# MoE 專家庫 gather（MUL_MAT_ID）的 kernel 側上界（2026-09-22）

載體 Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X · 探針 `scripts/check/shape_probe/`
（`dense_gemv_probe.cpp` ＋新 `mmid_shapes.py`）· 產物 `Backup/phase_decomp/L3/shape_probe/mmid_gather.json`
· 窗口 **busy-overridden**（usable 5.31 GiB < 8.0 GiB 門檻、swap 3.7 GiB，如實記錄進產物）
· `libggml-metal.0.dylib` md5 `544b6c94bc7a372d`

---

## 0. 一句話

> MoE 專家庫 gather 每步 **8.09 ms（T=1）→ 31.81 ms（T=4）**，跑在**峰值的 31–50%**
> （gate/up ~45–50%、down ~33%）。它的 bytes 在峰值上只要 3.07 / 12.30 ms，
> ⇒ kernel 側可回收 **5.0 / 19.5 ms**（T=4 是 134 ms 步的 **14.6%**）。
> 但同一批資料給出一個更重要的否定：**這個家族沒有批次攤薄** ——
> 第二個 token 的成本是第一個的 **98%**（8.09 → 7.95 ms/token）。
> 而「shared ids vs cycled ids 差 2%」證明那 40–50% 的天花板**不是路由散射造成的**，
> 是 `mul_mv_id` kernel 本身：**pool 暖度、prerouter、prefetch 都動不了它。**

---

## 1. 這支儀器加了什麼

`run_shapes.py` 原本**刻意跳過 3D 張量**（「MoE expert bank → MUL_MAT_ID, priced elsewhere」）。
這份就是那個 elsewhere：把 `--op mul_mat_id` 加進同一支探針（同一個 marginal 模型、同一支獨立二進位、不動樹上共用的 dylib）。

| 新增 | 作用 |
|---|---|
| `--op mul_mat_id` | 走 `ggml_mul_mat_id`：`as`=[K, N, E]、`b`=[K,1,T]、`ids`=[U,T]（形狀照 `ggml.c:3331` 的 assert） |
| `--experts/--used/--tokens` | 真幾何：E=256、U=8、T=1..4（T 就是 MTP 的 verify 寬度） |
| `--target-ms` | marginal 是兩個時間的**差**，雜訊是兩次相加 ⇒ 拉長單次量測是買精度的最便宜一步 |

**為什麼只配置一個 bank：** copy i 用 `[(i·U+j) % E]`，所以 32 個 copy 恰好把整個 bank（82 MiB）讀一遍、
圖內沒有專家被讀兩次 —— 與「32 個各自獨立的 bank」付出相同的 DRAM 流量，卻只用 1/32 的裝置記憶體，
而且**不需要 dense 那一格用過的「distinct」近似**。

真幾何（讀模型自己的 header）：`ffn_gate_exps`/`ffn_up_exps` = IQ2_S `(2048, 512, 256)` 82.0 MiB；
`ffn_down_exps` = IQ3_S `(512, 2048, 256)` 110.0 MiB；每層 gate+up+down 各一次，40 層。

---

## 2. 量測（marginal 模型，b=16 vs 32，1.2 s/點）

| T | 形狀 | marginal µs/op | MiB/op | GiB/s | %峰值 | fixed µs/graph | ms/步 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | down_exps | 100.30 | 3.44 | 33.5 | 30.8% | 235 | 4.01 |
| 1 | gate_exps | 50.79 | 2.56 | 49.3 | 45.3% | 385 | 2.03 |
| 1 | up_exps | 51.25 | 2.56 | 48.8 | 44.9% | 363 | 2.05 |
| | **合計** | | | | **38.0%** | | **8.09** |
| 2 | （三項合計） | | | | 38.5% | | **15.98** |
| 3 | （三項合計） | | | | 38.8% | | **23.79** |
| 4 | down/gate/up | 376.81 / 210.07 / 208.46 | 13.75 / 10.25 / 10.25 | 35.6 / 47.6 / 48.0 | 32.8 / 43.8 / 44.1% | 187 / 290 / 313 | **31.81** |

（家族列的 % 是整家族加權的結果，低於 gate/up 自己的 45–50%：down_exps 只跑到 ~33%。）

三次獨立重跑同一組：T=1 落在 7.41–8.09、T=4 落在 29.55–31.81 ms/步（家族內 spread ≈ 7%）。
**所以可以引用的量級是「T=1 ≈ 7.4–8.1、T=4 ≈ 29.6–31.8 ms/步」，單一讀數不可當定論。**

峰值用 `108.8 GiB/s`（本機 8 執行緒實測，與 `run_shapes.py` 同一常數），
**不是** spec 的 120 GB/s。單位是 GiB/s：探針印的 MiB/ms 與它同值 ——
寫這支 driver 時我自己踩過一次（`bytes / GiB·s⁻¹` 少了 `1024³`，把 3.07 ms 印成 3.15 s），
現在有自測釘住它。

---

## 3. 否定的那半：這個家族**沒有**批次攤薄

| | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| 家族 ms/步 | 8.09 | 15.98 | 23.79 | 31.81 |
| 每 token ms | 8.09 | 7.99 | 7.93 | 7.95 |

**多一個 verify token 要付第一個的 98%。** 這是一條對 MTP 經濟學不利的硬讀數：

- 攤薄的只有**每張圖的固定項**（fixed ≈ 190–385 µs/graph，圖越大攤得越薄），
  而權重讀取本身是**線性**的 —— 每個 token 都要把自己選中的 8 個專家整份讀一遍。
- 所以「把 verify 加寬來攤掉 round trip」在 **kernel 這一側拿不到折扣**。
  它只能靠 accept rate 賺（`S = E / (1 + m·k)` 裡的 `m` 幾乎不隨 k 下降）。
- 這也解釋了為什麼先前那條「verify 邊際 = 61 ms/token」的量測裡，
  gather 只佔 ~7.9 ms/token（13%），其餘 87% 在別處 —— 而**即使把 gather 全部歸零，
  也只能買回 13%**。

---

## 4. 天花板的性質：kernel，不是散射（這一格決定下一步做什麼）

同一個形狀、同一個 batch，只把 ids 從「每個 copy 用不同專家」換成「所有 copy 都用同樣那 8 個專家」：

| 形狀 | T | cycled（各自不同專家） | shared（全部同一批，完美 cache 友善） | 差 |
|---|---:|---:|---:|---:|
| iq2_s 2048×512 | 1 | 41.05 GB/s | 41.96 GB/s | **+2.2%** |
| iq2_s 2048×512 | 4 | 47.95 | 48.34 | +0.8% |
| iq3_s 512×2048 | 1 | 33.20 | 34.42 | +3.7% |
| iq3_s 512×2048 | 4 | 37.54 | 38.07 | +1.4% |

**結論：那 50–70% 的缺口不是「專家被讀得很散」造成的。** 讀同一批專家也一樣慢。
⇒ 命中率、池子暖度、prerouter 預測、prefetch 對這個家族**都沒有槓桿**；
   可動的只有 `mul_mv_id` kernel 自己的 tile/simdgroup 配置。

**而這裡有一個現成的、便宜的落點：** 另一條線今天的 NSG patch 只接了
`ggml_metal_library_get_pipeline_mul_mv()`（dense 那條），他們自己的註解寫著
「`get_pipeline_mul_mv_id` / `_down_combine` 里的同名 case **没动**（那是 MoE 的路）」。
換句話說：**被掃過的那個家族本來就有 45–91% 峰值；有 headroom 的這個家族還沒被掃過。**
它同樣是函式常數（pipeline 名印得出來：`kernel_mul_mv_id_iq2_s_f32_nsg=2`），
所以同一個 env 旋鈕延伸過去就能掃 —— 掃描器（`sweep_nsg.py`）已經在了，
現在缺的只是 id 那條 pipeline 的接線。**這是本檔推出來的第一順位動作。**

---

## 4b. 那個第一順位動作做完了：nsg 延伸過去之後，IQ2_S 是**否定**、IQ3_S 是**拒答**

`CGC_MMV_NSG` 現在也接上了 `ggml_metal_library_get_pipeline_mul_mv_id()`（IQ3_S 那格，
dense 的 patch 沒動它）。旋鈕有沒有真的到達 kernel **不由 env 推論**，而是讀探針印出的
pipeline 名：`kernel_mul_mv_id_iq3_s_f32_nsg=1|4|8|16|32`，五個值都證實到達。

> **這個 hunk 本身未 commit**（見 §7）：`ggml-metal-device.cpp` 同一個檔案裡混著另一條線
> 的 dense NSG hunk，而那個家族已被他們自己的量測判死 ⇒ 依先前建議 park，不把它們的
> in-flight 改動與樹上 binary 的重建一起凍進 commit。本檔的量測用的是 `/tmp` 的獨立 build
> （`libggml-metal` md5 `650dd9d1d9fb89a4`），不依賴樹上那份。

```sh
python3 scripts/check/shape_probe/mmid_shapes.py --nsg-sweep unset,1,4,8,16,32 \
        --reps 5 --tokens 4 --libdir /tmp/cgc-nsgid-build/bin \
        --json Backup/phase_decomp/L3/shape_probe/mmid_nsg_sweep_rotated.json
```

### IQ2_S（gate_exps／up_exps）：**否定，可引用**

唯一一個 null cell 站得住的通道（同一顆 dylib、同一窗口、交錯排列）：

| nsg | unset | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| %peak | 44.3 | 43.8 | 44.0 | 44.5 | 44.3 | 44.7 |

null cell（unset 在頭尾各一次）**0.4%**，全距 0.9 個百分點 ⇒ 這是**否定**，不是「還沒調好」。
第三個獨立讀數來自出貨 dylib 本身（改動前）：nsg 2/8/16 → 215.7/218.1/217.1 µs，一樣平。
**tile 維度對 IQ2_S 不是槓桿。**

### IQ3_S（down_exps）：**拒答，而且是被工具自己拒的**

今天所有 IQ3_S 讀數都落在通道自己的噪音裡：

| 跑法 | unset 這個「同一個值」自己散開 | 判定 |
|---|---:|---|
| 單趟（未輪替） | 29.8% → 20.4% = **31%** | 無讀數（舊規則卻印了「no lever」——見下） |
| 輪替 3 rep | 18.1–21.2% = **15.7%** | INVALID |
| 單臂 × 6 rep（1.2 s/點） | 22.6–36.2% = **43.5%** | INVALID |
| 同一跑法的另一次 | 12.8–45.6% = **105%** | INVALID |
| 單臂 × 6 rep（down_exps） | 23.6–25.9% = **9.2%** | INVALID |

**機制是量到的，不是猜的**：這台機器的 GPU 與合成器共用，同時活著 WorkBuddy renderer（90% CPU）、
WindowServer（37%）、Freebuff renderer（23%），compressor 3.36 GB、swap 3.7 GB。所以
**一次 invokation 內部就換了模式**——拉長積分時間救不了，實測 4 s/點反而更糟（11.8% → 53.7%）。

### 這條規則修掉了什麼（一個會印出相反結論的閘門）

未輪替那一趟的產物 `mmid_nsg_sweep.json` 對 IQ3_S 印的是 **「no lever」**，而它的 null cell 是 31%。
舊規則把噪聲通道的中位數當成讀數——**同一份資料既是「沒有槓桿」又是「沒有讀數」，而它選了前者**。
現在：每臂輪替順序（`rotated()`）、每 rep 記 swap、判定是純函式 `nsg_decide()`，
**通道自己的 unset 散開 > 5% 就回 INVALID 並拒絕發表中位數**。舊產物不回溯改寫，只標為已取代。

### 決策上真正重要的一格（不需要那個讀數也成立）

即使 nsg 把 down_exps 從 29.8% 一路打到 45% 峰值，這個家族在 T=1 也只有
**8.09 ms/步 ÷ 134 ms = 6.0%**，純頻寬地板 3.07 ms ⇒ 全拿回來是 **5.0 ms = 3.7%**
⇒ 12.57 t/s 上約 **+0.4 t/s**。**這個旋鈕的物理天花板不足以構成 25 t/s 的路**，
所以「等一個乾淨窗口把它量完」的價值是**關掉一條線**，不是賺到速度。

---

## 5. 與引擎側對帳

| 來源 | 每 token | 說明 |
|---|---:|---|
| 本探針（T=4，kernel 側，含固定項） | **7.95 ms** | 純 kernel，無引擎（T=1 是 8.09；取 T=4 的每-token 值與下面同口徑） |
| 引擎 `CGC-VERIFY-OP`（帶儀器） | ~8.2 ms | down 157 + gate 134 + up 113 µs/token/層 × 40 層（先前記錄） |
| 先前另一支 kernel 量測 | 8.36 ms | 記錄值，與上面同一量級 |
| 本探針的**純頻寬地板** | 3.07 ms | 342.6 MiB/步 ÷ 108.8 GiB/s |

引擎的實測 ≈ 探針的實測（差 5% 以內）⇒ **引擎沒有把這個家族做壞，也沒有把它做快：
兩邊都在 40% 峰值附近。** 這很重要，因為它把「修 host 簿記」這條路從這個家族上移除了 ——
剩下的差距是 kernel 的。

---

## 6. 對 25 t/s 的意義

`ml`=1.8 下 25 t/s 需要 **72 ms/步**；今天是 134 ms ⇒ **要拿掉 62 ms**。

| 假設 | T=4 家族耗時 | 可回收 | 佔 134 ms 步 |
|---|---:|---:|---:|
| 現在（實測） | 31.81 ms | — | 23.7% |
| 全部打到 70% 峰值（現實目標） | 17.57 ms | **14.24 ms** | **10.6%** |
| 全部打到 100% 峰值（物理上不可能的） | 12.30 ms | 19.51 ms | 14.6% |

⇒ 這是 dense GEMV 被結案之後**第一個越過 3% 門檻的家族**，而且它一個人就能吃掉
「必須拿掉的 62 ms」中的 **14–19 ms（23–31%）**。但它不是答案的全部 ——
剩下的 43–48 ms 仍在別處（remainder：post-layer → 下一輪那段沒有逐層儀器蓋到的工作）。

---

## 7. 誠實欄（必須跟著一起引用）

- **窗口不達標**：usable 5.31 GiB < 8.0 GiB（**class = busy-overridden**，已寫進產物）。
  GPU 微基準的宿主足跡很小，但**絕對 µs/op 可能描述的是鄰居**；可引用的首先是
  **佔峰值百分比**（本檔的結論都建立在比例上）。要拿絕對值，必須在乾淨窗口重取一次。
- **ids 模型是「無跨 token 共享」**：我讓每個 token 選各自 8 個專家。
  真實 verify 批次裡 4 個 token 常常選到重疊的專家 ⇒ **真實 bytes 只會更少** ⇒
  真實地板更低 ⇒ 本檔的 headroom 是**低估**（保守）。這一點方向對結論有利。
- **沒有量 CGC 的 pool slab 路徑**：探針量的是「以 expert id 索引 3D bank」，
  即 `ggml_mul_mat_id` 的路；引擎的 slab 是另一套機制（`CGC-VERIFY-OP` 那側的數）。
  兩者要並列才完整，單獨引用任一個都會誤導。
- **型別沒有平均**：down_exps 只價 IQ3_S（38/40 層；34/38 是 IQ4_XS、40 是 Q3_K）。
  混合的比例是上界的一部分，不是隱藏的加權平均。
- **fixed 是「每張圖」而不是「每個 command buffer」**：探針的圖是 16–32 op，
  引擎一步是 633 個 CB（`docs/DECODE_STEADY_BASELINE_2026-09-19.md`）。
  這兩個數字**不能直接相乘**（633 × 300 µs 會超過整個步），對帳是未竟項。
- **重跑 spread ≈ 7%**（三次）；單次讀數不可引用。探針二進位在 `.gitignore` 內、不入庫。
- **§4b 的否定只在 IQ2_S 上成立**：IQ3_S 今天沒有讀數，只有拒答。任何把
  「nsg 沒用」講成**整條 id 路徑**的句子，都超出了手上證據（等於把一個通道的否定借給另一個）。
- **§4b 的引擎 hunk 未 commit**：`ggml-metal-device.cpp` 是共用檔，我的 `_id`/IQ3_S 那格與
  另一條線的 dense hunk 互相依賴（我的呼叫的是他們的 `ggml_metal_nsg_env()`），所以不能只
  取一半；而 commit 引擎原始碼會依檢查 8 一併要求 `build/bin` 產物，那批 binary 是他們
  in-flight 的重建（`libllama` 3.19 MB → 3.18 MB）⇒ 先 park。要把它轉正需要由擁有該檔的
  那一側決定：要嘛一起 commit（source+binary 成對），要嘛把 dense 那半 revert。
- **%peak 的分母是乾淨狀態下量到的 108.8 GiB/s**：在超訂盒況下，同一顆 kernel 的
  每 arm 絕對值會掉到 1.5–2 倍慢，所以 §4b 的比較只用「同一通道內的相對位置」，不用絕對 µs。

---

## 8. 重現

```sh
./scripts/check/shape_probe/build.sh                     # 獨立編譯，不動共用 dylib
python3 scripts/check/shape_probe/mmid_shapes.py --selftest          # 26/26
python3 scripts/check/shape_probe/mmid_shapes.py --tokens 1,2,3,4 \
        --json Backup/phase_decomp/L3/shape_probe/mmid_gather.json
# nsg 掃描（§4b）——務必帶 --reps，單趟無法把旋鈕與漂移分開：
python3 scripts/check/shape_probe/mmid_shapes.py --nsg-sweep unset,1,4,8,16,32 --reps 5 \
        --tokens 4 --libdir <dir with the swept libggml-metal> \
        --json Backup/phase_decomp/L3/shape_probe/mmid_nsg_sweep_rotated.json
# 純噪音底（回答「這台機器現在能不能量 5% 的問題」）：
python3 scripts/check/shape_probe/mmid_shapes.py --nsg-sweep unset --reps 6 \
        --tokens 4 --libdir <same> --json .../mmid_nsg_repeatability.json
# 散射 vs kernel 的判別（§4）：
./scripts/check/shape_probe/cgc_shape_probe --op mul_mat_id --type iq2_s --k 2048 --out 512 \
        --experts 256 --used 8 --tokens 4 --mode batch --batch 32 --iters 1 --target-ms 900 \
        --pool-mib 320 [--shared-weight]
```

產物通過 repo 自己的出處契約：`server_window.py audit-products` → **1/1 帶 window、0 個答不出「盒子忙不忙」**。
