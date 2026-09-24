# 任務 1（執行側）：MoE NSG sweep 在 admit 窗口上的讀數 —— 一個可引用的否定、一個拒答

**日期** 2026-09-23 02:5x · 依 `docs/NEXT_ACTIONS_2026-09-23.md` 任務 1（【執行 Agent】等乾淨窗口跑 MoE
NSG sweep）· 載體 Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X · 探針 `cgc_shape_probe --op mul_mat_id`
（E=256、U=8、T=4、batch 16 vs 32、lean pool 146/174 MiB）· `libggml-metal.0.dylib` md5
`650dd9d1d9fb89a4`（獨立 build，先前已驗與樹上行為一致）· 產物 `Backup/shape_probe/mmid_nsg_admit_*.json`

命令（任務書原文，只把輸出改成不覆蓋既有檔）：

```bash
python3 scripts/check/shape_probe/mmid_shapes.py \
    --nsg-sweep unset,1,4,8,16,32 --reps 3 --pool-mib 0 \
    --libdir /tmp/cgc-nsgid-build/bin \
    --json Backup/shape_probe/mmid_nsg_admit_stamped_2026-09-23.json
```

---

## 0. 一句話

**`CGC_MMV_NSG` 在 MoE gather 路徑上不是槓桿，而這次是在 admit 窗口上量的**：gate 通道六個值全部落在
**45.3–46.9% 峰值**（null cell 0.9% / 3.3%），down 通道 35% 附近持平、`nsg≥16` 之後**明確變差**
（32.2%、21.4%）；兩次獨立啟動都得到「沒有任何臂贏過 unset 超過門檻」。第二支的 down 通道
**拒答**（unset 自己跨 6.1% > 5% 門檻），工具沒有給出判決 —— 這是對的行為，也一併記下來。

---

## 1. 兩支啟動，與任務書四條條件的對帳

| 條件（任務 1） | 第 1 支 `mmid_nsg_admit_…` | 第 2 支 `mmid_nsg_admit_stamped_…` |
|---|---|---|
| `server_window.decision() = admit`，begin 與 end 都要 | ✓ begin 8321.5 MB / end 8324.3 MB、`foreign=[]` | ✓ begin 8271.5 / end 8220 MB、`foreign=[]` |
| 可用記憶體 ≥ 7.8 GiB | ✓ 7.99 GiB（swap 3765 MiB） | ✓ 7.96 GiB（swap 3717 MiB） |
| thermal level = 0 NOMINAL | **當時無此欄位**（見 §3） | ✓ begin **NOMINAL**；end HEAVY（一輪掃描本身就會把盒子加熱；判準是**啟動當下**） |
| null cell ≤ 5%（gate 與 down） | gate **0.9%** ✓ / down **0.4%** ✓ | gate **3.3%** ✓ / down **6.1%** ✗ ⇒ **NO READING** |
| 沒有負的 %峰值 | ✓ 全部為正 | ✓ |
| 可引用（不是 NO READING） | ✓ 兩通道皆給出判決 | gate ✓ / down **拒答** |

---

## 2. 讀數

### 2.1 gate 通道（`ffn_gate_exps`, iq2_s, 2048×512）

| nsg | %峰值（第 1 支） | %峰值（第 2 支） | pipeline 實際值 | 到達？ |
|---|---:|---:|---|---|
| unset | 46.1% | 46.1% | `…_iq2_s_f32_nsg=2` | — |
| 1 | 45.6% | 45.6% | `…_nsg=1` | ✓ |
| 4 | 46.1% | 46.1% | `…_nsg=4` | ✓ |
| 8 | 45.8% | 45.8% | `…_nsg=8` | ✓ |
| 16 | 46.1% | 46.1% | `…_nsg=16` | ✓ |
| 32 | 45.8% | 45.8% | `…_nsg=32` | ✓ |

⇒ 六個值之間的總跨度 **0.6 個百分點**，而 null cell 分別是 0.9% / 3.3%。**沒有任何臂贏過 unset。**

### 2.2 down 通道（`ffn_down_exps`, iq3_s, 512×2048）

| nsg | %峰值（第 1 支） | %峰值（第 2 支） |
|---|---:|---:|
| unset | 35.7% | 34.5%（**跨 6.1% ⇒ 拒答**） |
| 1 | 35.5% | 35.5% |
| 4 | 35.6% | 35.5% |
| 8 | 35.2% | 35.2% |
| 16 | **32.4%** | **32.2%** |
| 32 | **21.5%** | **21.4%** |

⇒ 1/4/8 與 unset 同級（差 <0.5pp）；**16 與 32 是明確的回退**，-3 與 -13 個百分點，比任何 null cell 都大
——所以「這個旋鈕有作用，只是沒有任何一個方向是好的」這句話是量出來的，不是推論。

### 2.3 與舊讀數的關係

`docs/MOE_GATHER_BOUND_2026-09-22.md` 記 gate/up **45–50%**、down **33%**；本次 **45.3–46.9%** 與 **35.2–35.7%**
（跨 session、同型別、同幾何）⇒ 這支儀器在通道層級是可重現的。而該文件的主結論（天花板的主人是
`mul_mv_id` kernel，不是路由散射）與本輪一致：**tile 旋鈕動不了它，所以「pool 暖度／prerouter／prefetch」
整族也動不了它。**

---

## 3. 這輪暴露並修掉的兩個工具缺陷（都在同一支探針上）

1. **`--help` 直接崩潰**。`--pool-mib` 的說明字串用 `% POOL_MIB_DEFAULT` 自己先格式化一次，剩下的
   `26%%` 變成單一 `%`；argparse 之後**再**用 action 的 params 對每個 help 做一次 `%`-格式化
   ⇒ `TypeError: must be real number, not dict`。修法：改成字串串接，完全不進 `%`-格式化
   （`26%%` 留給 argparse 消費）。這是「任務 1 的命令能不能被驗證」的前置條件。
2. **產物沒有 thermal 欄位**，而任務書把 `thermal level = 0` 列為四條條件之一 ⇒ 第 1 支啟動**無從對帳**。
   修法：新增 `thermal_stamp()`（**匯入既有 `thermal_pressure`**，不重寫），在 begin/end 各記一次
   `thermal_begin` / `thermal_end` 並印在 window 行上；讀不到時保持 `None`，**不偽造成 NOMINAL**。
   第 2 支啟動即是帶著這個欄位跑的（begin NOMINAL、end HEAVY —— 後者正是「一輪掃描會加熱盒子」的證據，
   也是為什麼判準是**啟動當下**而不是整輪）。自測 41/41 仍過。

---

## 4. 判定，與任務 5 的分支

任務 5 的分支條件是「NSG < 5% ⇒ 排除這個方向，執行側準備測下一個 knob」。本次落在 **<5%** 那一支：
gate 通道 0.6pp 跨度、down 通道只有回退。⇒ **MoE NSG 這條路排除**，下一個 knob 就是任務 3
（expert cache 大小同場配對），已在同一窗口上開跑
（`scripts/check/cache_size_ab.py`，R T R T … R 交錯，見 `docs/CACHE_SIZE_AB_2026-09-23.md`）。

---

## 5. 誠實邊界

- **swap 3.7 GiB 是帶進來的**（別條線的痕跡），兩個 `window_begin/end` 都記錄了；探針自己的
  `window.class` 欄位說明「GPU 微基準 host 佔用極小，所以比值是讀數、絕對 µs 可能描述鄰居」。
- 第 1 支啟動**沒有 thermal 章**，所以它只能對上四條條件裡的三條；第 2 支補上了，但它的 down 通道
  因為通道自己移動而**拒答**。兩支合起來才覆蓋完整 —— 這點寫在 §1 的對帳表裡，而不是挑一支當結論。
- 只有 **T=4**（MTP verify 寬度）與 batch 16/32 的 marginal 模型；`--tokens` 沒有掃過 1/2/3。
- 引擎是**獨立 build**（`650dd9d1d9fb89a4`），樹上的 dylib 沒有被碰；`--pool-mib 0` 是 lean pool，
  與任務書一致。
- 未 commit；零殘留行程（`pgrep` 在兩個實驗之間確認為 0）。
