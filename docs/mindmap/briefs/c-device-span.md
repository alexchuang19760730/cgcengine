# device span 歸因（最大一塊時間） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：認領交付 step 裡最大一塊沒人歸因的時間：證明它是裝置忙碌時間（不是 host 開銷），並指出它住在哪些節點分區。

- 主題：GPU 計算／頻寬　·　子目標：**C kernel／頻寬效率**（天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

認領交付 step 裡最大一塊沒人歸因的時間：證明它是裝置忙碌時間（不是 host 開銷），並指出它住在哪些節點分區。

## 2. 判準

逐 buffer 的 GPU span 必須與 host 的 wait 對得上；且要能給出 MoE／attention 分區佔比

## 3. 結果

成立：57% 住在 MoE 區（層內節點 0–39）、42% 住在 attention／GDN 區（40–89）；邊際 verify token +26.3 ms 裡 MoE +14.2（kernel 地板 +7.9）、attention/GDN +13.4。⛔ 附帶作廢：step 級 gpu_sum／union_sum 不可引用（假 buffer 灌水）⇒ 先前「裝置 span 邊際 ~31 ms」作廢。

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 歸因成立，但不是可用槓桿：受模型形狀與 kernel 物理約束 ⇒ 歸 C 天花板軸，只作背景約束與上界。

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [頻寬屋頂／dense GEMV 上界](c-bandwidth.md) | 3a | dense GEMV 每步 13.39 ms（13.7–16.8%），實測已跑 56–108 GB/s（DRAM 峰值 108.8）；就算全部打滿 100% 峰值也只省 1.92  |
| [G1：可達上界 / G1-G7 sweep](g1.md) | 3a | 串行且空轉 19.2% 是「可恢復」的形狀，但 G1 已判定不可達（下界 9.0% > 5%）⇒ 「多少」目前不可引用（44/45 份非零 ⇒ 表內容會動） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `DEVICE_BUSY_ATTRIBUTED_2026-09-23.md §0／§1.1；CB_IS_THE_DEVICE_RESULT_2026-09-20.md` |
| 備註 | 歸因成立，但不是可用槓桿：受模型形狀與 kernel 物理約束 ⇒ 歸 C 天花板軸，只作背景約束與上界。 |
| 軸性質 | 天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領 |
| 對應報告 | 2 份 |

- [CB_IS_THE_DEVICE_RESULT_2026-09-20.md](../../CB_IS_THE_DEVICE_RESULT_2026-09-20.md)
- [DEVICE_BUSY_ATTRIBUTED_2026-09-23.md](../../DEVICE_BUSY_ATTRIBUTED_2026-09-23.md)

---

← [G4：元素級融合 kernel](g4.md)　·　[總目錄](index.md)　·　[HTML 版](c-device-span.html)　·　[頻寬屋頂／dense GEMV 上界 →](c-bandwidth.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
