
### §EN-7.1 §9.18.6 的收尾更正與一個未結項（2026-09-16 20:0x）

**(a) 我對使用者說過一句錯的話，已在這裡更正（不在已提交的文件裡）。**
我說「repo 裡沒有既有的『張量輸出擷取』」——**錯**。`CGC_S1_OUT_CAP=1|pre`（`llama-context.cpp:3441+`）
就是同一件事的**舊版**：它釘住 `ffn_moe_down` 的輸出並印 `fnv1a64` ＋ `vals=[...]`。
真正的差別在**擾動**：它用 `ggml_set_output` 釘住張量，而那會動 allocator／graph（原始碼自己寫
「診斷臂專用、閘門臂永不可開」，`llama-context.cpp:6259`）；`CGC_TENSOR_CAPTURE` 是**不擾動圖**的版本
（唯一例外是那個屏障）。**兩者互補，不是取代。**

**(b) 我用它做的交叉確認未成立（未結項，不要美化）。**
`CGC_S1_OUT_CAP=1 CGC_S1_OUT_LAYERS=1` 跑兩臂 → 各 480 列，但列上印的是 `il=27` 而不是 layer 1，
且 **480/480 全不同**（第一個差異在 `('dec', 0, 27)`）。在弄清那個儀器的參數語意之前，
**它既不能確認也不能否證 §EN-7 的結論**。證據在 `Backup/cgc_logs/outcap_crosscheck/`。
⇒ 下一個人若要獨立確認「layer 1 的輸出相同」，**先讀 `CGC_S1_OUT_LAYERS` 的解析**，
或直接用 `CGC_TENSOR_CAPTURE` 換節點（例如 `ffn_moe_down-2`）做同構的第二個讀數。

**(c) 方法學備註**：兩臂都跑在 HEAVY（launch=HEAVY(2)）。對**雜湊比對**這不影響（決定性），
但別把它當吞吐量資料引用。
