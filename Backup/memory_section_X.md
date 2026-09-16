
### §X 使用者裁定：後續 decode 量測統一用 llama-bench（2026-09-16 18:25）

**指令原文：「後面統一用 llama bench 來測量」**——即結束「兩個儀器各自出數字、被並排引用」的狀態。
本節記裁定本身與**隨它而來的三個必要條件**；未決的部分標明未決，不要當成已定。

它買到什麼：單一儀器 ⇒ 跨 build／跨機器／跨日可比；`d3cbe9744` 立的「llama-bench 是生產標準」
對 decode 這條線才真的成立；也直接解決 §W 開頭那個「兩數字並排」的問題。

**三個必須跟它一起定的細節（不做，標準就是有偏的標準）**

1. **報哪個數。** llama-bench 的 `avg_ts` 把**冷的第 1 個 rep** 平均進去
   （n=128：r1=120 ms vs 平台 87–90 ms）⇒ 報 9.4，而平台是 **11.3 t/s**。
   統一的標準必須指名：`-r` 幾個 rep、warmup 規則、報 avg 還是平台。
2. **散熱讀數入列。** `llama_bench_matrix.py` 現在有 `Sampler`（背景 0.5 s 一次）並把
   `thermal.launch` 寫進列；統一之後這一欄要變成必讀。本輪的 llama-bench 側
   NOMINAL 9.32/9.38 → HEAVY 8.75，但那一臂同時換了模型（MTP=1）⇒ 那個 −7% **未歸因**，
   不能拿來當「llama-bench 對熱壓的係數」。
3. **MTP 例外（未決）。** `llama-bench.cpp` 裡 `sampler|speculat|draft|MTP` **零命中**，
   token 是 `std::rand() % n_vocab`；投機迴圈在**伺服器**（`server-context.cpp` 的
   `slot.spec_draft` / `n_accepted`）⇒ **llama-bench 在結構上量不到 MTP 的加速**。
   目前 MTP 是 1.8× 淨損失且不過 `plain_match`，所以「現在不需要它」是自洽的；
   但 **MTP 一旦復活，它必須回到 HTTP 路徑**。這條註記要寫在標準旁邊，不是寫在人的記憶裡。

**尚未做**：把上面 1／2 寫進 `llama_bench_matrix.py` 的預設值（`--reps` 預設、冷 rep 警告、
發射讀數可否決可比性）。等使用者選定「報哪個數」再改，因為改預設值會讓過去的數字失去可比性
（09-15 的 9.16/9.45 是 `-r 3`）。
