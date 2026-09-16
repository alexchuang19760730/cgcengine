# harness_engine — prime-agent 的第二份 Continual Harness 狀態

這是 engine loop 的 harness 狀態目錄，等於 `PRIME_AGENT_CODING_AGENT_DIR`。
`tb_loop/harness/` 是任務迴圈的那一份；兩者**形狀相同、內容不同**（PLAN §6.1）。

```
harness_engine/
├── extensions -> ../../tb_loop/harness/extensions     ← 符號連結
├── build_memories.py                                  ← 產生器
└── memories/engine/<lesson_id>.md                     ← 106 檔，衍生物
```

## 三件動手前要知道的事

**1. `memories/engine/` 是衍生物，不要手改。**
它由 `traces/lessons.jsonl` 產生。手改會在下一次重生時被蓋掉，而**那次重生看起來完全正常**
— 這是本專案反覆出現的失效形狀。要改內容請改 record，再重跑：

```sh
python3 agent_harness/engine_loop/harness_engine/build_memories.py          # 重生
python3 agent_harness/engine_loop/harness_engine/build_memories.py --check  # 漂移則 exit 1
```

`--check` 順便會驗兩件事：`extensions` 的符號連結還在，而且它**沒有**變成真目錄。

**2. `extensions/` 是指向，不是複本——這是刻意的（CONVENTIONS R1）。**
provider extension（`gemma4-provider.ts`）只能有一份。複本會與 `tb_loop/harness/extensions/`
分叉，而分叉的失效方式是安靜的：harness 照常運作，只是用的是舊的端點定義。
`build_memories.py --check` 會在它變成真目錄時報錯。

⚠️ 已知邊界：**符號連結在打包／rsync 時容易斷**（PLAN R2 對同一手法說過一樣的話）。
`SNAPSHOT` 式的傳輸（`agent_harness/scripts/import_harness_snapshot.py`）沒有這個問題，
但它不搬 `extensions/`。要在另一台機器上重建時，請用 `ln -s` 重建這個連結，
不要 `cp -r`。

**3. 每個檔案的第一行固定是 `[engine] <rule>`。**
那是 `/refine --global` 分辨 scope 的唯一依據（`tb_loop` 那一份沒有這個前綴）。
第一行之下是 `class`、`because`、`counterexample_observed`、`applies_to`——全部逐字取自 record，
不是改寫。**沒有 YAML frontmatter**：prime-agent 的 memory 格式是純 markdown，
而 scope 標記由那個前綴承擔（PLAN §6.1）。

## 誰在讀它

- `distill/refine_engine.sh` 執行時把它設成 `PRIME_AGENT_CODING_AGENT_DIR`
- `distill/closed_loop.py` 的 `D_head_mem` 臂把每一條的**第一行**（即 rule 本體）注入 prompt，
  用來量「注入 lesson 有沒有改變決策」
- 人：想知道這個 repo 累積了什麼規訓時，讀 `traces/lessons.jsonl`（權威），不是讀這裡
