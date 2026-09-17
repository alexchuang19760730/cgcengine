# 三種 record 的權威定義在哪裡

這份文件**不是** schema 本身。它是 PLAN §3 替 `shared/` 安排的三個檔案之一，用途只有一個：
指出權威在哪、以及為什麼不在這裡。

## 權威

| record | 權威 | 台帳 |
|---|---|---|
| episode | `engine_loop/traces/schema/episode.schema.json` | `engine_loop/traces/episodes.jsonl` |
| decision | `engine_loop/traces/schema/decision.schema.json` | `engine_loop/traces/decisions.jsonl` |
| lesson | `engine_loop/traces/schema/lesson.schema.json` | `engine_loop/traces/lessons.jsonl` |

**`engine_loop/traces/validate.py` 是唯一能說「這批 record 合法」的東西。** 它比對的是上面三個
schema，不是這份文件。

## 為什麼這裡不重述一遍

把三個 schema 的欄位再抄一份到這裡，就製造了第二個權威。第二個權威的失效方式是安靜的：
schema 改了、這份文件沒改，而**兩者都還讀得通**——只有照著這份文件寫 record 的人會錯，
而他們錯的方式是 `validate.py` 拒絕，不是這份文件自己指出來。

同一條規則在 `CONVENTIONS.md` R1 已經寫過一次（`scripts/check/*` 不得複製進 `engine_loop/`）。
這裡只是把它套在文件上。

## 要改 schema 的時候

1. 改 `engine_loop/traces/schema/*.schema.json`
2. 跑 `python3 agent_harness/engine_loop/traces/validate.py` —— 既有 record 會立刻告訴你破壞了什麼
3. 跑 `python3 agent_harness/engine_loop/traces/selftest.py` —— 它會注入違規紀錄並要求被拒絕
   （沒有這一步，「驗證器沒有擋下東西」與「驗證器根本沒看」是同一個輸出）

## 三者的關係（一句話各一）

- **episode** 是一個臂產出的觀測，帶 `build` 指紋與 `usable_as_evidence`／`caveats` 兩個引用閘門。
- **decision** 是一個判斷：`question` ＋ 它讀過的 `evidence` ＋ `ruled_out` ＋ `judgement`（含 `refuted`）。
- **lesson** 是一條從付過代價的觀測一般化出來、換一個問題也適用的規訓，`superseded_by` 取代刪除。

衍生品（`harness_engine/memories/engine/`、`sft_pi/`、`sft_prime/`）全部由這三份檔推導，
且各自有 `--check`。**手改衍生物會被下一次重生蓋掉，而那次重生看起來完全正常。**
