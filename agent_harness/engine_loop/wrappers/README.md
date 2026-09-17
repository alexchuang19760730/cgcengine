# wrappers/ — `scripts/check/*` 的呼叫入口（PLAN §3）

這一層**不放邏輯**。它的工作是：把 44 支生產腳本按能力分層、檢查前置、
跑完之後留下「這次 run 是哪一批二進位檔跑的」的痕跡。

```sh
./sweep.sh  --list            # 這一類有哪些腳本（含 MANIFEST 的 role 與它自己的說明）
./sweep.sh  --self-test       # 分類表與磁碟一致嗎（4 項）
./sweep.sh  --dry-run <script> [args...]   # 只印出將執行的命令，不執行
./sweep.sh  <script> [args...]             # 執行，落一份 run manifest
```

| wrapper | 能力 | n | 它回答的問題 |
|---|---|---|---|
| `sweep.sh` | 對一個旋鈕做參數掃描，逐格重啟伺服器並記一列 | 4 | 「把 X 從 2 掃到 8，曲線長什麼樣」 |
| `ab.sh` | 對兩份設定／輸出做對照 | 9 | 「這兩者有沒有差」 |
| `bench.sh` | 產生吞吐量或品質的基準讀數 | 9 | 「現在多快」 |
| `gate.sh` | 判對錯 | 10 | 「合不合格、可不可比較」 |
| `triage.sh` | 對既有警報或讀數做鑑識 | 9 | 「這個數字為什麼長這樣」 |
| `env.sh` | 環境與服務健康檢查 | 3 | 「在啟動任何東西之前，環境有沒有壞」 |

## 分類是資料，不是目錄

`classes.tsv` 是產物、`classify.py` 是權威。**不要手改 `classes.tsv`。**

```sh
python3 classify.py            # 重新產生 classes.tsv
python3 classify.py --check    # 磁碟與分類表一致嗎（不一致就 exit 1）
python3 classify.py --stats    # 各類大小 ＋ MANIFEST role 的退化程度
```

## 為什麼分層不做成 `git mv`（E4 item 3 的落地形式）

實測：`scripts/check/*` 的 44 支腳本承載 **562 條引用關係** ——
`agent_harness/` 220、`docs/` **134**、`Backup/` 101、`scripts/` 62、`.workbuddy/memory/` **41**。

其中兩類**不可改寫**：

- `docs/*.html` 是**已定稿的白皮書**。把引用改成新路徑等於讓一份已發表的文件說假話，
  而本 repo 對這件事的立場是「標註而不是改寫」。
- `.workbuddy/memory/*.md` 是 **append-only 的日誌**，描述的是**當時**的狀態。

所以 `git mv` 的代價是 **175 條引用懸空**，收益只是目錄好看一點 ——
而且 PLAN §3 的紅線本來就站在「`scripts/check/*` 是生產腳本」那一邊。

於是分層的座標改成「它**做什麼**」而不是「它**在哪個資料夾**」，並且**可機檢**。

## 為什麼需要它：`role` 那一欄在這個目錄上是退化的

`MANIFEST.jsonl` 已經有一個分類欄，但實測它在 `scripts/check/*` 上幾乎不帶資訊：

```
role 分佈（44 支）：probe 34、gate 4、measure 3、compare 2、arms 1     ⇒ probe 佔 77%
```

而 44 支**全部可執行**，所以「可不可執行」也不是區分軸。
換句話說 `probe` 這個標籤無法回答「我要做 X 該跑哪一支」—— 能力類是第二個座標。

## 每次 run 的產物（`runs/`）

```
runs/<ts>_<class>_<script>/
├── run.log      # 完整輸出，檔頭是 provenance
└── run.json     # {wrapper, script, argv, rc, seconds, git_head,
                 #  build_fingerprint_keys, build_fingerprint_sha16, log, not_a_record}
```

**`runs/` 不納管**（`.gitignore`），性質與 `Backup/cgc_logs`、`harness_engine/logs/` 相同：
它們是某一次 run 的原始輸出，會無限增長，而**可引用的部分已經被正規化進 `traces/`**。

### 兩個刻意的「不做」

- **不把輸出轉成 episode record。** 那是 `traces/emit_episodes.py` 的工作（PLAN §5 T0）。
  wrapper 若自己再轉一次，同一個 episode 就會有兩個生產者，而它們會各自演化 ——
  於是「哪一份是對的」變成一個沒人能答的問題。
- **不要 `build.json` 就不跑。** 沒有 build 指紋的讀數在這個 repo 裡不可引用
  （`emit_episodes.py` 對 `build == null` 的列直接標 `usable_as_evidence: false`），
  所以缺它是**硬前置**而不是警告。明知故犯要 `ENGINE_ALLOW_NO_FINGERPRINT=1`。

## 與 `runners/` 的關係

`wrappers/` 呼叫**量測**腳本（會產生數字）；`runners/` 負責**機器狀態**
（`rebuild.sh` 產指紋、`server.sh` 起服務、`preflight.sh` 回答「現在能不能起」）。
一次正經的量測順序是：

```sh
runners/rebuild.sh                    # 1. 建置，閘門會 abort
runners/rebuild.sh --check            # 2. 確認二進位檔就是你以為的那一批
runners/preflight.sh                  # 3. 機器乾淨嗎
runners/server.sh --profile prefill250 --ctx 40960 --detach   # 4. 起服務
wrappers/bench.sh decode_bench.py ... # 5. 量
```

★ `.gitignore` 的 `build.json` 是**刻意**不納管的：它描述的是某一台機器在某一次的建置狀態，
不是一個決策。每個 episode 自己帶 `build` 欄位，所以跨機器要傳的是 episode 而不是這個檔。
