---
name: cgc-tb-smoke
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）跑 terminal-bench 的鏈路 smoke —— 不需要模型、不需要量測窗，用來證明 tb_loop 的套件重構沒有壞（E1 第 3 條驗收）。當要在該 repo 跑 `tb run`、或 `tb run` 失敗要診斷（`unknown command: docker compose`、`docker-credential-desktop: not found`）、或要確認 tb_loop 的 console script 進入點可用時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-tb-smoke/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# flashkv-devserver 的 tb smoke（不需要模型）

repo：`/Users/alexchuang/Documents/flashkv-devserver`。相關目錄：`agent_harness/tb_loop/`。

**為什麼它便宜**：`tb run` 的 `--agent` 預設是 **`oracle`**（跑 gold solution），所以這條鏈路
**不需要任何模型端點**。它驗的是：dataset → docker build → 容器啟動 → 跑 solution → 跑測試 → 寫 `results.json`。
2026-09-17 實測 **1/1 resolved**（`raman-fitting.easy`）。

## 指令

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
PYTHONPATH=agent_harness agent_harness/tb_loop/.venv/bin/tb run \
  -d "terminal-bench-core==0.1.1" --n-tasks 1 \
  --output-path /tmp/tb_smoke --run-id e1_smoke
```

- **入口是 venv 的 `tb` console script**，不是 `python -m terminal_bench.cli.tb`
  （後者在這個版本會 `'terminal_bench.cli.tb' is a package and cannot be directly executed`；
  `tb_loop/README.md` 原本就寫錯，已修）。
- `PYTHONPATH` 要指 **`agent_harness/`** —— `tb_loop` 才是那裡的套件名。
- 輸出寫到 `/tmp`，不要寫進 repo（`results/` 會多出未追蹤檔）。約 1–3 分鐘（映像有快取時更快）。

期望的尾端輸出：

```
Results Summary:
| Resolved Trials   | 1       |
| Accuracy          | 100.00% |
Results written to <out>/e1_smoke/results.json
```

## 前置（缺一不可）—— 2026-09-17 三次失敗才通

```bash
colima start --memory 4 --cpu 2          # 只用 4 GiB，別用 profile 的 8 GiB（本機 16 GB）
docker ps >/dev/null && echo daemon OK
docker compose version                   # 一定要印出 Docker Compose version …
```

**本機的坑（Docker Desktop 已被移除，殘骸還在）**：

| 症狀 | 原因 | 修法 |
|---|---|---|
| `docker: unknown command: docker compose`（exit 125） | `~/.docker/cli-plugins/` 底下 **16 個連結全斷**（都指向已不存在的 `/Applications/Docker.app`），docker **靜默忽略**斷連結 | `brew install docker-compose`，再把 `~/.docker/cli-plugins/docker-compose` 指到 `/opt/homebrew/lib/docker/cli-plugins/docker-compose` |
| `error listing credentials … docker-credential-desktop: not found` | `~/.docker/config.json` 的 `credsStore: "desktop"` 指向同一個消失的 app（`auths` 本來就是空的 ⇒ 移除該鍵不損失憑證） | 備份後移除 `credsStore` |

偵測斷連結：`for f in ~/.docker/cli-plugins/*; do [ -e "$f" ] || echo DANGLE "$f"; done`
（`[ -e ]` 對斷連結是 false；而 `ls` 用 lstat，**照樣把它列出來**，所以「ls 看得到」不代表它活著。）

## 動手前／收工後（本 repo 的硬規矩）

- **空窗要讀日誌，不是讀 `pgrep`**：行程掃描是空的**不代表**沒人在量測。
  要先看 `<repo>/.workbuddy/memory/YYYY-MM-DD.md` 的最新段落（另一條線一天做 20–29 輪，
  取樣空窗可能只有幾十秒）。這是同一個陷阱踩過兩次。
- **用 repo 已經有的閘門，不要手寫**：`bash agent_harness/engine_loop/runners/preflight.sh`
  （`--self-test` **5/5 含正負對照**：命令列提到名字的 wrapper 被**排除**、真正以那個名字執行的行程
  被**認出**）。★ 我手寫過 `ps -Ao comm= | grep -x llama-server` —— **在 macOS 上 `comm` 不是
  basename 而是完整路徑**（`/Users/.../build/bin/llama-server`），所以它**永遠不命中**：
  那個閘門回報的「乾淨」與「有、但我比對錯欄位」長得一模一樣（lesson `eng-gate-0054`）。
  用 `pgrep -f`／`grep command` 時要寫成 `[l]lama-server` 這種自我排除形式，否則會命中**自己的命令列**。
- 檢查與動作要在同一個分支裡（`if … ; then exit; fi` 緊接動作）。
- **動別的東西之前先問**：起 llama-server（13 GB）或任何服務，即使「機器看起來很空」也一樣 ——
  這一輪就是先被叫去跑、兩分鐘後被叫停。工作目錄一律是 `flashkv-devserver`；
  `flashkv0516` 是它的**主 repo**（worktree 關係），踏進去要另外授權。
- 收工：`colima stop`，並把 `~/.colima/default/colima.yaml` 的 `cpu`／`memory` 改回 `4`／`8`。
- **macOS 沒有 `timeout`**；直接跑，或用 `gtimeout`。

## 失敗時怎麼讀

- terminal-bench 只把 `CalledProcessError` 的 traceback 丟出來，**看不到 compose 的 stderr**。
  要手動重現那個 build：

```bash
D=~/.cache/terminal-bench/terminal-bench-core/0.1.1/<task>
T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME=tb-client \
T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME=tb-client-1 \
T_BENCH_TEST_DIR=/app T_BENCH_TASK_LOGS_PATH=/tmp/tblogs T_BENCH_CONTAINER_LOGS_PATH=/logs \
docker compose -p diag -f "$D/docker-compose.yaml" build
```

  沒有那 5 個 `T_BENCH_*` 變數會先死在 `container_name '' does not match pattern` ——
  那是**你的重現不忠實**，不是真缺陷。
- exit **125** ＝ compose 指令本身沒跑起來（外掛／設定）；exit **1** ＝ compose 跑了但失敗（憑證／Dockerfile）。
