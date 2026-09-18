---
name: github-transport-fallback
description: 在 GitHub 推送／拉取失敗時，判斷是「憑證問題」還是「傳輸通道問題」，並選對路走（SSH 優先、Git Data API 為最後手段）；以及「遠端在工作期間被別的 agent 推進」時的正確處理流程。當出現 `git push` 被拒、non-fast-forward、`github.com:443` timeout、`could not read Username`、`gh auth status` 失敗、或需要在不改動 remote 設定的情況下推送時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/github-transport-fallback/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# GitHub 推送失敗時：先量通道，再選路

## 核心判準

**「push 失敗」有五種完全不同的原因，而它們的訊息長得很像。** 不要一看到失敗就去改憑證。

| 症狀 | 真正的原因 | 走哪條 |
|---|---|---|
| `Failed to connect to github.com port 443 after 75000 ms` | **通道被擋**（中國大陸網路常見） | 換 SSH（見下） |
| `non-fast-forward` / `hint: Updates were rejected` | **遠端被別人推進了**（多 agent 環境常態） | fetch → 看重疊 → rebase |
| `could not read Username` / `Authentication failed` | 憑證不在 git 找得到的地方 | 見「憑證」那節 |
| `Permission denied (publickey)` | SSH key 沒設好或不是這個帳號 | 回到 HTTPS 或修 key |
| `everything up-to-date` 但遠端沒動 | 推錯分支／推錯 remote | `git ls-remote <url> <branch>` 對 sha |

## 第一步：一次量完三個出口（各一次、短逾時，不要重試迴圈）

```bash
curl -s -o /dev/null -w "github.com:443 HTTP %{http_code}\n" --max-time 12 https://github.com/
ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | head -2
curl -s -o /dev/null -w "api HTTP %{http_code}\n" --max-time 12 https://api.github.com/
```

**實測（2026-09-18，中國大陸）**：`github.com:443` **timeout**；`ssh -T git@github.com`
回 `Hi <user>! You've successfully authenticated`；`api.github.com` **200**；
`codeload.github.com` **301／tarball 200**。⇒ **同一台機器、同一個帳號，三個出口的可用性不一致**。

## 第二步：SSH 通就用 SSH，而且**不要改動 remote 設定**

用一次性 URL，避免污染使用者 repo 的 `.git/config`（也避免把 token 寫進 URL）：

```bash
URL="git@github.com:<owner>/<repo>.git"
git -c core.sshCommand="ssh -o ConnectTimeout=20" fetch "$URL" main:refs/remotes/origin/main
git -c core.sshCommand="ssh -o ConnectTimeout=25" push  "$URL" HEAD:main
```

★ 若 repo 慣用 HTTPS remote，**保持它不動**：上面的寫法不寫入任何設定，下次網路好了照樣能用。

## 第三步：遠端被別的 agent 推進時（**推之前一定要驗**）

多個 agent 同時對同一個 repo 全權寫入是常態。**`non-fast-forward` 不是憑證問題。**

1. **取回**：`git fetch <url> <branch>:refs/remotes/origin/<branch>`。
2. **看對方改了哪些檔、與自己有沒有重疊**（拉不到時用 API，不需要 clone）：
   ```bash
   curl -s -H "Authorization: token $TOKEN" \
     "https://api.github.com/repos/<owner>/<repo>/commits/<sha>" \
     | python3 -c "import json,sys; d=json.load(sys.stdin); print([f['filename'] for f in d['files']])"
   ```
   **有重疊就先看內容再決定**（可能是真衝突，也可能是對方也在改同一份文件）。
3. **零重疊才 rebase**：`git rebase origin/<branch>`。
4. ★ **rebase 之後重跑自己的自測**（rebase 可能把別人的改動帶進你依賴的檔案）。
5. 推。**只用 `HEAD:main`，不要 `--force`。**

## 最後手段：Git Data API（等同 `git push`）

SSH 也不通時，用 REST API 建 commit：`blob → tree → commit → PATCH /git/refs/heads/<branch>`。

- `POST /repos/{o}/{r}/git/blobs` 帶 `{"content": base64, "encoding": "base64"}`（**讀 bytes，不要用 text mode**）。
- `POST /git/trees` 帶 `{"base_tree": <parent tree>, "tree": [{"path","mode":"100644","type":"blob","sha"}]}`。
- `POST /git/commits` 帶 `message` / `tree` / `parents` / `author` / `committer`。
- `PATCH /git/refs/heads/<branch>` 帶 `{"sha": ..., "force": false}` ⇒ **`force:false` 是最後一道防線**，
  遠端動了它就會失敗，而不是覆蓋別人。

★ **想讓遠端 sha 與本地逐位元組相同**：把本地 commit 的
`%an %ae %aI %cn %ce %cI` 與**完整訊息**原樣帶上，並讓 `base_tree` 等於 parent 的 tree。
tree 與 commit 都是內容定址 ⇒ **可以 assert 相等**（這比「內容一樣但 sha 不同」乾淨得多，
因為本地與遠端就是同一個 commit）。
★ 若做不到（例如你在本地 commit 之後遠端動了），就**接受 sha 不同**，但要在報告裡明說。

## 憑證：不要因為 `gh auth status` 失敗就以為沒憑證

實測過的誤導：`gh auth status` 顯示帳號 active 但 `Failed to log in … (keyring)`，
而 `~/.git-credentials` 存在但 `github.com` 條目 **0** ⇒ 看起來「完全沒有憑證」。
但實際上 **SSH key 是好的**、**token 也能用**（只是沒放在 git 會找的地方）。

判準：**憑證要分開量**（env 變數／`~/.zshrc`／`~/.git-credentials`／`~/.config/gh/hosts.yml`／
`ssh -T`），不要用其中一個的失敗推論全部。**測試時只印 HTTP 碼與登入名，永遠不要印出憑證值。**

## 安全（每次都做）

- 憑證只放 **`~/.config/<專案>/…` chmod 600**，**不放 repo、不放 `/tmp`、不寫進 commit message**。
- git 指令用 `-c credential.helper= -c http.extraHeader="AUTHORIZATION: basic $(…)"` 時，
  這個 header **不會**寫進 `.git/config`；但**用完要確認 `git remote -v` 沒有把 token 帶進 URL**。
- 使用者在對話裡貼了 token ⇒ **在用完後提醒他撤銷重發**（逐字稿是明文的）。

## 反模式（看到就打斷）

- **一看到 push 失敗就 `git push --force`** ⇒ 會蓋掉別的 agent 的 commit。
- **`git config --global` 改 remote／credential** ⇒ 為了一次推送改掉全機設定。
- **把 token 寫進 remote URL**（`https://user:token@github.com/...`）⇒ 會留在 `.git/config`。
- **重試迴圈**（等 75 秒 × N 次）⇒ 先量通道，再決定路。
- **把「拉不到」當成「沒東西」** ⇒ 用 API 讀 commit 清單確認遠端到底有什麼。

## 邊界

這條 skill 講的是**傳輸**：它不判斷「該不該 commit」「commit 內容對不對」
（那是各 repo 自己的提交閘門，例如 CGC 的 `cgc-commit-gate`）。
