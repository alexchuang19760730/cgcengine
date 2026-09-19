# ④ O1 `CGC_PREFILL_PROTECT` 重跑（2026-09-19 01:46–01:57）—— 尸体解剖与「不要再跑第二遍」的结论

## TL;DR

这轮**没有结论**，而且它是**多余的**：同一件事 2026-09-13 已经用更严谨的设计做过，答案写在
`docs/PREFILL_PROTECT_AB_REPORT_2026-09-13.md`（keep OFF）。我今天重跑时**没有挂 rig**，
所以连「臂到底有没有翻」都没有证据——一个无法证明自己也生效的实验，其「没差异」是不可解读的。

从尸体里捞到的**真新知**有三条，都不是关于 PREFILL_PROTECT 的：

1. **acceptance 是确定性的**：同一 prompt + greedy，6/6 request 逐位相同
   `0.53061（78 accepted / 147 generated），mean len = 2.59`。⇒ 之前记的 0.46541 是**别的 prompt**，
   而 ① 的问题要改写成「是不是**系统性**偏差」——随机不一致类的 bug 已被这 6 个数排除。
2. **每个 request 的 decode 都在 post-prefill 暂态里**：实测 6.38–10.69 t/s，正好落在源码注释说的
   degraded band（8.1–8.9 t/s），远低于 steady 22.2 t/s。⇒ `n_predict=128` 走不完暂态，
   **所有 server 侧 t/s 都被这段暂态污染**。
3. 同一窗口里另一条线在 01:53 重建 `llama-bench`；01:46 起跑时 `run_server.sh` 的
   `MTPDBG mtp_ctor` 闸门报警（现在 `strings -a` 复检是 3/3）⇒ 那是**重建竞态**，不是真的缺 `-DMTP_SUPPORT`。

## 1. 发生了什么

`Backup/phase_decomp/prefill_protect_ab.py prod25 2`（ABBA × 2 block + 1 warmup = 9 reps）
在第 7 个 request 的 prefill 中途被杀（日志停 `Backup/cgc_logs/llama_server_20260919_014615.log`
01:57，`n_tokens=952/2024`，无 graceful 收尾、无 SIGTERM 标记）。**没有写出 JSON**，
但 server 自己的 `slot print_timing` 留下了 6 个完整 request。

臂↔request 映射（由脚本的 `seq` 推定，非日志实证）：

| rep | task | arm | prefill t/s | decode t/s | ms/tok | acceptance | mean_len |
|---|---|---|---|---|---|---|---|
| 1 | 0 | off（warmup，丢弃） | 32.82 | 10.69 | 93.57 | 0.53061 | 2.59 |
| 2 | 304 | **off** | 29.25 | 8.91 | 112.28 | 0.53061 | 2.59 |
| 3 | 608 | **on** | 23.12 | 8.64 | 115.71 | 0.53061 | 2.59 |
| 4 | 912 | **on** | 25.61 | 9.20 | 108.69 | 0.53061 | 2.59 |
| 5 | 1216 | **off** | 25.19 | 8.69 | 115.09 | 0.53061 | 2.59 |
| 6 | 1520 | **off** | 22.93 | 6.38 | 156.81 | 0.53061 | 2.59 |
| 7 | 1824 | on | 未完成（被杀） | — | — | — | — |

## 2. 为什么它不可解读（三条，各自致命）

**(a) 旋钮没有可观测证据。** 日志里 `grep -i protect` = **0 行**。09-13 那份是因为挂了
`CGC-RIG-SNAPSHOT` 才拿到 `defer_skip`（OFF 全 0 × 9、ON 121 668–154 017 × 9）来证明
「臂真的翻了」。我今天没挂 rig ⇒ engaged 与否未知。**no effect 与 not engaged 同形。**

**(b) 漂移远大于效应。** 线性拟合 decode t/s vs rep：**−0.619 t/s per rep**，6 个 rep 掉 3.09 t/s
（**1.68×**）；prefill 同时 32.82 → 22.93（**1.43×**）。臂内原始 spread：off 1.40×、on 1.06×。
这与 09-13 Finding 1 完全同形（「Even from a clean start with flat swap, 109 → 166 ms/tok」），
即**这是该指标的固有属性，不是今天运气不好**。

**(c) 序列设计有偏。** 脚本的 `seq` 是 `["0","1","1","0"] × 2`，**A 永远是 off**（没有做 unit 间
A/B 角色互换）。在单调下滑的漂移下，off 出现在位置 1/4/5、on 在 2/3 ⇒ **估计偏向「protect 有效」**。

漂移校正后的点估计：**resid(on) − resid(off) = +0.25 t/s（≈ +2.9%）**，n=2 vs 4，残差 SD ≈ 0.8 t/s。
⇒ 与 09-13 的判决「no detectable wall-clock effect（median −2.4 ms/tok, p=0.31）」**相容**，
且本设计连正负号都判不出来。

## 3. 与 09-13 结论的对照（同一机制，两套仪表的差别）

| | 09-13（有 rig） | 09-19（无 rig） |
|---|---|---|
| 证明 engage | ✅ `defer_skip` 0 vs 12–15 万 | ❌ 无 |
| 单位/total | reads −4.05 %、misses −4.24 %，4/4 paired unit | — |
| wall clock | median −2.4 ms/tok，p=0.31 ⇒ 无效 | +2.9 %，不可判 |
|  graceful SIGINT（teardown 才印） | ✅ | ❌ 没有 SIGTERM/SIGINT 痕迹 |
| prompt | 单一重复 prompt（最有利 caveat） | 同一份 `http_duo` 单位串，**同样踩这个 caveat** |

⇒ **这个实验没有在方法论上推进任何东西，只把它的失败模式重演了一遍。**

## 4. 对队列的影响（建议重排）

- **④ 关掉，不再排窗口。** 唯一还能排的是 09-13 自己点名的 open question：
  **rotating prompts** 下 reads/misses 的 −4 % 是否存活——但那必须用 rig + graceful SIGINT 重做，
  且它是 vs *reads*，不是 vs *wall clock*，所以对 S 仍无贡献。
- **① plain_match 变便宜了，升为唯一第一优先。** 今天这条「6/6 逐 rep 完全相同」直接消除了
  「两次跑本来就不同」这一层借口：同一 prompt + greedy 下，batch-verify 的 logits 与逐 token 解码的
  logits **必须**逐位相同；不相同就是 bug，且位置立刻可读。**这是把 ① 从「0.5–1 天只读」压到「一支跑」的关键。**
- **引用纪律追加**：server 侧 decode t/s 一律要标「post-prefill 暂态」或把 `n_predict` 拉到 ≥512 并只报尾段。
  在改之前，`http_duo` 的 t/s 不能与 llama-bench 并列引用（两侧取样位置不同，见 §EN-190）。

## 5. 复现命令（若要重做；先确认窗口干净）

```sh
lsof -nP -iTCP:8080 -sTCP:LISTEN        ;# 必须空
pgrep -fl 'llama-bench|llama-server|spec_cost_curve'   ;# 必须空
strings -a src/llama.cpp/build/bin/libllama-common.0.dylib | grep -c 'MTPDBG mtp_ctor'   ;# 必须 ≥1
# 额外：02 env
CGC_PREFILL_PROTECT_FILE=<flag>  CGC_RIG_SNAPSHOT=1     ;# rig 见 llama-expert-cache.cpp:405 起
# 收尾要 SIGINT（不是 SIGTERM），否则 teardown 计数器被丢（09-13 Finding 1 陷阱 2）
```

## Artifacts

- server log：`Backup/cgc_logs/llama_server_20260919_014615.log`
- 启动 wrapper log：`Backup/phase_decomp/pp_ab_prod25_20260919_014614.server.log`
- 驱动：`Backup/phase_decomp/prefill_protect_ab.py`（**缺 rig，勿再直接用**）
- 对照结论：`docs/PREFILL_PROTECT_AB_REPORT_2026-09-13.md`
