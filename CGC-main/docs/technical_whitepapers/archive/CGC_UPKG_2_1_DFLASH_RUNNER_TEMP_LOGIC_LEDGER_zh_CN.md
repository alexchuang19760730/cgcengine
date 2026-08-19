# CGC UPKG 2.1 DFLASH Benchmark Runner 临时逻辑拆账表

**版本**: v0.1  
**状态**: 工作稿（历史拆账表，现状已更新）  
**定位**: 盘点 `temp/misc/run_host2_upkg21_official_sglang_dflash_benchmark.py` 中为 `host2 + DeepSeek-V4-Flash + DFLASH + Qwen3.5-4B-DFlash` 收敛而引入的临时逻辑，并按 `profile setting + binding / environment bootstrap / model setting-system profile` 给出正式落点；同时补充当前 repo 中 `upkg21` 已实跑通过后的解释口径。

---

## 一、目的

本表只回答三件事：

- runner 里现在有哪些临时逻辑
- 这些逻辑未来应该搬去哪一层正式配置
- 哪些已经开始执行搬迁

---

## 二、已开始执行的搬迁

当前已开始的正式搬迁包括：

- runner 已支持通过 `CGC_HOST2_BENCH_PROFILE_PATH` 从外部 JSON 读取 benchmark shape 与 runtime shape，而不再只能依赖 `_build_config()` 内建默认值
- 已新增 `host2` benchmark 的 `profile_settings` 示例：
  - `docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_profile_settings.example.json`
- 已新增与其回链的 `system_execution_manifest` 示例：
  - `docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_system_manifest.example.json`

这表示拆账表里的第一类任务已经开始从 runner 内部逻辑迁出。

补充当前代码快照：

- `upkg21` 已实跑通过
- 因此本表描述的是 benchmark runner 的历史性临时逻辑拆账，而不是当前 `UPKG 2.1` 主 gate 仍被这些问题卡住

---

## 三、拆账表

| 临时逻辑 | 当前位置 | 当前用途 | 正式层 | 正式落点 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| `contexts / gen_tokens / warmup_runs / runs / timeout` 默认值 | `_parse_contexts()` / `_build_config()` | 决定这次 benchmark 怎么跑 | `profile setting + binding` | `profile_settings.json` + `execution_profile_binding_key` / `flow_parameter_contract_binding_key` | 已开始搬迁 |
| `tp_size / context_length / mem_fraction_static / default_runtime_extra_args` 默认值 | `_build_config()` | 决定本轮 runtime shape | `profile setting + binding` | `profile_settings.json` | 已开始搬迁 |
| `official -> cleanup -> vendored` 顺序模式 | 远端 `REMOTE_PY` 主流程 | 避免双 runtime 叠加污染 | `profile setting + binding` | benchmark execution profile / flow parameter contract | 待搬迁 |
| `detach / fetch_only / local_output_root` | `main()` | 解决长跑与抓取方式 | `profile setting + binding` | benchmark flow parameter contract | 待搬迁 |
| `SM120 -> deepep_mode=normal` | `_build_config()` 的默认 `extra_args` + 实验结论 | 规避 `deep_gemm` 不可用导致的 `DeepEP auto` 失效 | `environment bootstrap` | host2 / Blackwell runtime bootstrap contract | 已开始搬迁 |
| `cpu_offload_gb=4` 的安全收敛值 | `_build_config()` 默认 `extra_args` | 降低 GPU OOM，同时控制 host OOM | `model setting / system profile` | host2 DFLASH benchmark system profile / model contract safe_runtime_shape | 已开始搬迁 |
| `disable fp8 CPU pin-memory staging` | `_patch_fp8_cpu_staging()` + runtime env | 避免 `Fp8MoEMethod` 初始化阶段 host/shmem 压力过高 | `environment bootstrap` | host2 / Blackwell runtime bootstrap contract | 已开始搬迁 |
| `DeepSeek-V4 hook` 放开 DFLASH 人工限制 | `_patch_deepseek_v4_hook()` | 让 `DeepSeek-V4-Flash + DFLASH` 真正可启动 | `model setting / system profile` | DeepSeek-V4-Flash model bootstrap contract | 已开始搬迁 |
| `强制 kill process group + 端口清理 + CUDA cache 清理` | `_cleanup_runtime_processes()` / `_force_cleanup_runtime()` | 避免上一轮残留污染下一轮 | `environment bootstrap` | runtime cleanup / readiness contract | 已开始搬迁 |
| `CGC_SGLANG_DISABLE_FP8_CPU_PIN_MEMORY` env 注入 | `_build_config()` -> `baseline.env/optimized.env` | 为 benchmark 暂时注入安全兼容开关 | `environment bootstrap` | runtime bootstrap capability rule | 已开始搬迁 |
| `official upstream` 与 `vendored` 的 runtime family 区分 | `_build_config()` 的 `runtime_name/python/pythonpath` | 固定比较对象 | `model setting / system profile` | benchmark system manifest / component matrix | 已开始搬迁 |
| `DeepSeek-V4-Flash + Qwen3.5-4B-DFlash + DFLASH + DeepEP` 组合关系 | runner 默认值 + 实验结论 | 确定正式 benchmark 系统是谁在跑 | `model setting / system profile` | `system_execution_manifest.system_profile` | 已开始搬迁 |
| `wqkv_a / legacy_kv / legacy_o_proj` 权重兼容补丁需求 | 当前仍停留在实验分析 | 解决模型能否真正 ready | `model setting / system profile` | DeepSeek-V4-Flash model contract | 已开始搬迁 |

---

## 四、当前优先级

建议按以下顺序继续执行：

1. 先把 benchmark shape 与 runtime shape 继续从 runner 默认值迁到 `profile_settings.json`
2. 再把 `SM120 / deepep_mode / fp8 CPU staging / cleanup` 收为 `environment bootstrap`
3. 最后把 `DeepSeek-V4-Flash + DFLASH` 的 attention 权重契约收为 `model setting / system profile`

原因是：

- 第一层决定“这次怎么跑”，最容易先从 runner 中抽离
- 第二层决定“当前能跑什么”，已经有明确的 host2 / Blackwell 经验规则
- 第三层决定“系统与模型是否真正成立”，在本历史拆账语境下曾经集中暴露于 `wqkv_a` 等权重契约问题；但对当前 repo 主线而言，这已不再是 `upkg21` 的最终 blocker

---

## 五、一句话结论

当前这份历史拆账表所描述的核心问题已经不是“benchmark 参数还没试到”，而是：

- 哪些逻辑属于 `profile setting + binding`
- 哪些逻辑属于 `environment bootstrap`
- 哪些逻辑属于 `model setting / system profile`

只有把 runner 里的临时逻辑按这三层正式拆走，`host2` 上的 `DeepSeek-V4-Flash + DFLASH` benchmark 收敛才不会在每次重跑时重新回到手工试参状态。

若以当前 repo 主 gate 口径来读，则应同时记住：

- `UPKG 2.1` 主验证链已经通过
- 本文关注的是 benchmark runner 的临时逻辑治理，而不是当前 `upkg21` 的正式 PASS/FAIL 结论
