# rswaengine

**R-SWA + TrueOrthoKDA（窗口作用域正交 KV 压缩器）统一引擎**

自包含 C/C++ 引擎 + Python 桥接，实现「无限长对话 / 恒定 KV 显存」目标：

- **R-SWA**（Reference Sliding Window Attention）：Reference 永久区（标准注意力）+ 滑动窗口环形 KV（恒定上限），产出可见性掩码，专司 *recency / 可见性管理*。
- **TrueOrthoKDA（重构为窗口作用域压缩器）**：在 R-SWA 窗口内部做 Gram-Schmidt 局部正交基压缩，提供 *O(1) 传输表征 + 数值稳定注意力路径*，Reference 绝不进压缩基（硬约束）。
- **`inject_unified_ir_for_role`**：统一注入入口，以 backend 无关 IR 描述逐层角色，并**真正联动** `cgc_inject_strategy`（注入即自动启用 KDA replace mode）。

> 设计原理、风险与 MVP 顺序见 [`docs/CGC_TrueOrthoKDA_RSWA_Unified_Technical_Whitepaper_v1.0_zh_CN.md`](docs/CGC_TrueOrthoKDA_RSWA_Unified_Technical_Whitepaper_v1.0_zh_CN.md)。

---

## 目录

```
rswaengine/
├── docs/                        白皮书（设计 + 已落地状态）
├── cpp/                         C/C++ 引擎
│   ├── include/                 cgc_cpp.h / unified_ir.h / kernels/*.h
│   ├── src/                     cgc_cpp.cpp / unified_ir.cpp / kernels/*.cpp|.cu
│   ├── examples/                main_unified.cpp / main_engine.cpp
│   ├── build_unified.sh         CPU / C++17 构建（产出 libcgc_unified.so）
│   └── build_unified_cuda.sh    nvcc 构建（CUDA kernel）
└── python/                      ctypes 桥接 + sglang/vllm adapter
    ├── cgc_unified_injection.py
    ├── sglang_adapter.py
    └── vllm_adapter.py
```

---

## 构建与运行

### C/C++（无需 CUDA / torch）

```bash
cd rswaengine/cpp
bash build_unified.sh
```

产出：
- `build/libcgc_unified.so` —— 供 demo 与 Python ctypes 共用
- `build/cgc_unified_demo` —— R-SWA + 窗口 OrthoKDA 功能演示
- `build/cgc_engine_demo` —— `cgc_inject_strategy` ↔ 统一 IR 联动验证（SDPA→KDA 重路由）

### CUDA kernel（需 nvcc；本机无则跳过）

```bash
cd rswaengine/cpp
bash build_unified_cuda.sh   # 编译 window_ortho_kv_compressor.cu -> libcgc_unified_cuda.so
```

### Python 桥接与 adapter（需 numpy；torch 仅真实 patch 时需要）

```bash
cd rswaengine/python
python3 cgc_unified_injection.py   # ctypes 自检：联动 + RSWAManager 计算
python3 sglang_adapter.py          # sglang 消费路径（numpy 独立 demo）
python3 vllm_adapter.py            # vllm 消费路径（numpy 独立 demo）
```

真实接入 sglang / vllm 时，调用 `sglang_adapter.patch_sglang(model, config)` /
`vllm_adapter.patch_vllm(model, config)`（需对应框架 + torch，并以目标版本注意力 API 为准落地）。

---

## 联动关系（核心）

```
cgc_inject_unified_ir_for_role(role, model, config, adapter)
   └─> 构建 cgc_strategy_t（metadata 含 "unified_ir:..."）
         └─> cgc_inject_strategy(&strat)            ← 联动①（unified_ir.cpp 回调）
               └─> 检测到 "unified_ir:"  → cgc_set_kda_replace_mode(true)  ← 联动②
                     └─> cgc_execute_opcode(0x10) -> 0x11(KDA) 重路由生效
```

---

## 当前状态（诚实声明）

| 组件 | 状态 |
|---|---|
| R-SWA（C/C++） | ✅ 已实现（`rswa_manager.cpp`） |
| 窗口作用域 OrthoKDA（C/C++ CPU） | ✅ 已实现（`window_ortho_kv_compressor.cpp`） |
| `inject_unified_ir_for_role` + `cgc_inject_strategy` 联动 | ✅ 已实现并验证 |
| sglang / vllm Python monkey-patch adapter | ✅ 模板已提供；真实 patch 需框架 + torch 落地 |
| 窗口压缩器 CUDA kernel | ⚠️ 已编写（对齐 v4 风格），待 GPU 环境编译验证 |
| TrueOrthoKDA v4（既有 CUDA/CPU） | ✅ 已包含（参考 / 复用） |

**已撤销的夸大宣称**：256x 压缩、CGC env var 一键开关（白皮书 §〇/§十 已标注）。
