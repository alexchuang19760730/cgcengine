# CGC Engine 两主机差异报告
生成: 2026-07-24 05:58

## 路径差异(关键!)
- Host1: /root/flashkv0516/ComputeGraphCompiler-main/cgc_engine (31M, 1749 files)
- Host2: /root/flashkv0516/cgc_engine (21M, 1122 files)
- **路径不一致** — Host1 在 ComputeGraphCompiler-main 子目录, Host2 直接在 flashkv0516 下

## 汇总(核心档案 .so/.py/.cpp/.h/.cuh/.cu/.sh/.json/.md)
- Host1 有效: 861  Host2 有效: 848
- Host1 独有: 139  Host2 独有: 126  md5不同: 25  相同: 697

## A. md5 不同(25 个, 需确认以哪个为准)
### 核心编译产物(关键!)
- cgc/cgc_cpp/build/cgc_cpp.so  H1=507160B  H2=405336B  ← 本会话配套矛盾根源
- cgc/cgc_cpp/CMakeLists.txt  H1=5646B  H2=4865B
- cgc/cgc_cpp/include/cgc_cpp.h  H1=2242B  H2=2947B
- cgc/cgc_cpp/include/kernels/ortho_kda_v4.cuh  H1=2488B  H2=1635B  ← ortho_kda kernel
- cgc/cgc_cpp/src/binding.cpp  H1=15767B  H2=18071B
- cgc/cgc_cpp/src/cgc_cpp.cpp  H1=27919B  H2=28861B
- cgc/cgc_cpp/src/kernels/ortho_kda_v4.cpp  H1=9566B  H2=6487B
- cgc/cgc_cpp/src/cgc_cuda_backend.cpp  H1=4889B  H2=4813B
- cgc/cgc_cpp/src/anti_fraud.cpp  H1=11993B  H2=11953B
- cgc/cgc_cpp/src/magi_backend_unified.cpp  H1=17077B  H2=17073B

### Python 核心(功能差异)
- cli.py  H1=216917B  H2=39672B  ← Host1 大5倍(可能是新版)
- pipeline.py  H1=295675B  H2=290823B
- config.py  H1=11908B  H2=13857B
- prefill_pool/prefill_pool.py  H1=13824B  H2=13190B
- product/m76_gate.py  H1=79930B  H2=82243B
- product/m75_trueorthokda_active_runtime.py  H1=49996B  H2=48544B
- product/upkg21_gate.py  H1=40525B  H2=38571B
- agent/llm_auto_pipeline.py  H1=309643B  H2=311111B
- agent/trainers/__init__.py  H1=4605B  H2=4204B
- bridge/megatrain_vllm_bridge.py  H1=22319B  H2=29819B
- pd/dopd_schema.py  H1=5004B  H2=4522B
- tools/scripts/run/gate_test_framework.py  H1=83566B  H2=23697B  ← Host1 大3.5倍
- prefill_pool/__init__.py  H1=0B  H2=164B
- product/__init__.py  H1=2407B  H2=2264B
- rswa_integration/__init__.py  H1=0B  H2=205B

## B. Host1 独有(139 个, Host2 缺)
### 重要目录(Host1 独有整个目录)
- cli_universe/* (15 个文件, agent_benchmarks/agent_model/engine/fusionroute_agent 等)
- cpp/cgc_moe_engine/* (6 个, CMakeLists/bindings/cgc_moe_engine.cpp/.h/__init__) ← MoE engine
- gate_verifiers/* (25+ 个, 各类 gate 验证器: cq4/deepep/dflash/dopd/dspark/eplb/g21/g22/g23/jetspec/kv_cache/layer_adaptive/lplb/nfsordma/ray_engine/rswa_double/sglang_tp4ep4/trueorthokda 等)
- agent/distributed_topology.py (10614B)
- agent/trainers/nemo_automodel_adapter.py (11849B)
- ... 还有 91 个其他

## C. Host2 独有(126 个, Host1 缺)
### 重要: cgc_cpp 新版重构(Host2 独有, Host1 是旧版结构)
- cgc/cgc_cpp/cgc_engines.cpp/.h (Host2 新版, Host1 无)
- cgc/cgc_cpp/cgc_backend.cpp/.h (Host2 新版)
- cgc/cgc_cpp/cgc_cpu_backend.cpp (Host2 新版)
- cgc/cgc_cpp/cgc_platform.cpp/.h (Host2 新版)
- cgc/cgc_cpp/include/kda_engine.h (7448B, Host2 新版)
- cgc/cgc_cpp/include/kda_simd.h (6504B, Host2 新版)
- cgc/cgc_cpp/kernels/* (大量: activation/attention/gemm_cpu/kv_cache/linear/norm/quant/rope/sampling, Host2 拆分新版)
- cgc/cgc_cpp/src/kda_binding.cpp (Host2 新版)
### Host2 独有 bridge
- bridge/geomirror_action_atom_schema.py (35834B)
- bridge/psi0_bridge_generator.py (51752B)
- bridge/psi0_bridge_schema.py (13064B)
- bridge/holomotion_supervision_bundle_schema.json (1723B)
- ... 还有 103 个其他

## 冗余档案(可安全清理)
### Host1
- __pycache__: 64 个目录 (12K)
- build/CMakeFiles: 4.8M (编译中间产物)
- *.o: 27 个 (目标文件)
- *.bak*: 12 个 (备份文件)
### Host2
- __pycache__: 28 个目录 (20K)
- build/CMakeFiles: 4.1M
- *.o: 25 个
- *.bak*: 17 个

## 同步风险
- Host1 cgc_cpp 是旧版结构(src/cgc_cpp.cpp + binding.cpp), Host2 是新版重构(cgc_engines/backend/platform + kernels 拆分)
- Host1 V4-Flash edge 实际在用 cgc_engine(旧版 .so), 盲目覆盖会 break V4-Flash
- Host2 跑 Qwen3-VL(不用 cgc_engine), 新版重构未实测
- cgc_cpp.so 编译产物不同(H1 507KB vs H2 405KB), 需重新编译统一