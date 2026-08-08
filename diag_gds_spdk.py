#!/usr/bin/env python3
"""GDS + SPDK 诊断脚本"""
import sys, traceback
sys.path.insert(0, ".")

print("=== GDS 诊断 ===")
try:
    import cgc_engine.gds_service.cufile_wrapper as cw
    print("  模块加载: OK")
    attrs = [x for x in dir(cw) if not x.startswith("_")]
    print("  模块属性:", attrs[:20])
except Exception as e:
    print("  模块加载: FAIL")
    traceback.print_exc()

print()
print("=== GDS __init__.py ===")
try:
    import cgc_engine.gds_service as gs
    attrs = [x for x in dir(gs) if not x.startswith("_")]
    print("  __init__ 属性:", attrs[:15])
except Exception as e:
    print("  __init__ 加载: FAIL", e)

print()
print("=== SPDK 诊断 ===")
try:
    import cgc_engine.spdk_adapter.spdk_io_manager as sim
    attrs = [x for x in dir(sim) if x[0].isupper() or x.startswith("SPDK")]
    print("  模块属性:", attrs[:15])
except Exception as e:
    print("  模块加载: FAIL", e)

try:
    from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager, SPDKConfig
    print("  SPDKIOManager + SPDKConfig: OK")
except Exception as e:
    print("  SPDKIOManager: FAIL", e)

# SPDKTask 搜索
try:
    from cgc_engine.spdk_adapter.spdk_io_manager import SPDKTask
    print("  SPDKTask: OK")
except ImportError:
    print("  SPDKTask: NOT FOUND, 搜索替代...")
    for name in dir(sim):
        if "task" in name.lower() or "Task" in name:
            print(f"    找到: {name}")

print()
print("=== GDS 实际 API 测试 ===")
try:
    from cgc_engine.gds_service.cufile_wrapper import is_gds_available
    print("  is_gds_available: import OK")
    print("  is_gds_available():", is_gds_available())
except Exception as e:
    print("  is_gds_available: FAIL", e)

try:
    from cgc_engine.gds_service.cufile_wrapper import cuFileRead, cuFileWrite
    print("  cuFileRead/cuFileWrite: import OK")
except Exception as e:
    print("  cuFileRead/cuFileWrite: FAIL", e)

try:
    from cgc_engine.gds_service.cufile_wrapper import CuFileHandle
    print("  CuFileHandle: import OK")
except Exception as e:
    print("  CuFileHandle: FAIL", e)

print()
print("=== liburing 安装检查 ===")
try:
    import liburing
    print("  liburing: available")
except ImportError:
    print("  liburing: NOT available")
    import subprocess
    r = subprocess.run(["pip", "install", "liburing"], capture_output=True, text=True, timeout=30)
    print("  pip install liburing:", "OK" if r.returncode == 0 else "FAIL")
    if r.returncode != 0:
        print("  stderr:", r.stderr[:200])
    else:
        try:
            import liburing
            print("  liburing 安装后: available")
        except ImportError:
            print("  liburing 安装后仍不可用")

print()
print("=== FlashMoE + GDS/SPDK 集成 ===")
try:
    from cgc_engine.flash_moe.distributed_expert_store import DistributedExpertStore
    print("  DistributedExpertStore: import OK")
except Exception as e:
    print("  DistributedExpertStore: FAIL", e)

try:
    from cgc_engine.flash_moe.gds_expert_loader import GDSExpertLoader
    print("  GDSExpertLoader: import OK")
except Exception as e:
    print("  GDSExpertLoader: FAIL", e)
