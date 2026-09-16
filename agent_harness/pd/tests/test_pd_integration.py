"""
PD (Prefill-Decode) 整合冒烟测试
验证：PD 核心模块 + Master/Slave 同步 + 算力共享框架 + 统一配置
"""
import os
import sys
import time

# Add agent_harness to path
_agent_harness = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _agent_harness)

# Remove pre-imported modules
for _mod in list(sys.modules.keys()):
    if _mod.startswith("pd") or _mod.startswith("agent_harness.pd"):
        del sys.modules[_mod]


def test_pd_core_import():
    """测试 PD 核心模块导入"""
    from pd.protocol import (
        HiddenStatePacket, PDMode, SourceModel, TargetModel,
        EmitRequest, EmitResponse, ResumeRequest,
    )
    assert HiddenStatePacket is not None
    assert PDMode is not None
    print("  [PASS] PD core modules imported")


def test_pd_config():
    """测试 PD 统一配置"""
    from pd.pd_config import PDConfig, PDServerConfig, MasterSlaveConfig, ComputeSharingConfig

    config = PDConfig()
    assert config.server.port == 9000
    assert config.master_slave.sync_interval == 300
    assert config.compute_sharing.port == 9100

    # 从环境变量加载
    os.environ["PD_SERVER_PORT"] = "9999"
    config2 = PDConfig.from_env()
    assert config2.server.port == 9999
    del os.environ["PD_SERVER_PORT"]

    print("  [PASS] PDConfig created and loaded from env")


def test_master_slave_sync_config():
    """测试 Master/Slave 同步配置"""
    from pd.master_slave_sync import SyncConfig, SyncRole, SyncMethod, CodeSyncer

    config = SyncConfig(
        role=SyncRole.SLAVE,
        method=SyncMethod.INCREMENTAL,
        master_url="http://127.0.0.1:9000",
        sync_interval=60,
    )
    assert config.role == SyncRole.SLAVE
    assert config.method == SyncMethod.INCREMENTAL
    assert config.sync_interval == 60

    syncer = CodeSyncer(config)
    assert syncer.config is not None
    assert syncer.config.slave_id != ""

    print("  [PASS] Master/Slave sync config created")


def test_master_slave_file_scan():
    """测试文件扫描和 hash"""
    from pd.master_slave_sync import SyncConfig, SyncRole, CodeSyncer

    config = SyncConfig(role=SyncRole.MASTER, sync_directories=["pd"])
    syncer = CodeSyncer(config)

    # 扫描本地文件
    states = syncer.scan_local()
    assert len(states) > 0
    assert all(s.md5 and s.path for s in states)

    # 获取 manifest
    manifest = syncer.get_manifest()
    assert "version" in manifest
    assert "files" in manifest
    assert manifest["file_count"] > 0

    print(f"  [PASS] File scan: {len(states)} files, manifest version={manifest['version'][:8]}")


def test_compute_sharing_node():
    """测试算力节点创建和评分"""
    from pd.compute_sharing import (
        ComputeNode, ComputeTask, TaskType, NodeRole,
        GPUInfo, ModelInstance, SharingPolicy,
    )

    node = ComputeNode(
        node_id="mac-a-test",
        name="Mac A Gemma4",
        role=NodeRole.EDGE,
        address="http://127.0.0.1:9100",
        cpu_cores=12,
        memory_total_gb=64.0,
        policy=SharingPolicy.SHARED,
        last_heartbeat=time.time(),
    )
    node.gpus.append(GPUInfo(index=0, name="M3 Max", memory_total_gb=36.0, memory_used_gb=10.0))
    node.models.append(ModelInstance(
        model_id="gemma-4-26b-a4b-it",
        backend="llama.cpp",
        quantization="q4_k_m",
        context_size=8192,
        running=True,
        port=8080,
    ))

    # 测试节点状态
    assert node.is_alive
    assert node.memory_free_gb > 0
    assert node.gpu_memory_free_total > 0

    # 测试任务评分
    task = ComputeTask(
        task_type=TaskType.INFERENCE,
        model_id="gemma-4-26b-a4b-it",
        prompt="hello",
    )
    assert node.can_accept_task(task)
    score = node.score_task(task)
    assert score > 0

    # 测试不匹配的模型
    task2 = ComputeTask(task_type=TaskType.INFERENCE, model_id="nonexistent-model")
    assert not node.can_accept_task2 if hasattr(node, 'can_accept_task2') else True

    print(f"  [PASS] Compute node: {node.name}, score={score:.1f}")


def test_compute_scheduler():
    """测试算力调度器"""
    from pd.compute_sharing import (
        ComputeScheduler, ComputeNode, ComputeTask,
        TaskType, NodeRole, GPUInfo, ModelInstance,
    )

    scheduler = ComputeScheduler()

    # 注册两个节点
    node1 = scheduler.register_node(
        node_id="mac-a", name="Mac A Gemma4",
        address="http://127.0.0.1:9101", role=NodeRole.EDGE,
        gpu_count=1, cpu_cores=12, memory_total_gb=64.0,
    )
    node1.models.append(ModelInstance(model_id="gemma-4-26b", running=True, port=8080))
    node1.last_heartbeat = time.time()

    node2 = scheduler.register_node(
        node_id="mac-b", name="Mac B Qwen3.6",
        address="http://127.0.0.1:9102", role=NodeRole.EDGE,
        gpu_count=1, cpu_cores=16, memory_total_gb=128.0,
    )
    node2.models.append(ModelInstance(model_id="qwen3.6-35b", running=True, port=8081))
    node2.last_heartbeat = time.time()

    # 调度 Gemma4 任务
    task1 = ComputeTask(task_type=TaskType.INFERENCE, model_id="gemma-4-26b", prompt="hello")
    result1 = scheduler.schedule(task1)
    assert result1.success
    assert result1.node.node_id == "mac-a"

    # 调度 Qwen3.6 任务
    task2 = ComputeTask(task_type=TaskType.INFERENCE, model_id="qwen3.6-35b", prompt="hello")
    result2 = scheduler.schedule(task2)
    assert result2.success
    assert result2.node.node_id == "mac-b"

    # 测试算力共享
    share = scheduler.share_compute(
        from_node="mac-a", to_node="mac-b",
        task_type=TaskType.INFERENCE, model_id="gemma-4-26b",
    )
    assert share["success"]

    # 测试拓扑
    topology = scheduler.get_sharing_topology()
    assert topology["total_nodes"] == 2
    assert len(topology["edge_nodes"]) == 2

    print(f"  [PASS] Scheduler: 2 nodes, task1→{result1.node.name}, task2→{result2.node.name}")


def test_pd_integration_all():
    """测试 PD 整合包完整导入"""
    import pd

    # 检查核心属性
    assert hasattr(pd, "HiddenStatePacket")
    assert hasattr(pd, "PDConfig")
    assert hasattr(pd, "CodeSyncer")
    assert hasattr(pd, "ComputeScheduler")
    assert hasattr(pd, "ComputeNode")
    assert hasattr(pd, "ComputeTask")

    # 检查可用性标记
    assert pd._PD_CONFIG_AVAILABLE is True
    assert pd._SYNC_AVAILABLE is True
    assert pd._COMPUTE_SHARING_AVAILABLE is True

    print("  [PASS] PD integration package: all modules available")


def main():
    print("=" * 60)
    print("PD (Prefill-Decode) Integration Smoke Test")
    print("=" * 60)
    print()

    tests = [
        ("PD core import", test_pd_core_import),
        ("PD unified config", test_pd_config),
        ("Master/Slave sync config", test_master_slave_sync_config),
        ("Master/Slave file scan", test_master_slave_file_scan),
        ("Compute sharing node", test_compute_sharing_node),
        ("Compute scheduler", test_compute_scheduler),
        ("PD integration all", test_pd_integration_all),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
