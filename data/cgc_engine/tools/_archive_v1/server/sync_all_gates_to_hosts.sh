#!/bin/bash
# CGC Gate 系列文档完整同步脚本
# 同步所有 Gate 技术白皮书与 CLI 主入口到 host1 和 host2

set -e

HOST1_IP="39.106.118.206"
HOST2_IP="47.95.250.55"
USER="root"
HOST1_PASS="Gen@song@2026622"
HOST2_PASS="Gen@song123"
LOCAL_DIR="/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/"
REMOTE_DIR="/root/flashkv0516/ComputeGraphCompiler-main/"

echo "============================================================"
echo "  🔥 CGC Gate 系列文档 / CLI 完整同步"
echo "============================================================"

# 同步到 Host1
echo ""
echo "[Step 1/2] 同步到 Host1 (${HOST1_IP})..."
sshpass -p "${HOST1_PASS}" rsync -avz \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    --include='cgc_engine/cli.py' \
    --include='cgc_engine/tools/scripts/run/gate_test_framework.py' \
    --include='cgc_engine/tools/scripts/run/self_harness_validation_framework.py' \
    --include='cgc_engine/tools/scripts/run/gate6_model_verify_to_m76_manifest.py' \
    --include='cgc_engine/gate_verifiers/**' \
    --include='Backend/CGC/edge_moe_transport/**' \
    --include='Backend/CGC/cloud_sglang/python/sglang/srt/models/deepseek_v4.py' \
    --include='Backend/CGC/cloud_sglang/python/sglang/srt/model_loader/loader.py' \
    --include='docs/technical_whitepapers/CGC_Gate_1.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.1*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.2*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.3*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_3.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_3.1*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_5.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_6.0*/**' \
    --include='docs/technical_whitepapers/archive/**' \
    --include='docs/technical_whitepapers/*.md' \
    --include='docs/technical_whitepapers/*.json' \
    --include='docs/technical_whitepapers/examples/**' \
    --exclude='._*' \
    --exclude='.DS_Store' \
    --include='*/' \
    --exclude='*' \
    "$LOCAL_DIR" \
    "${USER}@${HOST1_IP}:${REMOTE_DIR}/"

echo "✅ Host1 同步完成"

# 同步到 Host2
echo ""
echo "[Step 2/2] 同步到 Host2 (${HOST2_IP})..."
sshpass -p "${HOST2_PASS}" rsync -avz \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    --include='cgc_engine/cli.py' \
    --include='cgc_engine/tools/scripts/run/gate_test_framework.py' \
    --include='cgc_engine/tools/scripts/run/self_harness_validation_framework.py' \
    --include='cgc_engine/tools/scripts/run/gate6_model_verify_to_m76_manifest.py' \
    --include='cgc_engine/gate_verifiers/**' \
    --include='Backend/CGC/edge_moe_transport/**' \
    --include='Backend/CGC/cloud_sglang/python/sglang/srt/models/deepseek_v4.py' \
    --include='Backend/CGC/cloud_sglang/python/sglang/srt/model_loader/loader.py' \
    --include='docs/technical_whitepapers/CGC_Gate_1.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.1*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.2*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_2.3*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_3.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_3.1*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_5.0*/**' \
    --include='docs/technical_whitepapers/CGC_Gate_6.0*/**' \
    --include='docs/technical_whitepapers/archive/**' \
    --include='docs/technical_whitepapers/*.md' \
    --include='docs/technical_whitepapers/*.json' \
    --include='docs/technical_whitepapers/examples/**' \
    --exclude='._*' \
    --exclude='.DS_Store' \
    --include='*/' \
    --exclude='*' \
    "$LOCAL_DIR" \
    "${USER}@${HOST2_IP}:${REMOTE_DIR}/"

echo "✅ Host2 同步完成"

# 验证 Host1
echo ""
echo "[Step 3/3] 验证 Host1 文件..."
sshpass -p "${HOST1_PASS}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${HOST1_IP}" \
    "echo '=== Host1 Gate 文档列表 ===' && \
    ls -la /root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/ | grep CGC_Gate_ && \
    echo '' && echo '=== Host1 CLI 文件 ===' && \
    ls -l /root/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cli.py && \
    echo '' && echo '=== Host1 gate_map.json 文件 ===' && \
    find /root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers -name '*gate_map.json' -type f | head -20"

echo ""
echo "[Step 4/4] 验证 Host2 文件..."
sshpass -p "${HOST2_PASS}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${HOST2_IP}" \
    "echo '=== Host2 Gate 文档列表 ===' && \
    ls -la /root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/ | grep CGC_Gate_ && \
    echo '' && echo '=== Host2 CLI 文件 ===' && \
    ls -l /root/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cli.py && \
    echo '' && echo '=== Host2 gate_map.json 文件 ===' && \
    find /root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers -name '*gate_map.json' -type f | head -20"

echo ""
echo "============================================================"
echo "  ✅ 全部完成！"
echo ""
echo "  同步的 Gate 版本："
echo "  - CGC_Gate_1.0_edge_cloud_autonomy/"
echo "  - CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/"
echo "  - CGC_Gate_2.1_speculative_decode_fusion_optimization/"
echo "  - CGC_Gate_2.2_deepep_moe_load_balancing/"
echo "  - CGC_Gate_2.3_unlimited_rswa_prefill_pool/"
echo "  - CGC_Gate_3.0_train_inference_unification/"
echo "  - CGC_Gate_3.1_self_harness/"
echo "  - CGC_Gate_5.0_audit_trace_replay_visualization/"
echo "  - CGC_Gate_6.0_fusionroute_complete/"
echo "  - 包含 cgc_engine/cli.py"
echo "  - 包含 self_harness_validation_framework.py"
echo "  - 包含所有 gate_map.json、checkin.json、summary.json、README.md"
echo "============================================================"
