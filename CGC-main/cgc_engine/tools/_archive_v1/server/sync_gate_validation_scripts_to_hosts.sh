#!/bin/bash
# 同步 Gate 1.0 / 2.0 / 3.0 文档验证脚本与 CLI 主入口到 Host1 / Host2，并在远端执行 legacy mapping / CLI 入口检查

set -e

HOST1_IP="39.106.118.206"
HOST2_IP="47.95.250.55"
USER="root"
HOST1_PASS="Gen@song@2026622"
HOST2_PASS="Gen@song123"
LOCAL_DIR="/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/"
REMOTE_DIR="/root/flashkv0516/ComputeGraphCompiler-main/"
REMOTE_APP_DIR="/root/flashkv0516/app/"

SCRIPTS=(
    "cgc_engine/cli.py"
    "cgc_engine/pipeline.py"
    "cgc_engine/product/m76_gate.py"
    "cgc_engine/gate_verifiers/__init__.py"
    "cgc_engine/gate_verifiers/trueorthokda_verifier.py"
    "cgc_engine/gate_verifiers/deepseek_v4_flash_resume_verifier.py"
    "cgc_engine/gate_verifiers/edge_omlx_flashmoe_verifier.py"
    "cgc_engine/gate_verifiers/ray_engine_dual_host_verifier.py"
    "cgc_engine/gate_verifiers/colossalai_runtime_candidate_verifier.py"
    "cgc_engine/gate_verifiers/g21_fusion_governance_verifier.py"
    "cgc_engine/gate_verifiers/g22_deepep_l20n_verifier.py"
    "cgc_engine/gate_verifiers/kv_cache_verifier.py"
    "cgc_engine/gate_verifiers/rswa_double_layer_kv_verifier.py"
    "cgc_engine/gate_verifiers/g23_trueorthokda_adapter_verifier.py"
    "cgc_engine/gate_verifiers/g23_cloud_l20n_tp4_verifier.py"
    "cgc_engine/gate_verifiers/unified_ir_inject_verifier.py"
    "cgc_engine/gate_verifiers/nfsordma_verifier.py"
    "cgc_engine/gate_verifiers/endtoend_moe_transport_verifier.py"
    "cgc_engine/utils/envs.py"
    "cgc_engine/product/release_alias_contracts.py"
    "cgc_engine/product/upkg21_gate.py"
    "cgc_engine/product/upkg30_common.py"
    "docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/README.md"
    "docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_gate_map.json"
    "docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_summary.example.json"
    "docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_checkin.example.json"
    "docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/README.md"
    "docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_gate_map.json"
    "docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_summary.example.json"
    "docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_checkin.example.json"
    "docs/technical_whitepapers/CGC_Gate_3.0_train_inference_unification/README.md"
    "docs/technical_whitepapers/CGC_Gate_3.0_train_inference_unification/CGC_Gate_3.0_train_inference_unification_gate_map.json"
    "docs/technical_whitepapers/CGC_Gate_3.0_train_inference_unification/CGC_Gate_3.0_train_inference_unification_summary.example.json"
    "docs/technical_whitepapers/CGC_Gate_3.0_train_inference_unification/CGC_Gate_3.0_train_inference_unification_checkin.example.json"
    "docs/technical_whitepapers/CGC_Gate_3.1_self_harness/README.md"
    "docs/technical_whitepapers/CGC_Gate_3.1_self_harness/CGC_Gate_3.1_self_harness_gate_map.json"
    "docs/technical_whitepapers/CGC_Gate_3.1_self_harness/CGC_Gate_3.1_self_harness_summary.example.json"
    "docs/technical_whitepapers/CGC_Gate_3.1_self_harness/CGC_Gate_3.1_self_harness_checkin.example.json"
    "docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization/README.md"
    "docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization/CGC_Gate_5.0_audit_trace_replay_visualization_Technical_Whitepaper_v1.0_zh_CN.md"
    "docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization/CGC_Gate_5.0_audit_trace_replay_visualization_gate_map.json"
    "docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization/CGC_Gate_5.0_audit_trace_replay_visualization_summary.example.json"
    "docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization/CGC_Gate_5.0_audit_trace_replay_visualization_checkin.example.json"
    "docs/technical_whitepapers/examples/dualnode_blackwell_deepep_ep16_tp1_runtime_bootstrap_contract.example.json"
    "docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_system_manifest.example.json"
    "docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_profile_settings.example.json"
    "docs/technical_whitepapers/examples/host2_blackwell_sglang_runtime_bootstrap_contract.example.json"
    "docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_system_manifest.example.json"
    "docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_profile_settings.example.json"
    "docs/gate_whitepapers/CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME_REPORT_SCHEMA_v1.0.json"
    "cgc_engine/tools/scripts/run/validate_gate10_legacy_mapping.py"
    "cgc_engine/tools/scripts/run/validate_gate20_legacy_mapping.py"
    "cgc_engine/tools/scripts/run/gate_test_framework.py"
    "cgc_engine/tools/scripts/run/self_harness_validation_framework.py"
    "cgc_engine/tools/scripts/run/run_all_gate_tests_final.py"
    "cgc_engine/tools/scripts/run/run_all_gate_tests_final.sh"
    "Backend/__init__.py"
    "Backend/CGC/__init__.py"
    "Backend/CGC/ray_serve_sglang_gateway.py"
    "Backend/CGC/deepep_sglang_patch.py"
    "Backend/CGC/compiler/unified_compiler.py"
    "Backend/CGC/vendored/jetspec/jetspec/inference_engine/engine.py"
    "Backend/CGC/vendored/jetspec/jetspec/tree/layer_conditional/path_conditional_refresh.py"
    "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/kv_cache_builder.py"
    "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/cache_init_params.py"
    "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/memory_pool_host.py"
    "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/radix_cache_cpp.py"
    "Backend/CGC/cloud_sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
    "Backend/CGC/edge_moe_transport/__init__.py"
    "Backend/CGC/edge_moe_transport/cq4_session.py"
    "Backend/CGC/edge_moe_transport/nfsordma_transport.py"
    "Backend/CGC/edge_moe_transport/rdma_cm_exchange.py"
    "Backend/CGC/edge_moe_transport/transport_contract.py"
    "cgc_engine/product/m75_api_compat_gate.py"
    "cgc_engine/product/m1_m6_pipeline_gates.py"
    "cgc_engine/rswa_integration/__init__.py"
    "cgc_engine/rswa_integration/rswa_prefill_pool_adapter.py"
    "cgc_engine/prefill_pool/__init__.py"
    "cgc_engine/prefill_pool/prefill_pool.py"
    "cgc_engine/gds_service/__init__.py"
    "cgc_engine/gds_service/cufile_wrapper.py"
    "cgc_engine/pd/kv_async_prefetch.py"
    "../app/__init__.py"
    "../app/edge_engine/__init__.py"
    "../app/edge_engine/kda_state_runtime.py"
    "../app/edge_engine/local_infer.py"
    "../app/shared/task_type_contract.py"
    "../app/shared/profile_bundle_validator.py"
    "../app/shared/contracts/task_type_contract.json"
    "../app/servers/__init__.py"
    "../app/servers/cgc_api_server.py"
    "../app/servers/cloud_socket_server.py"
    "../app/cli/cgc.py"
)

sync_host() {
    local host_name="$1"
    local host_ip="$2"
    local host_pass="$3"

    echo ""
    echo "[Sync] ${host_name} (${host_ip})"

    sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
        "mkdir -p ${REMOTE_DIR}/cgc_engine/tools/scripts/run ${REMOTE_DIR}/cgc_engine/rswa_integration ${REMOTE_DIR}/cgc_engine/prefill_pool ${REMOTE_DIR}/cgc_engine/gds_service ${REMOTE_DIR}/cgc_engine/pd ${REMOTE_DIR}/docs/gate_whitepapers ${REMOTE_DIR}/docs/technical_whitepapers/examples ${REMOTE_DIR}/docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy ${REMOTE_DIR}/docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation ${REMOTE_DIR}/docs/technical_whitepapers/CGC_Gate_3.0_train_inference_unification ${REMOTE_DIR}/docs/technical_whitepapers/CGC_Gate_3.1_self_harness ${REMOTE_DIR}/docs/technical_whitepapers/CGC_Gate_5.0_audit_trace_replay_visualization ${REMOTE_DIR}/Backend/CGC/compiler ${REMOTE_DIR}/Backend/CGC/edge_moe_transport ${REMOTE_DIR}/Backend/CGC/vendored/jetspec/jetspec/inference_engine ${REMOTE_DIR}/Backend/CGC/vendored/jetspec/jetspec/tree/layer_conditional ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/distributed/device_communicators ${REMOTE_APP_DIR}/edge_engine ${REMOTE_APP_DIR}/shared/contracts ${REMOTE_APP_DIR}/servers ${REMOTE_APP_DIR}/cli"

    for rel_path in "${SCRIPTS[@]}"; do
        echo "  - ${rel_path}"
        remote_path="${REMOTE_DIR}${rel_path}"
        if [[ "${rel_path}" == ../app/* ]]; then
            remote_path="${REMOTE_APP_DIR}${rel_path#../app/}"
        fi
        remote_parent="$(dirname "${remote_path}")"
        sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
            "mkdir -p ${remote_parent}"
        sshpass -p "${host_pass}" rsync -avz \
            -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
            "${LOCAL_DIR}${rel_path}" \
            "${USER}@${host_ip}:${remote_path}"
    done

    echo "[Verify] ${host_name} file presence"
    sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
        "ls -l \
        ${REMOTE_DIR}/cgc_engine/cli.py \
        ${REMOTE_DIR}/cgc_engine/pipeline.py \
        ${REMOTE_DIR}/cgc_engine/product/m76_gate.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/__init__.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/trueorthokda_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/deepseek_v4_flash_resume_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/edge_omlx_flashmoe_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/ray_engine_dual_host_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/colossalai_runtime_candidate_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/g21_fusion_governance_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/g22_deepep_l20n_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/kv_cache_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/rswa_double_layer_kv_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/g23_trueorthokda_adapter_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/g23_cloud_l20n_tp4_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/unified_ir_inject_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/nfsordma_verifier.py \
        ${REMOTE_DIR}/cgc_engine/gate_verifiers/endtoend_moe_transport_verifier.py \
        ${REMOTE_DIR}/cgc_engine/utils/envs.py \
        ${REMOTE_DIR}/cgc_engine/product/release_alias_contracts.py \
        ${REMOTE_DIR}/cgc_engine/product/upkg21_gate.py \
        ${REMOTE_DIR}/cgc_engine/product/upkg30_common.py \
        ${REMOTE_DIR}/cgc_engine/product/m75_api_compat_gate.py \
        ${REMOTE_DIR}/cgc_engine/product/m1_m6_pipeline_gates.py \
        ${REMOTE_DIR}/Backend/CGC/ray_serve_sglang_gateway.py \
        ${REMOTE_DIR}/Backend/CGC/deepep_sglang_patch.py \
        ${REMOTE_DIR}/Backend/CGC/compiler/unified_compiler.py \
        ${REMOTE_DIR}/Backend/CGC/vendored/jetspec/jetspec/inference_engine/engine.py \
        ${REMOTE_DIR}/Backend/CGC/vendored/jetspec/jetspec/tree/layer_conditional/path_conditional_refresh.py \
        ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/kv_cache_builder.py \
        ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/cache_init_params.py \
        ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/memory_pool_host.py \
        ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/radix_cache_cpp.py \
        ${REMOTE_DIR}/Backend/CGC/cloud_sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py \
        ${REMOTE_DIR}/Backend/CGC/edge_moe_transport/cq4_session.py \
        ${REMOTE_DIR}/Backend/CGC/edge_moe_transport/nfsordma_transport.py \
        ${REMOTE_DIR}/Backend/CGC/edge_moe_transport/rdma_cm_exchange.py \
        ${REMOTE_DIR}/Backend/CGC/edge_moe_transport/transport_contract.py \
        ${REMOTE_DIR}/cgc_engine/rswa_integration/rswa_prefill_pool_adapter.py \
        ${REMOTE_DIR}/cgc_engine/prefill_pool/prefill_pool.py \
        ${REMOTE_DIR}/cgc_engine/gds_service/cufile_wrapper.py \
        ${REMOTE_DIR}/cgc_engine/pd/kv_async_prefetch.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/validate_gate10_legacy_mapping.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/validate_gate20_legacy_mapping.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/gate_test_framework.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/self_harness_validation_framework.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/run_all_gate_tests_final.py \
        ${REMOTE_DIR}/cgc_engine/tools/scripts/run/run_all_gate_tests_final.sh \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/dualnode_blackwell_deepep_ep16_tp1_runtime_bootstrap_contract.example.json \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_system_manifest.example.json \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_profile_settings.example.json \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/host2_blackwell_sglang_runtime_bootstrap_contract.example.json \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_system_manifest.example.json \
        ${REMOTE_DIR}/docs/technical_whitepapers/examples/host2_upkg21_dflash_benchmark_profile_settings.example.json \
        ${REMOTE_APP_DIR}/cli/cgc.py \
        ${REMOTE_APP_DIR}/edge_engine/kda_state_runtime.py \
        ${REMOTE_APP_DIR}/shared/task_type_contract.py \
        ${REMOTE_APP_DIR}/shared/profile_bundle_validator.py \
        ${REMOTE_APP_DIR}/servers/cgc_api_server.py \
        ${REMOTE_APP_DIR}/servers/cloud_socket_server.py"

    echo "[Verify] ${host_name} CLI sync markers"
    sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
        "if command -v rg >/dev/null 2>&1; then \
            rg -n 'g23_unified_ir_inject_sglang_compute_graph|CGC_Gate_1.0' ${REMOTE_DIR}/cgc_engine/cli.py; \
        else \
            grep -nE 'g23_unified_ir_inject_sglang_compute_graph|CGC_Gate_1.0' ${REMOTE_DIR}/cgc_engine/cli.py; \
        fi"

    echo "[Verify] ${host_name} Gate 1.0 legacy mapping"
    sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
        "python3 ${REMOTE_DIR}/cgc_engine/tools/scripts/run/validate_gate10_legacy_mapping.py"

    echo "[Verify] ${host_name} Gate 2.0 legacy mapping"
    sshpass -p "${host_pass}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${USER}@${host_ip}" \
        "python3 ${REMOTE_DIR}/cgc_engine/tools/scripts/run/validate_gate20_legacy_mapping.py"
}

echo "============================================================"
echo "  Gate 文档验证脚本同步"
echo "============================================================"

sync_host "Host1" "${HOST1_IP}" "${HOST1_PASS}"
sync_host "Host2" "${HOST2_IP}" "${HOST2_PASS}"

echo ""
echo "============================================================"
echo "  完成"
echo "============================================================"
