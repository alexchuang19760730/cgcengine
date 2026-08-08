from __future__ import annotations

from typing import Any

from app.shared.task_type_contract import TASK_TYPE_REPO_DEBUG
from app.shared.task_type_contract import task_type_contract_ref

SWE_AGENT_REQUEST_MARKERS = (
    "<uploaded_files>",
    "<pr_description>",
    "Follow these steps to resolve the issue:",
)

SWE_AGENT_SYSTEM_PROFILE_ID = "cgc.swe_agent_thought_action_v1"
SWE_AGENT_OUTPUT_CONTRACT_ID = "discussion_fenced_bash_v1"
SWE_AGENT_STATE_ABI_ID = "united_pipeline_kernel_v1"
SWE_AGENT_BOOTSTRAP_CONTRACT_BINDING_KEY = "cgc.bootstrap.swe_agent.repo_debug.v1"
SWE_AGENT_EXECUTION_PROFILE_BINDING_KEY = "cgc.execution.swe_agent.repo_debug.v1"
SWE_AGENT_FLOW_PARAMETER_CONTRACT_BINDING_KEY = "cgc.flow.swe_agent.thought_action.v1"
SWE_AGENT_PROFILE_BINDING = {
    "profile_id": SWE_AGENT_SYSTEM_PROFILE_ID,
    "task_type": TASK_TYPE_REPO_DEBUG,
    "initiator": "cgc_4d_perception_matrix",
    "state_abi": SWE_AGENT_STATE_ABI_ID,
    "output_contract": SWE_AGENT_OUTPUT_CONTRACT_ID,
    "execution_profile_binding_key": SWE_AGENT_EXECUTION_PROFILE_BINDING_KEY,
    "bootstrap_contract_binding_key": SWE_AGENT_BOOTSTRAP_CONTRACT_BINDING_KEY,
    "flow_parameter_contract_binding_key": SWE_AGENT_FLOW_PARAMETER_CONTRACT_BINDING_KEY,
    "task_type_contract_ref": task_type_contract_ref(),
}
SWE_AGENT_SYSTEM_PROFILE_VERSION = "v1"
SWE_AGENT_SYSTEM_PROFILE_SOURCE = "cgc_4d_perception_matrix"
SWE_AGENT_SYSTEM_PROFILE_TAG = (
    "[CGC_SYSTEM_PROFILE_BINDING "
    "profile_id=cgc.swe_agent_thought_action_v1 "
    "task_type=repo_debug "
    "initiator=cgc_4d_perception_matrix "
    "state_abi=united_pipeline_kernel_v1 "
    "output_contract=discussion_fenced_bash_v1 "
    "execution_profile_binding_key=cgc.execution.swe_agent.repo_debug.v1 "
    "bootstrap_contract_binding_key=cgc.bootstrap.swe_agent.repo_debug.v1 "
    "flow_parameter_contract_binding_key=cgc.flow.swe_agent.thought_action.v1]"
)
SWE_AGENT_SYSTEM_PROFILE_INSTRUCTION = (
    f"{SWE_AGENT_SYSTEM_PROFILE_TAG}\n"
    "Reply in exactly this format:\n"
    "DISCUSSION\n"
    "short plan\n\n"
    "```bash\n"
    "one command\n"
    "```\n"
    "Rules:\n"
    "- Exactly one bash code block in every reply.\n"
    "- No JSON or XML tool-call markup.\n"
    "- Prefer one short read-only command while exploring; keep output compact.\n"
    "- If finished, reply with:\n"
    "```bash\n"
    "submit\n"
    "```"
)


def is_swe_agent_request(raw_messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(msg, dict)
        and any(marker in str(msg.get("content", "")) for marker in SWE_AGENT_REQUEST_MARKERS)
        for msg in raw_messages
    )


def apply_swe_agent_system_profile(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return messages
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role") or "") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if SWE_AGENT_SYSTEM_PROFILE_TAG in content:
            return messages
        message["content"] = content + "\n\nCRITICAL INSTRUCTION: " + SWE_AGENT_SYSTEM_PROFILE_INSTRUCTION
        return messages
    messages.insert(
        0,
        {
            "role": "system",
            "content": "CRITICAL INSTRUCTION: " + SWE_AGENT_SYSTEM_PROFILE_INSTRUCTION,
        },
    )
    return messages


def swe_agent_profile_binding_ref(
    *,
    profile_settings_path: str = "",
    bootstrap_contract_path: str = "",
    system_manifest_path: str = "",
) -> dict[str, Any]:
    binding_ref = dict(SWE_AGENT_PROFILE_BINDING)
    if str(profile_settings_path or "").strip():
        binding_ref["profile_settings_path"] = str(profile_settings_path)
    if str(bootstrap_contract_path or "").strip():
        binding_ref["bootstrap_contract_path"] = str(bootstrap_contract_path)
    if str(system_manifest_path or "").strip():
        binding_ref["system_manifest_path"] = str(system_manifest_path)
    return binding_ref


def swe_agent_system_profile_ref(*, system_manifest_path: str = "") -> dict[str, Any]:
    return {
        "profile_id": SWE_AGENT_SYSTEM_PROFILE_ID,
        "profile_version": SWE_AGENT_SYSTEM_PROFILE_VERSION,
        "source": SWE_AGENT_SYSTEM_PROFILE_SOURCE,
        "source_path": str(system_manifest_path or ""),
    }


def swe_agent_system_profile_summary(
    *,
    profile_settings_path: str = "",
    bootstrap_contract_path: str = "",
    system_manifest_path: str = "",
) -> dict[str, Any]:
    return {
        "profile_id": SWE_AGENT_SYSTEM_PROFILE_ID,
        "profile_version": SWE_AGENT_SYSTEM_PROFILE_VERSION,
        "task_type": TASK_TYPE_REPO_DEBUG,
        "initiator": "cgc_4d_perception_matrix",
        "state_abi": SWE_AGENT_STATE_ABI_ID,
        "output_contract": SWE_AGENT_OUTPUT_CONTRACT_ID,
        "task_type_contract_ref": task_type_contract_ref(),
        "profile_binding_ref": swe_agent_profile_binding_ref(
            profile_settings_path=profile_settings_path,
            bootstrap_contract_path=bootstrap_contract_path,
            system_manifest_path=system_manifest_path,
        ),
    }


def apply_swe_agent_request_contract(
    payload: dict[str, Any],
    *,
    profile_settings_path: str = "",
    bootstrap_contract_path: str = "",
    system_manifest_path: str = "",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    binding_ref = swe_agent_profile_binding_ref(
        profile_settings_path=profile_settings_path,
        bootstrap_contract_path=bootstrap_contract_path,
        system_manifest_path=system_manifest_path,
    )
    system_profile_ref = swe_agent_system_profile_ref(system_manifest_path=system_manifest_path)
    system_profile_summary = swe_agent_system_profile_summary(
        profile_settings_path=profile_settings_path,
        bootstrap_contract_path=bootstrap_contract_path,
        system_manifest_path=system_manifest_path,
    )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload["metadata"] = {
        **dict(metadata),
        "task_type": TASK_TYPE_REPO_DEBUG,
        "initiator": "cgc_4d_perception_matrix",
        "state_abi": SWE_AGENT_STATE_ABI_ID,
        "output_contract": SWE_AGENT_OUTPUT_CONTRACT_ID,
        "profile_binding_ref": binding_ref,
        "system_profile_ref": system_profile_ref,
        "system_profile_summary": system_profile_summary,
        "task_type_contract_ref": task_type_contract_ref(),
    }
    extra_body = payload.get("extra_body") if isinstance(payload.get("extra_body"), dict) else {}
    payload["extra_body"] = {
        **dict(extra_body),
        "task_type": TASK_TYPE_REPO_DEBUG,
        "profile_binding_ref": binding_ref,
        "system_profile_ref": system_profile_ref,
    }
    payload["task_type"] = TASK_TYPE_REPO_DEBUG
    return payload
