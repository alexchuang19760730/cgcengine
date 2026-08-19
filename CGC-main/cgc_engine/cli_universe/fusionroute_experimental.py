from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSPEC_ROOT = REPO_ROOT / "Backend" / "CGC" / "vendored" / "deepspec"
DEEPSPEC_TRAIN_SCRIPT = DEEPSPEC_ROOT / "train.py"
DEFAULT_DFLASH_CONFIG = (
    DEEPSPEC_ROOT / "config" / "dflash" / "dflash_deepseek_v4_flash.py"
)
DEFAULT_TARGET_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _flatten_opts(opts: Iterable[str]) -> List[str]:
    flattened: List[str] = []
    for opt in opts:
        flattened.extend(["--opts", str(opt)])
    return flattened


def _print_json(payload: Dict[str, Any]):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def build_deepseek_fusionroute_train_command(
    config_path: Path,
    target_model: str = DEFAULT_TARGET_MODEL,
    target_cache_path: str = "",
    exp_name: str = "",
    precision: str = "",
    global_batch_size: int = 0,
    local_batch_size: int = 0,
    num_train_epochs: int = 0,
    learning_rate: float = 0.0,
    extra_opts: Optional[Iterable[str]] = None,
) -> List[str]:
    opts: List[str] = []
    if target_model:
        opts.append(f"model.target_model_name_or_path={target_model}")
    if target_cache_path:
        opts.append(f"data.target_cache_path={target_cache_path}")
    if exp_name:
        opts.append(f"exp_name={exp_name}")
    if precision:
        opts.append(f"train.precision={precision}")
    if global_batch_size > 0:
        opts.append(f"train.global_batch_size={global_batch_size}")
    if local_batch_size > 0:
        opts.append(f"train.local_batch_size={local_batch_size}")
    if num_train_epochs > 0:
        opts.append(f"train.num_train_epochs={num_train_epochs}")
    if learning_rate > 0:
        opts.append(f"train.lr={learning_rate}")
    if extra_opts:
        opts.extend([str(opt) for opt in extra_opts if str(opt).strip()])

    return [
        sys.executable,
        str(DEEPSPEC_TRAIN_SCRIPT),
        "--config",
        str(config_path),
        *_flatten_opts(opts),
    ]


def run_deepseek_fusionroute_train(args) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[FusionRoute Experimental] Config not found: {config_path}")
        return 1
    if not DEEPSPEC_TRAIN_SCRIPT.exists():
        print(
            "[FusionRoute Experimental] deepspec train.py not found: "
            f"{DEEPSPEC_TRAIN_SCRIPT}"
        )
        return 1

    command = build_deepseek_fusionroute_train_command(
        config_path=config_path,
        target_model=args.target_model,
        target_cache_path=args.target_cache_path,
        exp_name=args.exp_name,
        precision=args.precision,
        global_batch_size=args.global_batch_size,
        local_batch_size=args.local_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        extra_opts=args.opts or [],
    )
    report = {
        "mode": "fusionroute_train",
        "train_script": _repo_relative(DEEPSPEC_TRAIN_SCRIPT),
        "config_path": _repo_relative(config_path),
        "target_model": args.target_model,
        "target_cache_path": args.target_cache_path or None,
        "exp_name": args.exp_name or None,
        "precision": args.precision or None,
        "global_batch_size": args.global_batch_size or None,
        "local_batch_size": args.local_batch_size or None,
        "num_train_epochs": args.num_train_epochs or None,
        "learning_rate": args.learning_rate or None,
        "opts": args.opts or [],
        "command": command,
    }

    print("=" * 70)
    print("[FusionRoute Experimental] DeepSeek-V4-Flash training recipe")
    print("=" * 70)
    _print_json(report)
    print("=" * 70)

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("USE_TORCH", "true")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = subprocess.run(
        command,
        cwd=str(DEEPSPEC_ROOT),
        env=env,
    )
    return int(result.returncode)


def _normalize_task_type(task_type: str):
    from cgc_engine.cli_universe.fusionroute_agent import TaskType

    alias_map = {
        "orchestration": TaskType.ORCHESTRATION,
        "orchestrate": TaskType.ORCHESTRATION,
        "planning": TaskType.PLANNING,
        "plan": TaskType.PLANNING,
        "execution": TaskType.EXECUTION,
        "execute": TaskType.EXECUTION,
        "data_synthesis": TaskType.DATA_SYNTHESIS,
        "synthesis": TaskType.DATA_SYNTHESIS,
        "rl_training": TaskType.RL_TRAINING,
        "rl": TaskType.RL_TRAINING,
        "audit_trace": TaskType.AUDIT_TRACE,
        "audit": TaskType.AUDIT_TRACE,
        "health_check": TaskType.HEALTH_CHECK,
        "health": TaskType.HEALTH_CHECK,
        "tenant_management": TaskType.TENANT_MANAGEMENT,
        "tenant": TaskType.TENANT_MANAGEMENT,
    }
    key = (task_type or "").strip().lower()
    if key not in alias_map:
        raise ValueError(f"Unsupported task type: {task_type}")
    return alias_map[key]


def _build_payload(
    instruction: str,
    domain: str,
    step: int,
    payload_json: str = "",
    prompt_field: str = "instruction",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if payload_json:
        payload = json.loads(payload_json)
    if instruction:
        payload.setdefault(prompt_field, instruction)
        payload.setdefault("task", instruction)
    if domain:
        payload.setdefault("domain", domain)
    if step > 0:
        payload.setdefault("step", step)
    return payload


def _iter_requests(args) -> List[Dict[str, Any]]:
    if not args.input_jsonl:
        return [
            {
                "task_type": args.task_type,
                "payload": _build_payload(
                    instruction=args.instruction,
                    domain=args.domain,
                    step=args.step,
                    payload_json=args.payload_json,
                    prompt_field=args.prompt_field,
                ),
            }
        ]

    input_path = Path(args.input_jsonl).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            task_type = item.get("task_type", args.task_type)
            payload = dict(item.get("payload", {}))
            prompt_value = (
                item.get(args.prompt_field)
                or item.get("instruction")
                or item.get("task")
                or item.get("prompt")
                or ""
            )
            if prompt_value:
                payload.setdefault(args.prompt_field, prompt_value)
                payload.setdefault("task", prompt_value)
            if args.domain:
                payload.setdefault("domain", args.domain)
            if args.step > 0:
                payload.setdefault("step", args.step)
            rows.append(
                {
                    "line": idx,
                    "task_type": task_type,
                    "payload": payload,
                }
            )
    return rows


def run_deepseek_fusionroute_infer(args) -> int:
    from cgc_engine.cli_universe.fusionroute_agent import create_fusionroute_agent

    requests = _iter_requests(args)
    agent = create_fusionroute_agent(host=args.host)
    records: List[Dict[str, Any]] = []

    for idx, request in enumerate(requests, start=1):
        task_type = _normalize_task_type(request["task_type"])
        task = agent.submit_and_execute(
            task_type=task_type,
            payload=request["payload"],
            tenant_id=args.tenant_id,
        )
        records.append(
            {
                "index": idx,
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "status": task.status,
                "error": task.error,
                "route_history": task.route_history,
                "result": task.result,
            }
        )

    summary = {
        "mode": "fusionroute_run",
        "request_count": len(records),
        "host": args.host,
        "tenant_id": args.tenant_id,
        "default_task_type": args.task_type,
        "records": records,
    }

    print("=" * 70)
    print("[FusionRoute Experimental] DeepSeek-backed run summary")
    print("=" * 70)
    _print_json(summary)
    print("=" * 70)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[FusionRoute Experimental] Wrote summary to: {output_path}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[FusionRoute Experimental] Wrote JSONL to: {output_path}")

    failed = sum(1 for row in records if row["status"] not in {"completed", "pending"})
    return 1 if failed else 0
