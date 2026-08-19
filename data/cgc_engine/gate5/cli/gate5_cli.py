"""
CGC Gate 5.0 CLI
实现白皮书定义的所有命令：task / audit / trace / config
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cgc_engine.gate5.core.engine import Gate5Engine


def load_config(config_path: str = None) -> dict:
    """加载 Gate 5.0 配置"""
    if config_path is None:
        config_path = PROJECT_ROOT / "cgc_engine" / "gate5" / "config" / "gate5_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_engine(config_path: str = None) -> Gate5Engine:
    """获取 Gate5Engine 实例"""
    config = load_config(config_path)
    gate5_cfg = config.get("gate5", {})
    storage_cfg = gate5_cfg.get("storage", {})
    storage_path = storage_cfg.get("path", "gate5_data")
    # 使用绝对路径
    if not os.path.isabs(storage_path):
        storage_path = str(PROJECT_ROOT / storage_path)
    from cgc_engine.gate5.core.engine import FileStorageBackend
    backend = FileStorageBackend(base_path=storage_path)
    return Gate5Engine(storage_backend=backend)


def cmd_task_create(args):
    """创建任务"""
    engine = get_engine(args.config)
    inputs = json.loads(args.input) if args.input else {}
    user_id = args.user
    task_id = engine.create_task(user_id=user_id, inputs=inputs)
    result = {"task_id": task_id, "status": "running", "created_at": datetime.now().isoformat()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_task_get(args):
    """获取任务详情"""
    engine = get_engine(args.config)
    task = engine.get_task(args.task_id)
    if task is None:
        print(json.dumps({"error": f"Task {args.task_id} not found"}, ensure_ascii=False))
        return 1
    print(json.dumps(task, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_task_list(args):
    """列出任务"""
    engine = get_engine(args.config)
    tasks = engine.list_tasks(user_id=args.user, limit=args.limit)
    print(json.dumps(tasks, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_task_replay(args):
    """回溯任务"""
    engine = get_engine(args.config)
    events = engine.replay_task(args.task_id, speed=args.speed)
    if not events:
        print(json.dumps({"error": f"No snapshots found for task {args.task_id}"}, ensure_ascii=False))
        return 1
    print(json.dumps(events, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_audit_list(args):
    """查询审计日志"""
    engine = get_engine(args.config)
    start_time = None
    end_time = None
    if args.start:
        start_time = datetime.fromisoformat(args.start).timestamp()
    if args.end:
        end_time = datetime.fromisoformat(args.end).timestamp()
    records = engine.query_audit(
        start_time=start_time,
        end_time=end_time,
        action=args.action,
        user_id=args.user,
        task_id=args.task_id,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_audit_report(args):
    """生成审计报告"""
    engine = get_engine(args.config)
    start_time = None
    end_time = None
    if args.start:
        start_time = datetime.fromisoformat(args.start).timestamp()
    if args.end:
        end_time = datetime.fromisoformat(args.end).timestamp()
    report = engine.generate_audit_report(start_time=start_time, end_time=end_time)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_trace_get(args):
    """获取追踪信息"""
    engine = get_engine(args.config)
    trace = engine.get_task_trace(args.task_id)
    print(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_trace_export(args):
    """导出追踪数据"""
    engine = get_engine(args.config)
    data = engine.export_trace(args.task_id, format=args.format)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"Trace exported to {args.output}")
    else:
        print(data)
    return 0


def cmd_config_show(args):
    """显示配置"""
    config = load_config(args.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def cmd_config_set(args):
    """设置配置项"""
    config_path = args.config or str(PROJECT_ROOT / "cgc_engine" / "gate5" / "config" / "gate5_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # 支持 dot notation: gate5.audit.enabled
    keys = args.key.split(".")
    target = config
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    # 尝试解析 JSON 值
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    target[keys[-1]] = value
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Set {args.key} = {value}")
    return 0


def add_task_subparser(subparsers):
    """task 子命令"""
    parser = subparsers.add_parser("task", help="任务管理")
    task_sub = parser.add_subparsers(dest="task_command", description="任务子命令")

    # task create
    p = task_sub.add_parser("create", help="创建任务")
    p.add_argument("--input", type=str, default="{}", help='任务输入 JSON, 例如 \'{"query": "..."}\'')
    p.add_argument("--user", type=str, default=None, help="用户 ID")
    p.set_defaults(func=cmd_task_create)

    # task get
    p = task_sub.add_parser("get", help="获取任务详情")
    p.add_argument("task_id", type=str, help="任务 ID")
    p.set_defaults(func=cmd_task_get)

    # task list
    p = task_sub.add_parser("list", help="列出任务")
    p.add_argument("--user", type=str, default=None, help="按用户过滤")
    p.add_argument("--limit", type=int, default=50, help="返回数量上限")
    p.set_defaults(func=cmd_task_list)

    # task replay
    p = task_sub.add_parser("replay", help="回溯任务")
    p.add_argument("task_id", type=str, help="任务 ID")
    p.add_argument("--speed", type=float, default=1.0, help="回放速度")
    p.set_defaults(func=cmd_task_replay)


def add_audit_subparser(subparsers):
    """audit 子命令"""
    parser = subparsers.add_parser("audit", help="审计管理")
    audit_sub = parser.add_subparsers(dest="audit_command", description="审计子命令")

    # audit list
    p = audit_sub.add_parser("list", help="查询审计日志")
    p.add_argument("--start", type=str, default=None, help="开始时间 (ISO 格式)")
    p.add_argument("--end", type=str, default=None, help="结束时间 (ISO 格式)")
    p.add_argument("--action", type=str, default=None, help="按操作类型过滤")
    p.add_argument("--user", type=str, default=None, help="按用户过滤")
    p.add_argument("--task-id", type=str, default=None, help="按任务 ID 过滤")
    p.set_defaults(func=cmd_audit_list)

    # audit report
    p = audit_sub.add_parser("report", help="生成审计报告")
    p.add_argument("--start", type=str, default=None, help="开始时间 (ISO 格式)")
    p.add_argument("--end", type=str, default=None, help="结束时间 (ISO 格式)")
    p.set_defaults(func=cmd_audit_report)


def add_trace_subparser(subparsers):
    """trace 子命令"""
    parser = subparsers.add_parser("trace", help="追踪管理")
    trace_sub = parser.add_subparsers(dest="trace_command", description="追踪子命令")

    # trace get
    p = trace_sub.add_parser("get", help="获取追踪信息")
    p.add_argument("task_id", type=str, help="任务 ID")
    p.set_defaults(func=cmd_trace_get)

    # trace export
    p = trace_sub.add_parser("export", help="导出追踪数据")
    p.add_argument("task_id", type=str, help="任务 ID")
    p.add_argument("--format", type=str, default="json", choices=["json", "csv"], help="导出格式")
    p.add_argument("--output", type=str, default=None, help="输出文件路径")
    p.set_defaults(func=cmd_trace_export)


def add_config_subparser(subparsers):
    """config 子命令"""
    parser = subparsers.add_parser("config", help="配置管理")
    config_sub = parser.add_subparsers(dest="config_command", description="配置子命令")

    # config show
    p = config_sub.add_parser("show", help="显示配置")
    p.set_defaults(func=cmd_config_show)

    # config set
    p = config_sub.add_parser("set", help="设置配置项")
    p.add_argument("key", type=str, help="配置键 (支持 dot notation, 例如 gate5.audit.enabled)")
    p.add_argument("value", type=str, help="配置值")
    p.set_defaults(func=cmd_config_set)


def create_parser():
    """创建 CLI 解析器"""
    parser = argparse.ArgumentParser(
        prog="cgc gate5",
        description="CGC Gate 5.0 CLI - 可审计、可追踪、可回溯、可可视化",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 5.0.0",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )

    add_task_subparser(subparsers)
    add_audit_subparser(subparsers)
    add_trace_subparser(subparsers)
    add_config_subparser(subparsers)

    return parser


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
