"""
Master/Slave 代码同步机制
=========================
支持 slave 节点定时从 master 节点同步代码，保证端端/端云代码一致性。

同步方式：
1. git 同步：slave 执行 git pull 从 master 远程仓库拉取
2. 文件同步：通过 HTTP/SMB/rsync 从 master 复制文件
3. 增量同步：只同步变更的文件（基于 hash 对比）

使用方式：
    # 作为 slave 启动后台同步
    python master_slave_sync.py --role slave --master http://192.168.1.101:9000 --interval 300

    # 作为 master 提供同步服务
    python master_slave_sync.py --role master --port 9000

    # Python API
    from agent_harness.pd.master_slave_sync import SyncConfig, CodeSyncer
    syncer = CodeSyncer(SyncConfig(role="slave", master_url="http://..."))
    syncer.sync_once()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SyncRole(str, Enum):
    MASTER = "master"
    SLAVE = "slave"


class SyncMethod(str, Enum):
    GIT = "git"              # git pull 同步
    FILE = "file"            # 文件复制同步
    HTTP = "http"            # HTTP API 同步
    INCREMENTAL = "incremental"  # 增量 hash 同步


@dataclass
class SyncConfig:
    """同步配置"""
    role: SyncRole = SyncRole.SLAVE
    method: SyncMethod = SyncMethod.INCREMENTAL

    # Master 信息
    master_url: str = ""              # master HTTP API 地址
    master_path: str = ""             # master 本地路径（file 方式）
    master_git_remote: str = "origin" # git 远程名
    master_git_branch: str = "main"   # git 分支

    # Slave 信息
    local_path: str = ""              # 本地代码根目录
    slave_id: str = ""                # slave 唯一标识

    # 同步策略
    sync_interval: int = 300          # 同步间隔（秒）
    sync_directories: List[str] = field(default_factory=lambda: [
        "pd", "loopmoe", "qwen36", "agents", "finetune", "scripts", "config.env"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", ".venv", "node_modules", "*.pyc",
        "results/", "logs/", "*.log", "data/", "checkpoints/"
    ])
    auto_restart_after_sync: bool = False  # 同步后自动重启服务

    # 验证
    verify_after_sync: bool = True
    verify_test_command: str = ""     # 验证命令（如 pytest）

    # 冲突处理
    conflict_strategy: str = "master_wins"  # master_wins / slave_wins / skip / manual


@dataclass
class FileState:
    """文件状态（用于增量同步）"""
    path: str
    md5: str
    size: int
    mtime: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "md5": self.md5,
            "size": self.size,
            "mtime": self.mtime,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileState":
        return cls(
            path=d["path"],
            md5=d["md5"],
            size=d["size"],
            mtime=d["mtime"],
        )


@dataclass
class SyncResult:
    """同步结果"""
    success: bool
    files_updated: int = 0
    files_added: int = 0
    files_deleted: int = 0
    files_skipped: int = 0
    conflicts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    master_version: str = ""
    slave_version: str = ""

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{status}] updated={self.files_updated} added={self.files_added} "
            f"deleted={self.files_deleted} skipped={self.files_skipped} "
            f"conflicts={len(self.conflicts)} errors={len(self.errors)} "
            f"in {self.duration_seconds:.1f}s"
        )


class CodeSyncer:
    """代码同步器"""

    def __init__(self, config: SyncConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self._last_sync: Optional[SyncResult] = None

        if not self.config.local_path:
            self.config.local_path = str(Path(__file__).parent.parent.resolve())

        if not self.config.slave_id:
            import socket
            self.config.slave_id = f"{socket.gethostname()}-{os.getpid()}"

    # ------------------------------------------------------------------
    # 文件扫描与 hash
    # ------------------------------------------------------------------

    def _should_exclude(self, path: str) -> bool:
        """检查路径是否应被排除"""
        for pattern in self.config.exclude_patterns:
            if pattern.startswith("*"):
                if path.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("/"):
                if pattern in path.replace("\\", "/"):
                    return True
            elif pattern in path:
                return True
        return False

    def _scan_directory(self, root: str, base: str = "") -> List[FileState]:
        """扫描目录，返回文件状态列表"""
        states = []
        root_path = Path(root)
        for item in root_path.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(root_path)).replace("\\", "/")
                if base:
                    rel_path = f"{base}/{rel_path}"
                if self._should_exclude(rel_path):
                    continue
                try:
                    md5 = hashlib.md5(item.read_bytes()).hexdigest()
                    states.append(FileState(
                        path=rel_path,
                        md5=md5,
                        size=item.stat().st_size,
                        mtime=item.stat().st_mtime,
                    ))
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot read {rel_path}: {e}")
        return states

    def scan_local(self) -> List[FileState]:
        """扫描本地文件状态"""
        all_states = []
        for sync_dir in self.config.sync_directories:
            dir_path = os.path.join(self.config.local_path, sync_dir)
            if os.path.isdir(dir_path):
                all_states.extend(self._scan_directory(dir_path, sync_dir))
            elif os.path.isfile(dir_path):
                item = Path(dir_path)
                md5 = hashlib.md5(item.read_bytes()).hexdigest()
                all_states.append(FileState(
                    path=sync_dir,
                    md5=md5,
                    size=item.stat().st_size,
                    mtime=item.stat().st_mtime,
                ))
        return all_states

    # ------------------------------------------------------------------
    # 同步方式实现
    # ------------------------------------------------------------------

    def _sync_git(self) -> SyncResult:
        """Git 方式同步"""
        import subprocess
        result = SyncResult(success=False)
        start = time.time()

        try:
            # 获取当前版本
            slave_ver = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.local_path,
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
            result.slave_version = slave_ver

            # git fetch
            subprocess.check_output(
                ["git", "fetch", self.config.master_git_remote, self.config.master_git_branch],
                cwd=self.config.local_path,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # 获取 master 版本
            master_ver = subprocess.check_output(
                ["git", "rev-parse", f"{self.config.master_git_remote}/{self.config.master_git_branch}"],
                cwd=self.config.local_path,
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
            result.master_version = master_ver

            if master_ver == slave_ver:
                result.success = True
                result.files_skipped = -1  # 表示已是最新
                result.duration_seconds = time.time() - start
                logger.info("Already up to date")
                return result

            # git pull (merge)
            pull_output = subprocess.check_output(
                ["git", "pull", "--no-edit", self.config.master_git_remote, self.config.master_git_branch],
                cwd=self.config.local_path,
                stderr=subprocess.STDOUT,
                text=True,
            )
            result.success = True

            # 统计变更文件数
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", slave_ver, "HEAD"],
                cwd=self.config.local_path,
                stderr=subprocess.STDOUT,
                text=True,
            )
            changed = [f for f in diff_output.strip().split("\n") if f]
            result.files_updated = len(changed)

        except subprocess.CalledProcessError as e:
            result.errors.append(f"Git command failed: {e.output}")
            if "CONFLICT" in e.output or "conflict" in e.output:
                result.conflicts.append("Merge conflict detected")
        except Exception as e:
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _sync_incremental(self) -> SyncResult:
        """增量 hash 同步（通过 HTTP API 获取 master 文件列表）"""
        result = SyncResult(success=False)
        start = time.time()

        if not self.config.master_url:
            result.errors.append("master_url is required for HTTP incremental sync")
            result.duration_seconds = time.time() - start
            return result

        try:
            import urllib.request

            # 1. 从 master 获取文件清单
            manifest_url = f"{self.config.master_url.rstrip('/')}/v1/sync/manifest"
            req = urllib.request.Request(manifest_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                manifest_data = json.loads(resp.read())

            master_states = {
                fs["path"]: FileState.from_dict(fs)
                for fs in manifest_data.get("files", [])
            }
            result.master_version = manifest_data.get("version", "")

            # 2. 扫描本地文件
            local_states = {fs.path: fs for fs in self.scan_local()}

            # 3. 计算差异
            to_update = []
            to_add = []
            to_delete = []

            for path, m_state in master_states.items():
                if path in local_states:
                    if local_states[path].md5 != m_state.md5:
                        to_update.append(path)
                else:
                    to_add.append(path)

            for path in local_states:
                if path not in master_states:
                    to_delete.append(path)

            # 4. 执行同步
            for path in to_update + to_add:
                try:
                    file_url = f"{self.config.master_url.rstrip('/')}/v1/sync/file?path={path}"
                    req = urllib.request.Request(file_url)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        content = resp.read()

                    local_file = os.path.join(self.config.local_path, path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    with open(local_file, "wb") as f:
                        f.write(content)

                    if path in to_update:
                        result.files_updated += 1
                    else:
                        result.files_added += 1
                except Exception as e:
                    result.errors.append(f"Failed to sync {path}: {e}")

            # 删除 master 不存在的文件（可选，默认跳过）
            for path in to_delete:
                if self.config.conflict_strategy == "master_wins":
                    local_file = os.path.join(self.config.local_path, path)
                    if os.path.exists(local_file):
                        os.remove(local_file)
                        result.files_deleted += 1
                else:
                    result.files_skipped += 1

            result.success = len(result.errors) == 0
            result.slave_version = hashlib.md5(
                json.dumps(sorted(local_states.keys())).encode()
            ).hexdigest()[:12]

        except Exception as e:
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _sync_file(self) -> SyncResult:
        """文件复制方式同步（master_path → local_path）"""
        result = SyncResult(success=False)
        start = time.time()

        if not self.config.master_path:
            result.errors.append("master_path is required for file sync")
            result.duration_seconds = time.time() - start
            return result

        try:
            for sync_dir in self.config.sync_directories:
                src = os.path.join(self.config.master_path, sync_dir)
                dst = os.path.join(self.config.local_path, sync_dir)

                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                        *self.config.exclude_patterns
                    ))
                    result.files_updated += sum(
                        1 for _ in Path(dst).rglob("*") if _.is_file()
                    )
                elif os.path.isfile(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    result.files_updated += 1

            result.success = True
        except Exception as e:
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def sync_once(self) -> SyncResult:
        """执行一次同步"""
        logger.info(f"Starting sync (method={self.config.method.value}, role={self.config.role.value})")

        if self.config.role == SyncRole.MASTER:
            result = SyncResult(success=True)
            result.master_version = self._get_local_version()
            logger.info("Master node: serving manifest, no sync needed")
            self._last_sync = result
            return result

        if self.config.method == SyncMethod.GIT:
            result = self._sync_git()
        elif self.config.method == SyncMethod.FILE:
            result = self._sync_file()
        elif self.config.method in (SyncMethod.HTTP, SyncMethod.INCREMENTAL):
            result = self._sync_incremental()
        else:
            result = SyncResult(success=False, errors=[f"Unknown sync method: {self.config.method}"])

        # 验证
        if result.success and self.config.verify_after_sync and self.config.verify_test_command:
            try:
                import subprocess
                subprocess.check_output(
                    self.config.verify_test_command,
                    cwd=self.config.local_path,
                    shell=True,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=120,
                )
            except Exception as e:
                result.errors.append(f"Verification failed: {e}")
                result.success = False

        # 自动重启
        if result.success and self.config.auto_restart_after_sync and result.files_updated > 0:
            logger.info("Files updated, auto-restart triggered")
            # 重启逻辑由调用方实现（发送信号等）

        self._last_sync = result
        logger.info(result.summary())
        return result

    def start_background_sync(self) -> None:
        """启动后台同步线程"""
        if self._sync_thread and self._sync_thread.is_alive():
            logger.warning("Background sync already running")
            return

        self._stop_event.clear()

        def _sync_loop():
            while not self._stop_event.is_set():
                try:
                    self.sync_once()
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
                self._stop_event.wait(self.config.sync_interval)

        self._sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info(f"Background sync started (interval={self.config.sync_interval}s)")

    def stop_background_sync(self) -> None:
        """停止后台同步"""
        self._stop_event.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("Background sync stopped")

    def get_manifest(self) -> Dict[str, Any]:
        """获取本地文件清单（master 提供给 slave）"""
        states = self.scan_local()
        return {
            "version": self._get_local_version(),
            "slave_id": self.config.slave_id,
            "timestamp": time.time(),
            "file_count": len(states),
            "files": [fs.to_dict() for fs in states],
        }

    def _get_local_version(self) -> str:
        """获取本地版本标识"""
        try:
            import subprocess
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.config.local_path,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            states = self.scan_local()
            h = hashlib.md5()
            for fs in sorted(states, key=lambda x: x.path):
                h.update(f"{fs.path}:{fs.md5}".encode())
            return h.hexdigest()[:12]

    @property
    def last_sync(self) -> Optional[SyncResult]:
        return self._last_sync


# ----------------------------------------------------------------------
# Master HTTP 服务（提供文件清单和文件下载）
# ----------------------------------------------------------------------

def start_master_server(config: SyncConfig, host: str = "0.0.0.0", port: int = 9000) -> None:
    """启动 master 同步服务（FastAPI）"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, FileResponse
        import uvicorn
    except ImportError:
        logger.error("fastapi and uvicorn are required for master server")
        sys.exit(1)

    syncer = CodeSyncer(config)
    app = FastAPI(title="PD Code Sync Master", version="1.0")

    @app.get("/v1/sync/manifest")
    def get_manifest():
        return syncer.get_manifest()

    @app.get("/v1/sync/file")
    def get_file(path: str):
        local_file = os.path.join(config.local_path, path)
        if not os.path.exists(local_file) or not os.path.isfile(local_file):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if syncer._should_exclude(path):
            raise HTTPException(status_code=403, detail=f"File excluded: {path}")
        return FileResponse(local_file)

    @app.get("/v1/sync/status")
    def get_status():
        return {
            "role": config.role.value,
            "method": config.method.value,
            "slave_id": config.slave_id,
            "version": syncer._get_local_version(),
            "last_sync": syncer.last_sync.summary() if syncer.last_sync else None,
        }

    logger.info(f"Starting master sync server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PD Master/Slave Code Sync")
    parser.add_argument("--role", choices=["master", "slave"], default="slave")
    parser.add_argument("--method", choices=["git", "file", "http", "incremental"], default="incremental")
    parser.add_argument("--master-url", default="", help="Master HTTP API URL")
    parser.add_argument("--master-path", default="", help="Master local path (file sync)")
    parser.add_argument("--local-path", default="", help="Local code root")
    parser.add_argument("--interval", type=int, default=300, help="Sync interval in seconds")
    parser.add_argument("--port", type=int, default=9000, help="Master server port")
    parser.add_argument("--once", action="store_true", help="Sync once and exit")
    parser.add_argument("--verify", default="", help="Verification command after sync")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = SyncConfig(
        role=SyncRole(args.role),
        method=SyncMethod(args.method),
        master_url=args.master_url,
        master_path=args.master_path,
        local_path=args.local_path,
        sync_interval=args.interval,
        verify_test_command=args.verify,
    )

    if args.role == "master":
        start_master_server(config, port=args.port)
    elif args.once:
        syncer = CodeSyncer(config)
        result = syncer.sync_once()
        print(result.summary())
        sys.exit(0 if result.success else 1)
    else:
        syncer = CodeSyncer(config)
        syncer.start_background_sync()
        print(f"Slave sync running (interval={args.interval}s). Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            syncer.stop_background_sync()


if __name__ == "__main__":
    main()
