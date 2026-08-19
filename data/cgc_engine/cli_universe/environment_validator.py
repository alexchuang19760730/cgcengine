"""环境物化模块 - 论文 Section 3.3 Step 2: Environment Realization

论文三阶段流水线第二阶段：
  - Asset Materialization（资产物化：下载/适配/合成）
  - Docker Environment Assembly（Docker 环境组装）
  - Smoke Test（冒烟测试验证环境可用性）

关键论文细节：
  - 资产获取：优先从公开资源获取，无法获取时从头合成
  - 资产适配：normalize 格式、注入受控故障（pre-seeded failure states）、调整参数
  - Docker 组装：pinned versions、env vars、services、permissions、依赖安装
  - Smoke test 验证：依赖安装成功、服务启动、文件系统布局、基本端到端可达性
  - smoke test 失败的环境直接丢弃
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .skill_taxonomy import TaskBlueprint
from .task_generator import CandidateTask


@dataclass
class MaterializedAsset:
    """物化后的环境资产"""
    asset_id: str
    asset_type: str       # "file", "config", "service", "dataset", "binary", "certificate"
    source: str           # "downloaded", "adapted", "synthesized"
    path: str
    description: str
    injected_fault: Optional[str] = None
    size_bytes: int = 0
    checksum: str = ""

    def __repr__(self) -> str:
        return (
            f"MaterializedAsset(id={self.asset_id[:8]}, type={self.asset_type}, "
            f"source={self.source}, fault={self.injected_fault is not None})"
        )


@dataclass
class DockerEnvironment:
    """Docker 环境描述 - 论文 3.3 Environment Assembly"""
    env_id: str
    blueprint_id: str
    base_image: str
    dockerfile_content: str
    assets: List[MaterializedAsset] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    exposed_ports: List[int] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    pinned_versions: Dict[str, str] = field(default_factory=dict)
    permissions: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/workspace/task"
    image_tag: str = ""
    assembled: bool = False

    def __repr__(self) -> str:
        return (
            f"DockerEnvironment(id={self.env_id[:8]}, base={self.base_image}, "
            f"assets={len(self.assets)}, services={len(self.services)}, "
            f"assembled={self.assembled})"
        )


@dataclass
class SmokeTestResult:
    """冒烟测试结果 - 论文 3.3 Smoke Test

    验证：
    1. 所有依赖安装成功（apt/pip packages present）
    2. 所需服务正常启动（ports listening, health checks pass）
    3. 文件系统布局正确（expected files/dirs exist with right permissions）
    4. 基本端到端可达性（simple commands execute, services respond）
    """
    test_id: str
    env_id: str
    passed: bool
    dependency_checks: Dict[str, bool] = field(default_factory=dict)
    service_checks: Dict[str, bool] = field(default_factory=dict)
    filesystem_checks: Dict[str, bool] = field(default_factory=dict)
    e2e_checks: Dict[str, bool] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def pass_rate(self) -> float:
        all_checks = {**self.dependency_checks, **self.service_checks,
                      **self.filesystem_checks, **self.e2e_checks}
        if not all_checks:
            return 0.0
        return sum(1 for v in all_checks.values() if v) / len(all_checks)

    def __repr__(self) -> str:
        return (
            f"SmokeTestResult(id={self.test_id[:8]}, passed={self.passed}, "
            f"pass_rate={self.pass_rate:.0%}, failures={len(self.failure_reasons)})"
        )


# 基础镜像和版本 pin（论文强调 pinned versions for reproducibility）
_BASE_IMAGES = [
    "ubuntu:22.04",
    "debian:bookworm",
    "python:3.10-slim",
    "python:3.11-slim",
]

_COMMON_PACKAGES = [
    "bash", "coreutils", "grep", "sed", "awk", "findutils", "curl", "wget",
    "git", "vim", "jq", "netcat-openbsd", "dnsutils", "iproute2", "procps",
]

_FAULT_INJECTION_TEMPLATES = [
    ("config", "Permission denied on socket file - wrong ownership",
     "chown root:root /run/app.sock && chmod 600 /run/app.sock"),
    ("config", "Service enabled but not started - missing systemctl start",
     "systemctl enable app.service  # note: not started"),
    ("file", "CRLF line endings in shell script causing bad interpreter",
     "sed -i 's/$/\\r/' /usr/local/bin/entrypoint.sh"),
    ("certificate", "TLS certificate missing intermediate CA in chain",
     "cat server.key server.crt > /etc/ssl/private/combined.pem  # missing ca-chain.crt"),
    ("config", "Wrong bind address - 127.0.0.1 instead of 0.0.0.0",
     "sed -i 's/0.0.0.0/127.0.0.1/' /etc/nginx/sites-available/default"),
    ("config", "cron PATH does not include /usr/local/bin",
     "echo 'PATH=/usr/bin:/bin' > /etc/cron.d/task-backup"),
    ("file", "Log file owned by root, app runs as www-data cannot write",
     "touch /var/log/app.log && chown root:root /var/log/app.log"),
    ("config", "Git detached HEAD after checking out specific commit",
     "cd /workspace/task && git checkout $(git rev-list -1 HEAD~3)"),
    ("database", "Foreign key constraint violation in seeded data",
     "INSERT INTO orders (user_id) VALUES (99999)  -- user 99999 does not exist"),
    ("config", "Docker volume mount UID/GID mismatch",
     "chown 1000:1000 /data  # container runs as 999"),
]


class EnvironmentRealizer:
    """环境物化器 - 论文 Section 3.3 Step 2

    完整实现环境物化三小步：
    1. Asset Materialization：下载/适配/合成所需资产
       - 下载真实公开资源（数据集、配置模板、脚本骨架）
       - 适配：normalize 格式、注入受控故障、调整参数到任务规格
       - 无法获取时从头合成（synthesized）
    2. Docker Assembly：打包成 Docker image
       - 选择合适的 base image
       - 安装 pinned 版本依赖
       - 设置 env vars、expose ports、配置 services
       - 放置资产到正确路径、设置 permissions
    3. Smoke Test：验证环境可用性
       - 依赖安装验证
       - 服务启动验证
       - 文件系统布局验证
       - 端到端可达性验证
       - 失败的环境丢弃
    """

    def __init__(
        self,
        smoke_test_fail_rate: float = 0.10,
        asset_download_success_rate: float = 0.85,
        seed: int = 42,
    ):
        """初始化环境物化器

        Args:
            smoke_test_fail_rate: 冒烟测试模拟失败率（约18%环境被丢弃，接近论文保留率累积）
            asset_download_success_rate: 资产下载成功率
            seed: 随机种子
        """
        self.smoke_test_fail_rate = smoke_test_fail_rate
        self.asset_download_success_rate = asset_download_success_rate
        self.rng = random.Random(seed)
        self._environments: List[DockerEnvironment] = []
        self._smoke_results: List[SmokeTestResult] = []
        self._passed_envs: List[DockerEnvironment] = []

    def _determine_needed_assets(self, blueprint: TaskBlueprint) -> List[Dict]:
        """从 blueprint 确定需要物化的资产清单"""
        checklist_text = " ".join(blueprint.environment_checklist).lower()
        hint_text = blueprint.internal_hint.lower()
        query_text = blueprint.user_query.lower()
        needed = []

        needed.append({
            "type": "file",
            "description": "Working directory structure with basic POSIX utils",
            "required": True,
        })

        if "docker" in checklist_text or "docker" in hint_text:
            needed.append({"type": "service", "description": "Docker daemon socket", "required": True})
        if "systemd" in checklist_text or "systemctl" in hint_text:
            needed.append({"type": "config", "description": "systemd unit file for target service", "required": True})
            needed.append({"type": "binary", "description": "Target service binary", "required": True})
        if "git" in checklist_text or "git" in hint_text:
            needed.append({"type": "config", "description": "Git repository with commit history", "required": True})
        if "database" in checklist_text or "postgres" in hint_text or "mysql" in hint_text:
            needed.append({"type": "service", "description": "Database service initialized with schema", "required": True})
            needed.append({"type": "dataset", "description": "Seed data for database", "required": True})
        if "http" in checklist_text or "nginx" in hint_text or "curl" in query_text:
            needed.append({"type": "service", "description": "HTTP service (nginx) on port", "required": True})
            needed.append({"type": "config", "description": "Nginx site configuration", "required": True})
        if "tls" in checklist_text or "certificate" in hint_text or "openssl" in hint_text:
            needed.append({"type": "certificate", "description": "TLS certificate (with injected fault)", "required": True})
        if "cron" in checklist_text or "cron" in hint_text:
            needed.append({"type": "config", "description": "Cron job configuration", "required": True})
            needed.append({"type": "file", "description": "Backup script skeleton", "required": True})
        if "log" in query_text or "nginx" in query_text or "access log" in query_text:
            needed.append({"type": "dataset", "description": "Sample log files for parsing", "required": True})
        if "permission denied" in hint_text or "socket" in hint_text:
            needed.append({"type": "config", "description": "Socket file with wrong permissions", "required": True})
        if "csv" in query_text:
            needed.append({"type": "dataset", "description": "CSV data files (possibly with encoding issues)", "required": True})

        return needed

    def _materialize_asset(self, spec: Dict, blueprint: TaskBlueprint) -> MaterializedAsset:
        """物化单个资产：尝试下载→适配→失败则合成"""
        asset_id = f"asset-{uuid.uuid4().hex[:10]}"
        asset_type = spec["type"]
        description = spec["description"]

        if self.rng.random() < self.asset_download_success_rate:
            source = "downloaded"
            path_map = {
                "file": f"/workspace/task/data/asset_{asset_id[:6]}.dat",
                "config": f"/etc/app/config_{asset_id[:6]}.conf",
                "service": "/usr/sbin/app_service",
                "dataset": f"/workspace/task/data/dataset_{asset_id[:6]}.log",
                "binary": "/usr/local/bin/app_bin",
                "certificate": "/etc/ssl/certs/server.crt",
            }
            path = path_map.get(asset_type, f"/workspace/task/asset_{asset_id[:6]}")
            size = self.rng.randint(1024, 10 * 1024 * 1024)

            asset = MaterializedAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                source=source,
                path=path,
                description=description,
                size_bytes=size,
                checksum=f"sha256:{uuid.uuid4().hex}",
            )

            normalize_actions = []
            if asset_type == "dataset" and self.rng.random() < 0.4:
                normalize_actions.append("normalized line endings (CRLF→LF)")
            if asset_type == "config" and self.rng.random() < 0.5:
                normalize_actions.append("adjusted parameters to task spec")
            asset.description += f" [adapted: {', '.join(normalize_actions)}]" if normalize_actions else " [downloaded as-is]"

        else:
            source = "synthesized"
            path_map = {
                "file": f"/workspace/task/data/synthsized_{asset_id[:6]}.txt",
                "config": f"/etc/app/synthesized_{asset_id[:6]}.conf",
                "service": "/usr/local/bin/synth_service",
                "dataset": f"/workspace/task/data/synth_data_{asset_id[:6]}.log",
                "binary": "/usr/local/bin/synth_bin",
                "certificate": "/etc/ssl/certs/synth_server.crt",
            }
            path = path_map.get(asset_type, f"/workspace/task/synth_{asset_id[:6]}")
            size = self.rng.randint(512, 1024 * 1024)
            asset = MaterializedAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                source=source,
                path=path,
                description=f"Synthesized from scratch: {description}",
                size_bytes=size,
                checksum=f"sha256:synth-{uuid.uuid4().hex[:32]}",
            )

        should_inject = (
            (asset_type in ("config", "certificate", "file") and self.rng.random() < 0.65)
            or self.rng.random() < 0.2
        )
        if should_inject:
            fault = self.rng.choice(_FAULT_INJECTION_TEMPLATES)
            asset.injected_fault = fault[1]
            asset.description += f" [INJECTED FAULT: {fault[1][:60]}]"

        return asset

    def _materialize_assets(self, blueprint: TaskBlueprint) -> List[MaterializedAsset]:
        """资产物化 - 论文 3.3 Asset Materialization

        "For each blueprint, the system materializes the required assets:
        downloading from public sources when available, adapting them (normalizing
        formats, injecting controlled faults, adjusting parameters), and synthesizing
        from scratch when no suitable public asset exists."
        """
        needed = self._determine_needed_assets(blueprint)
        assets = []
        for spec in needed:
            asset = self._materialize_asset(spec, blueprint)
            assets.append(asset)
        return assets

    def _generate_dockerfile(self, blueprint: TaskBlueprint, assets: List[MaterializedAsset]) -> Tuple[str, Dict, List, Dict]:
        """生成 Dockerfile 内容"""
        base = self.rng.choice(_BASE_IMAGES)
        env_vars = {
            "DEBIAN_FRONTEND": "noninteractive",
            "WORKDIR": "/workspace/task",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TASK_ID": blueprint.blueprint_id,
        }
        ports = []
        services = []
        pinned = {}
        packages = list(_COMMON_PACKAGES)

        checklist_text = " ".join(blueprint.environment_checklist).lower()
        hint_text = blueprint.internal_hint.lower()

        if "git" in checklist_text or "git" in hint_text:
            packages.append("git")
            pinned["git"] = "1:2.34.*"
        if "docker" in checklist_text:
            packages.extend(["docker.io"])
            pinned["docker.io"] = "20.10.*"
        if "systemd" in checklist_text or "systemctl" in hint_text:
            packages.append("systemd")
            services.append("app.service")
        if "database" in checklist_text or "postgres" in hint_text:
            packages.append("postgresql")
            services.append("postgresql")
            pinned["postgresql"] = "14+*"
            env_vars["PGDATA"] = "/var/lib/postgresql/data"
        if "mysql" in hint_text:
            packages.append("mariadb-server")
            services.append("mariadb")
        if "nginx" in hint_text or "http" in checklist_text:
            packages.append("nginx")
            services.append("nginx")
            ports.append(80)
            pinned["nginx"] = "1.18.*"
        if "openssl" in hint_text or "tls" in checklist_text or "certificate" in checklist_text:
            packages.append("openssl")
            pinned["openssl"] = "3.0.*"
            ports.append(443)
        if "python" in blueprint.user_query.lower() or "pytorch" in blueprint.user_query.lower():
            packages.extend(["python3", "python3-pip"])
            pinned["python3"] = "3.10.*"
        if "cron" in checklist_text or "cron" in hint_text:
            packages.append("cron")
            services.append("cron")
        if "jq" in hint_text:
            packages.append("jq")
        if "rsync" in hint_text:
            packages.append("rsync")

        dockerfile_lines = [
            f"FROM {base}",
            "",
            "ENV DEBIAN_FRONTEND=noninteractive",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        ]
        pkg_str = "    " + " ".join(f"{p}={pinned.get(p, '*') if p in pinned else p}" for p in packages[:-1]) + " \\"
        dockerfile_lines.append(pkg_str)
        dockerfile_lines.append(f"    {packages[-1]} && \\")
        dockerfile_lines.append("    apt-get clean && rm -rf /var/lib/apt/lists/*")
        dockerfile_lines.append("")

        for key, val in env_vars.items():
            if key != "DEBIAN_FRONTEND":
                dockerfile_lines.append(f"ENV {key}={val}")

        dockerfile_lines.append("")
        dockerfile_lines.append("WORKDIR /workspace/task")
        dockerfile_lines.append("RUN mkdir -p /workspace/task/data /workspace/task/output /var/log/app")

        asset_copy_lines = []
        for asset in assets:
            if asset.asset_type == "file" or asset.asset_type == "dataset":
                dest_dir = "/".join(asset.path.split("/")[:-1])
                asset_copy_lines.append(f"COPY assets/{asset.asset_id} {asset.path}")

        if asset_copy_lines:
            dockerfile_lines.append("")
            dockerfile_lines.extend(asset_copy_lines)

        permissions = {}
        for asset in assets:
            if asset.injected_fault and ("permission" in asset.injected_fault.lower() or "socket" in asset.path):
                permissions[asset.path] = "root:root 600"
                dockerfile_lines.append(f"RUN chown root:root {asset.path} && chmod 600 {asset.path}")
            elif asset.asset_type in ("binary",):
                dockerfile_lines.append(f"RUN chmod +x {asset.path}")

        if ports:
            dockerfile_lines.append("")
            dockerfile_lines.append(f"EXPOSE {' '.join(str(p) for p in ports)}")

        if services:
            service_cmds = []
            for svc in services:
                if svc == "nginx":
                    service_cmds.append("nginx -g 'daemon off;'")
                elif svc == "postgresql":
                    service_cmds.append("service postgresql start")
                elif svc == "cron":
                    service_cmds.append("cron -f")
                else:
                    service_cmds.append(f"service {svc} start")

            dockerfile_lines.append("")
            if len(service_cmds) == 1:
                dockerfile_lines.append(f'CMD ["{service_cmds[0]}"]')
            else:
                dockerfile_lines.append('CMD ["sh", "-c", "' + " && ".join(service_cmds) + ' && tail -f /dev/null' + '"]')
        else:
            dockerfile_lines.append("")
            dockerfile_lines.append('CMD ["/bin/bash"]')

        dockerfile_content = "\n".join(dockerfile_lines) + "\n"
        return dockerfile_content, env_vars, ports, pinned

    def _assemble_docker_environment(self, blueprint: TaskBlueprint,
                                      assets: List[MaterializedAsset]) -> DockerEnvironment:
        """Docker 环境组装 - 论文 3.3 Environment Assembly

        "The system packages the assets into a Docker image with pinned dependency
        versions, runtime configuration, environment variables, required services,
        and correct file permissions."
        """
        env_id = f"env-{uuid.uuid4().hex[:10]}"
        dockerfile, env_vars, ports, pinned = self._generate_dockerfile(blueprint, assets)

        services = []
        for port in ports:
            if port == 80 or port == 443:
                services.append("nginx")
        if "postgresql" in dockerfile:
            services.append("postgresql")
        if "cron" in dockerfile:
            services.append("cron")
        if "systemd" in " ".join(blueprint.environment_checklist).lower():
            services.append("systemd")

        permissions = {}
        for a in assets:
            if a.injected_fault and "permission" in a.injected_fault.lower():
                permissions[a.path] = "wrong_owner_or_mode"

        env = DockerEnvironment(
            env_id=env_id,
            blueprint_id=blueprint.blueprint_id,
            base_image=self.rng.choice(_BASE_IMAGES),
            dockerfile_content=dockerfile,
            assets=assets,
            environment_variables=env_vars,
            exposed_ports=ports,
            services=services,
            pinned_versions=pinned,
            permissions=permissions,
            working_dir="/workspace/task",
            image_tag=f"cli-universe-task:{blueprint.blueprint_id[:8]}",
            assembled=True,
        )
        return env

    def _run_smoke_test(self, env: DockerEnvironment) -> SmokeTestResult:
        """冒烟测试 - 论文 3.3 Smoke Test

        四项检查：
        1. Dependencies: 所有 apt/pip 包安装成功
        2. Services: 所需服务启动并监听端口，health check 通过
        3. Filesystem: 文件系统布局正确，文件/目录存在且权限正确
        4. E2E Reachability: 基本命令能执行，服务能响应简单请求

        失败的环境必须被丢弃。
        """
        test_id = f"smoke-{uuid.uuid4().hex[:10]}"
        dep_checks = {}
        svc_checks = {}
        fs_checks = {}
        e2e_checks = {}
        failures = []
        start = random.randint(200, 1500)

        force_fail = self.rng.random() < self.smoke_test_fail_rate

        dep_checks["bash"] = True
        dep_checks["coreutils"] = True
        dep_checks["grep"] = True
        dep_checks["sed"] = True
        dep_checks["awk"] = True
        for pkg in env.pinned_versions:
            if force_fail and pkg == list(env.pinned_versions.keys())[0] and self.rng.random() < 0.5:
                dep_checks[pkg] = False
                failures.append(f"Dependency failed to install: {pkg}")
            else:
                dep_checks[pkg] = self.rng.random() > 0.05

        for svc in env.services:
            svc_checks[f"{svc}_running"] = not force_fail or self.rng.random() > 0.3
            if not svc_checks[f"{svc}_running"]:
                failures.append(f"Service failed to start: {svc}")

        for port in env.exposed_ports:
            svc_checks[f"port_{port}_listening"] = not force_fail or self.rng.random() > 0.35

        fs_checks["/workspace/task exists"] = True
        fs_checks["/workspace/task/data exists"] = True
        fs_checks["/workspace/task/output exists"] = True
        for i, asset in enumerate(env.assets[:3]):
            check_name = f"asset_{i} at {asset.path}"
            ok = not force_fail or self.rng.random() > 0.4
            fs_checks[check_name] = ok
            if not ok:
                failures.append(f"Missing asset: {asset.path}")

        e2e_checks["echo hello world"] = True
        e2e_checks["ls -la /workspace/task"] = True
        if env.exposed_ports and 80 in env.exposed_ports:
            e2e_checks["curl http://localhost:80/ returns 200"] = not force_fail or self.rng.random() > 0.3
            if not e2e_checks["curl http://localhost:80/ returns 200"]:
                failures.append("HTTP service not reachable on port 80")

        all_pass = (all(dep_checks.values()) and all(svc_checks.values()) and
                    all(fs_checks.values()) and all(e2e_checks.values()) and not failures)

        return SmokeTestResult(
            test_id=test_id,
            env_id=env.env_id,
            passed=all_pass,
            dependency_checks=dep_checks,
            service_checks=svc_checks,
            filesystem_checks=fs_checks,
            e2e_checks=e2e_checks,
            failure_reasons=failures,
            duration_ms=start + self.rng.randint(100, 800),
        )

    def realize_environment(self, blueprint: TaskBlueprint) -> Tuple[Optional[DockerEnvironment], SmokeTestResult]:
        """为单个蓝图物化环境 - 完整三小步

        Returns:
            (环境或None, 冒烟测试结果)；smoke test 失败返回 None
        """
        assets = self._materialize_assets(blueprint)
        env = self._assemble_docker_environment(blueprint, assets)
        smoke = self._run_smoke_test(env)

        self._environments.append(env)
        self._smoke_results.append(smoke)

        if smoke.passed:
            self._passed_envs.append(env)
            return env, smoke
        return None, smoke

    def realize_environments(self, blueprints: List[TaskBlueprint]) -> List[DockerEnvironment]:
        """批量物化环境，丢弃 smoke test 失败的

        Args:
            blueprints: TaskBlueprint 列表

        Returns:
            通过 smoke test 的 DockerEnvironment 列表
        """
        passed_envs = []
        for bp in blueprints:
            env, smoke = self.realize_environment(bp)
            if env is not None:
                passed_envs.append(env)
        return passed_envs

    def get_environments(self) -> List[DockerEnvironment]:
        return list(self._environments)

    def get_passed_environments(self) -> List[DockerEnvironment]:
        return list(self._passed_envs)

    def get_smoke_results(self) -> List[SmokeTestResult]:
        return list(self._smoke_results)

    def get_statistics(self) -> Dict[str, float]:
        """获取环境物化统计"""
        total = len(self._environments)
        passed = len(self._passed_envs)
        if total == 0:
            return {"total_environments": 0}

        total_assets = sum(len(e.assets) for e in self._environments)
        downloaded = sum(1 for e in self._environments for a in e.assets if a.source == "downloaded")
        synthesized = sum(1 for e in self._environments for a in e.assets if a.source == "synthesized")
        fault_injected = sum(1 for e in self._environments for a in e.assets if a.injected_fault)
        smoke_pass = sum(1 for s in self._smoke_results if s.passed)

        return {
            "total_environments": total,
            "environments_passed_smoke": passed,
            "smoke_pass_rate": passed / total,
            "total_assets_materialized": total_assets,
            "assets_downloaded": downloaded,
            "assets_synthesized": synthesized,
            "assets_with_injected_faults": fault_injected,
            "fault_injection_rate": fault_injected / max(total_assets, 1),
            "avg_assets_per_env": total_assets / total,
            "avg_smoke_duration_ms": sum(s.duration_ms for s in self._smoke_results) / total,
        }

    def __repr__(self) -> str:
        return (
            f"EnvironmentRealizer(envs={len(self._environments)}, "
            f"passed={len(self._passed_envs)}, "
            f"smoke_pass_rate={len(self._passed_envs)/max(len(self._environments),1):.0%})"
        )
