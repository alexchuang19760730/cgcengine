"""证据引导深度研究 - 论文 3.2 Evidence-Guided Refinement

论文原文："the agent searches real technical materials (repositories, documentation,
issue discussions, tutorials, and usage examples) and progressively incorporates
the evidence into the task specification, grounding it in specific tools, realistic
constraints, known failure modes, and concrete input/output contracts."

关键效果（论文图 2a）：
- 精炼后任务需要 3.45× 更多 solver turns
- pass rate 下降 13.3 个百分点（真实难度提升，而不仅是加长轨迹）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random


@dataclass
class ResearchEvidence:
    """一条研究证据 - 从真实技术资料中提取"""
    source_type: str       # "repo" / "doc" / "issue" / "tutorial" / "example"
    source_url: str        # 来源标识
    tool_name: str         # 涉及的具体工具
    failure_mode: str      # 已知失败模式
    constraint: str        # 现实约束
    snippet: str           # 关键片段
    relevance_score: float = 0.0


# 模拟真实技术资料库（实际应用中连接 GitHub/Docs/StackOverflow 等）
_TECHNICAL_KNOWLEDGE_BASE = {
    # ---------- 工具 + 失败模式 ----------
    "grep": {
        "failure_modes": [
            "forgets to use -r for recursive search",
            "confuses regex vs fixed strings (-F)",
            "misses binary files without -a",
            "exit code confusion: 1 = no matches (not error)",
        ],
        "constraints": ["large directories slow without -F", "POSIX vs GNU grep flags differ"],
        "tools_related": ["xargs", "find", "sed", "awk"],
    },
    "sed": {
        "failure_modes": [
            "in-place -i flag syntax differs on macOS vs Linux",
            "greedy regex causing unexpected replacements",
            "forgets -E for extended regex (groups)",
            "line-endings CRLF vs LF causing mismatch",
        ],
        "constraints": ["BSD sed vs GNU sed incompatibilities"],
        "tools_related": ["awk", "grep", "tr"],
    },
    "find": {
        "failure_modes": [
            "missing -print0 causes issues with spaces in filenames",
            "-exec {} + vs {} ; semantics confusion",
            "forgets -type f/d for file/directory filtering",
            "mtime/ctime/atime misunderstanding",
        ],
        "constraints": ["depth-first by default", "permission errors halt early"],
        "tools_related": ["xargs", "rm", "grep"],
    },
    "awk": {
        "failure_modes": [
            "field separator -F not set for CSV",
            "NR vs FNR confusion across files",
            "floating point precision in sum/avg",
            "print without newline (printf needed)",
        ],
        "constraints": ["variations: gawk/mawk/nawk"],
        "tools_related": ["sed", "sort", "uniq"],
    },
    "git": {
        "failure_modes": [
            "force push without lease (--force-with-lease safer)",
            "detached HEAD after checkout commit",
            "merge conflict resolution mistakes",
            "rebase on shared branches",
            "stash pop conflicts",
        ],
        "constraints": ["shallow clone missing history", "submodule init needed"],
        "tools_related": ["git", "gh"],
    },
    "docker": {
        "failure_modes": [
            "port mapping conflicts (-p host:container)",
            "volume mount permission denied (UID/GID)",
            "container exits immediately (missing -it or CMD)",
            "image pull rate limits",
            "network mode bridge vs host confusion",
        ],
        "constraints": ["disk space from dangling images", "resource limits"],
        "tools_related": ["docker-compose", "kubectl"],
    },
    "curl": {
        "failure_modes": [
            "missing -L for redirects",
            "forgets -s silent mode in scripts",
            "POST data encoding issues (-d vs --data-urlencode)",
            "certificate verification in private networks (-k)",
            "timeout not set causing hangs",
        ],
        "constraints": ["HTTP/2 requires --http2 flag", "proxy settings"],
        "tools_related": ["wget", "jq"],
    },
    "jq": {
        "failure_modes": [
            "forgets -r for raw strings (outputs quoted)",
            "array vs object access confusion (.[] vs .key)",
            "pipe vs comma operator precedence",
            "select filter not excluding nulls",
        ],
        "constraints": ["jq -n for null input"],
        "tools_related": ["curl"],
    },
    "ssh": {
        "failure_modes": [
            "host key verification failed (known_hosts)",
            "permission denied on .ssh/config or keys (chmod 600)",
            "agent forwarding not enabled (-A)",
            "tunnel -L/-R direction confusion",
        ],
        "constraints": ["connection timeouts", "ProxyJump for bastions"],
        "tools_related": ["scp", "rsync"],
    },
    "systemd/systemctl": {
        "failure_modes": [
            "forgets daemon-reload after editing unit",
            "service enabled but not started (enable vs start)",
            "logs not showing (journalctl -u needed)",
            "restart always vs on-failure semantics",
        ],
        "constraints": ["user vs system units"],
        "tools_related": ["journalctl"],
    },
    "cron": {
        "failure_modes": [
            "PATH not set in cron environment",
            "missing MAILTO causing silent failures",
            "relative paths not working",
            "weekday numbering (0=Sunday vs 7=Sunday)",
        ],
        "constraints": ["environment minimal", "no interactive shell"],
        "tools_related": ["crontab", "anacron"],
    },
    "openssl": {
        "failure_modes": [
            "certificate vs key mismatch",
            "PEM vs DER format confusion",
            "CN/SAN not matching hostname",
            "expiry date not checked",
        ],
        "constraints": ["different versions: 1.1 vs 3.x"],
        "tools_related": ["keytool"],
    },
    "mysql/psql": {
        "failure_modes": [
            "password in command line visible in ps",
            "socket vs TCP connection confusion",
            "foreign key constraints blocking operations",
            "transaction not committed (BEGIN without COMMIT)",
        ],
        "constraints": ["connection limits", "lock timeouts"],
        "tools_related": ["mysqldump", "pg_dump"],
    },
    "tar": {
        "failure_modes": [
            "forgets -z for gzip/-j for bzip2",
            "absolute paths stripped by default",
            "overwriting files without warning",
            "permission preservation (-p) needed",
        ],
        "constraints": ["large file support"],
        "tools_related": ["gzip", "bzip2"],
    },
    "rsync": {
        "failure_modes": [
            "trailing slash semantics (with vs without)",
            "permission preservation missing (-a)",
            "delete flag too aggressive",
            "bandwidth limit not set (--bwlimit)",
        ],
        "constraints": ["delta algorithm memory use"],
        "tools_related": ["scp"],
    },
}


# 真实工程场景种子（模拟 issue/tutorial/example）
_REAL_SCENARIOS: List[Dict] = [
    {
        "domain": "software_and_system",
        "skill": "debugging",
        "description": "Service fails to start after deploy; logs show permission denied on socket",
        "tools": ["systemctl", "journalctl", "ls", "chmod", "ss"],
        "failure_mode": "systemd unit file points to wrong socket path created with wrong user",
        "difficulty": "medium",
    },
    {
        "domain": "data_processing",
        "skill": "text_processing",
        "description": "Parse 10GB nginx access log, extract top 10 IPs by 5xx errors, aggregate per hour",
        "tools": ["awk", "sort", "uniq", "grep", "head"],
        "failure_mode": "awk field separator for nginx log needs escaping; memory blow-up on large sort",
        "difficulty": "hard",
    },
    {
        "domain": "networking_and_security",
        "skill": "cryptography",
        "description": "Rotate expired TLS cert on nginx, verify chain, reload without downtime",
        "tools": ["openssl", "nginx", "curl", "systemctl"],
        "failure_mode": "missing intermediate cert in chain causing browser errors; reload kills active connections",
        "difficulty": "hard",
    },
    {
        "domain": "machine_learning",
        "skill": "configuration",
        "description": "Debug training job OOM; set CUDA_VISIBLE_DEVICES, batch size, gradient accumulation correctly",
        "tools": ["nvidia-smi", "python", "torch.cuda", "ps"],
        "failure_mode": "GPU memory fragmentation; workers not releasing; gradient accumulation steps mismatch",
        "difficulty": "hard",
    },
    {
        "domain": "software_and_system",
        "skill": "version_control",
        "description": "Recover from detached HEAD after accidental checkout; preserve local commits",
        "tools": ["git", "git reflog", "git branch", "git cherry-pick"],
        "failure_mode": "reflog expired; commits not reachable; force gc run",
        "difficulty": "medium",
    },
    {
        "domain": "software_and_system",
        "skill": "configuration",
        "description": "Set up cron job for daily backup with correct PATH, logging, and lock to prevent overlap",
        "tools": ["cron", "flock", "rsync", "tar", "logger"],
        "failure_mode": "cron PATH minimal; no lock causing overlapping runs; relative paths fail",
        "difficulty": "medium",
    },
    {
        "domain": "data_processing",
        "skill": "database",
        "description": "Bulk import CSV into Postgres, handle errors, verify row count, create indexes",
        "tools": ["psql", "\copy", "CREATE INDEX", "ANALYZE"],
        "failure_mode": "COPY needs superuser or \copy; encoding mismatches; index creation locks table",
        "difficulty": "medium",
    },
    {
        "domain": "networking_and_security",
        "skill": "networking",
        "description": "Debug why curl to internal service fails; diagnose DNS/firewall/TLS/port issues step by step",
        "tools": ["curl", "dig", "nslookup", "telnet/nc", "openssl s_client", "iptables"],
        "failure_mode": "MTU blackhole; DNS cache stale; TLS SNI mismatch; firewall silent drop not reject",
        "difficulty": "expert",
    },
    {
        "domain": "devops",
        "skill": "systems",
        "description": "Clean up Docker environment: remove stopped containers, dangling images, unused volumes safely",
        "tools": ["docker", "docker system df", "docker volume prune"],
        "failure_mode": "prune removes volumes needed by stopped but not removed containers; image in use by other tags",
        "difficulty": "easy",
    },
    {
        "domain": "software_and_system",
        "skill": "file_manipulation",
        "description": "Batch rename files with spaces and special chars; replace spaces with underscores recursively",
        "tools": ["find", "rename", "sed", "xargs", "-print0"],
        "failure_mode": "spaces break xargs without -0; regex too greedy; rename syntax differs per distro",
        "difficulty": "medium",
    },
    {
        "domain": "machine_learning",
        "skill": "debugging",
        "description": "Investigate NaN loss during training; check learning rate, gradient clipping, mixed precision",
        "tools": ["python", "torch", "numpy", "wandb"],
        "failure_mode": "AMP scaling factor too high; division by zero in loss; exploding gradients",
        "difficulty": "expert",
    },
    {
        "domain": "software_and_system",
        "skill": "debugging",
        "description": "Process listening on port but connection refused; check bind address vs firewall vs SELinux",
        "tools": ["ss", "netstat", "iptables", "sestatus", "curl"],
        "failure_mode": "bound to 127.0.0.1 not 0.0.0.0; SELinux enforcing; iptables DROP vs REJECT",
        "difficulty": "hard",
    },
]


class EvidenceGuidedResearcher:
    """证据引导深度研究代理 - 论文 3.2
    
    论文："Starting from the abstract idea, the agent searches real technical materials ...
    and progressively incorporates the evidence into the task specification,
    grounding it in specific tools, realistic constraints, known failure modes,
    and concrete input/output contracts."
    
    关键效果：3.45× more solver turns, -13.3pt pass rate
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.knowledge_base = _TECHNICAL_KNOWLEDGE_BASE
        self.scenarios = _REAL_SCENARIOS

    def research(self, anchor_domain: str, anchor_skill: str,
                 base_idea: str) -> List[ResearchEvidence]:
        """对一个抽象想法进行深度研究，返回证据列表"""
        evidence_list: List[ResearchEvidence] = []
        
        # 匹配知识库工具
        relevant_tools = [t for t in self.knowledge_base.keys()
                         if self._tool_relevant(t, anchor_skill, anchor_domain)]
        selected_tools = self.rng.sample(relevant_tools, k=min(3, len(relevant_tools)))
        
        for tool in selected_tools:
            kb = self.knowledge_base[tool]
            failure = self.rng.choice(kb["failure_modes"])
            constraint = self.rng.choice(kb["constraints"]) if kb["constraints"] else ""
            evidence_list.append(ResearchEvidence(
                source_type=self.rng.choice(["doc", "issue", "tutorial", "example"]),
                source_url=f"https://example.com/tech/{tool}/{self.rng.randint(1000,9999)}",
                tool_name=tool,
                failure_mode=failure,
                constraint=constraint,
                snippet=f"Common pitfall with {tool}: {failure}",
                relevance_score=self.rng.uniform(0.6, 0.95),
            ))
        
        return evidence_list

    def _tool_relevant(self, tool: str, skill: str, domain: str) -> bool:
        """判断工具是否与 skill/domain 相关"""
        skill_tool_map = {
            "text_processing": ["grep", "sed", "awk", "sort", "uniq", "jq", "tr", "cut"],
            "file_manipulation": ["find", "tar", "rsync", "cp", "mv", "rm", "xargs"],
            "debugging": ["ps", "top", "strace", "lsof", "journalctl", "ss", "netstat", "nvidia-smi"],
            "configuration": ["systemctl", "cron", "docker", "ssh"],
            "networking": ["curl", "ssh", "nc", "dig", "iptables", "openssl"],
            "cryptography": ["openssl", "ssh"],
            "version_control": ["git"],
            "database": ["mysql/psql"],
            "systems": ["docker", "systemctl", "tar", "rsync", "cron"],
            "algorithmic": ["awk", "sort", "uniq"],
        }
        return tool in skill_tool_map.get(skill, [tool]) or self.rng.random() < 0.3

    def ground_blueprint(self, blueprint) -> "TaskBlueprint":
        """将证据注入蓝图，提升真实难度（3.45× turns, -13.3pt pass rate）"""
        # 论文数据：证据精炼后 turns ×3.45
        blueprint.estimated_turns = max(8, int(blueprint.estimated_turns * 3.45))
        # 加入具体失败模式到 hint
        if blueprint.evidence:
            fm = [e.failure_mode for e in blueprint.evidence if e.failure_mode]
            if fm:
                blueprint.internal_hint += f"  Key pitfalls to handle: {'; '.join(fm[:3])}."
        return blueprint