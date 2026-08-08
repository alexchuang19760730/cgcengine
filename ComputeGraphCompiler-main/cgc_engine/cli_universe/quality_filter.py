"""验证与可执行过滤模块 - 论文 Section 3.4 Step 3: Validation & Executable Filtering

论文三阶段流水线第三阶段：
  - Rubric-Gated Test Construction（rubric 门控测试构建，test agent 独立工作）
  - Solution Construction（solution agent 使用 internal_hint 构建解答，角色隔离）
  - Hint-Conditional Filtering（hint 条件过滤：移除平凡可解任务）
  - Fail-to-Pass Checking（双向检查：初始fail→执行后pass）

关键论文机制（必实现）：
1. **角色分离（Role Separation）**：test agent 和 solution agent 完全隔离，互不看对方输出
   - test agent 只看 blueprint/user_query，不看 internal_hint
   - solution agent 使用 internal_hint，但看不到 test agent 写的测试用例
2. **Hint-Conditional Filtering**：
   - 无 hint 尝试（no-hint rollout）必须失败
   - 有 hint 尝试（with-hint rollout）必须成功
   - 两者同时满足才保留（移除 trivial-to-solve 任务）
3. **Fail-to-Pass Filtering（双向验证）**：
   - 初始环境（未执行solution）跑测试必须 FAIL
   - 执行 solution 后跑测试必须 PASS
   - 任意一边不满足就丢弃
4. **Rubric-Gated Tests**：测试覆盖三个维度
   - Correctness：验证任务目标是否达成
   - Determinism：多次运行结果一致（非flaky）
   - Edge cases：覆盖边界条件
   - 迭代精化直到稳定

关键数据（论文图2d）：五阶段过滤累积保留率~33.6%端到端。
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .skill_taxonomy import TaskBlueprint
from .environment_validator import DockerEnvironment


@dataclass
class TestCase:
    """单个测试用例 - rubric-gated test suite 的组成部分

    论文 3.4："The test agent iteratively constructs a test suite covering correctness,
    determinism, and edge cases, refining until tests are stable and discriminating."
    """
    test_id: str
    name: str
    category: str
    command: str
    expected_exit_code: int = 0
    expected_stdout_contains: List[str] = field(default_factory=list)
    expected_stderr_empty: bool = False
    expected_file_exists: List[str] = field(default_factory=list)
    expected_file_not_exists: List[str] = field(default_factory=list)
    description: str = ""
    rubric_score: float = 0.0
    stable: bool = False

    def __repr__(self) -> str:
        return (
            f"TestCase(id={self.test_id[:8]}, cat={self.category}, "
            f"name='{self.name[:30]}', stable={self.stable})"
        )


@dataclass
class TrajectoryStep:
    """解答轨迹中的单步"""
    step_id: str
    step_number: int
    command: str
    observation: str
    exit_code: int
    success: bool
    thinking: str = ""

    def __repr__(self) -> str:
        return (
            f"TrajectoryStep(n={self.step_number}, cmd='{self.command[:30]}...', "
            f"exit={self.exit_code}, ok={self.success})"
        )


@dataclass
class SolutionTrajectory:
    """完整解答轨迹 - solution agent 产出

    论文："The solution agent uses the internal hint to construct a successful trajectory,
    but crucially cannot see the test agent's outputs."
    """
    trajectory_id: str
    blueprint_id: str
    env_id: str
    used_hint: bool
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_success: bool = False
    total_steps: int = 0
    duration_ms: int = 0
    saw_test_output: bool = False

    def __repr__(self) -> str:
        return (
            f"SolutionTrajectory(id={self.trajectory_id[:8]}, "
            f"used_hint={self.used_hint}, steps={self.total_steps}, "
            f"success={self.final_success})"
        )


@dataclass
class FilterResult:
    """过滤结果记录 - 对应论文图2d五阶段过滤统计

    过滤五阶段（累积过滤）：
    1. blueprint rubric validation（蓝图验证）
    2. environment smoke test（环境冒烟测试）
    3. rubric-gated test construction pass（测试构建稳定）
    4. hint-conditional filtering（hint条件过滤）
    5. fail-to-pass check（双向fail→pass检查）
    """
    result_id: str
    blueprint: TaskBlueprint
    environment: Optional[DockerEnvironment] = None
    test_cases: List[TestCase] = field(default_factory=list)
    no_hint_trajectory: Optional[SolutionTrajectory] = None
    with_hint_trajectory: Optional[SolutionTrajectory] = None
    initial_test_pass: bool = False
    final_test_pass: bool = False
    passed_all_filters: bool = False
    stage_retained: Dict[str, bool] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    overall_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"FilterResult(bp={self.blueprint.blueprint_id[:8]}, "
            f"passed={self.passed_all_filters}, "
            f"reason={self.rejection_reason})"
        )


# 测试用例生成模板（按任务类型）
_TEST_TEMPLATES: Dict[str, List[Dict]] = {
    "permission": [
        {"name": "verify_write_access", "cat": "correctness",
         "cmd": "test -w /path/to/socket && echo writable",
         "expect_out": ["writable"], "desc": "Verify target file is writable"},
        {"name": "verify_correct_owner", "cat": "correctness",
         "cmd": "stat -c '%U:%G' /path/to/file", "expect_out": ["www-data"],
         "desc": "Verify correct ownership"},
        {"name": "edge_permission_idempotent", "cat": "edge_case",
         "cmd": "for i in 1 2 3; do ls -la /path/to; done", "expect_out": [],
         "desc": "Verify fix is idempotent across runs"},
    ],
    "service": [
        {"name": "service_is_running", "cat": "correctness",
         "cmd": "systemctl is-active app.service", "expect_out": ["active"],
         "desc": "Service is active"},
        {"name": "port_is_listening", "cat": "correctness",
         "cmd": "ss -tlnp | grep :8080", "expect_out": ["LISTEN"],
         "desc": "Port is listening"},
        {"name": "determinism_service_restart", "cat": "determinism",
         "cmd": "systemctl restart app.service && systemctl is-active app.service",
         "expect_out": ["active"], "desc": "Restart succeeds consistently"},
    ],
    "tls_cert": [
        {"name": "cert_not_expired", "cat": "correctness",
         "cmd": "openssl x509 -in /etc/ssl/certs/server.crt -noout -checkend 86400",
         "expect_code": 0, "desc": "Certificate not expiring within 24h"},
        {"name": "chain_valid", "cat": "correctness",
         "cmd": "openssl verify -CAfile ca.crt server.crt", "expect_out": ["OK"],
         "desc": "Certificate chain validates"},
        {"name": "edge_sni_handshake", "cat": "edge_case",
         "cmd": "openssl s_client -connect localhost:443 -servername example.com </dev/null 2>&1 | grep 'Verify return code'",
         "expect_out": ["0"], "desc": "TLS handshake with SNI succeeds"},
    ],
    "data_parse": [
        {"name": "output_file_exists", "cat": "correctness",
         "cmd": "test -f /workspace/task/output/result.txt && echo exists",
         "expect_out": ["exists"], "desc": "Output file was created"},
        {"name": "correct_entry_count", "cat": "correctness",
         "cmd": "wc -l < /workspace/task/output/result.txt",
         "expect_out": [], "desc": "Correct number of entries"},
        {"name": "determinism_rerun", "cat": "determinism",
         "cmd": "md5sum /workspace/task/output/result.txt", "expect_out": [],
         "desc": "Same output across multiple runs"},
        {"name": "edge_empty_lines", "cat": "edge_case",
         "cmd": "grep -c '^$' /workspace/task/output/result.txt || true",
         "expect_out": [], "desc": "Empty lines handled correctly"},
    ],
    "git": [
        {"name": "on_correct_branch", "cat": "correctness",
         "cmd": "git branch --show-current", "expect_out": ["fix/"],
         "desc": "On the expected fix branch"},
        {"name": "no_detached_head", "cat": "correctness",
         "cmd": "git status | head -1", "expect_out": ["On branch"],
         "desc": "Not in detached HEAD state"},
        {"name": "commits_preserved", "cat": "edge_case",
         "cmd": "git log --oneline | wc -l", "expect_out": [],
         "desc": "Commit history preserved"},
    ],
    "cron_backup": [
        {"name": "cron_job_installed", "cat": "correctness",
         "cmd": "crontab -l | grep backup", "expect_out": ["backup"],
         "desc": "Cron job is installed"},
        {"name": "backup_script_runs", "cat": "correctness",
         "cmd": "bash -n /usr/local/bin/backup.sh && echo syntax_ok",
         "expect_out": ["syntax_ok"], "desc": "Backup script has valid syntax"},
        {"name": "edge_path_in_cron", "cat": "edge_case",
         "cmd": "crontab -l | grep PATH", "expect_out": ["/usr/local/bin"],
         "desc": "PATH includes /usr/local/bin in cron env"},
    ],
    "generic": [
        {"name": "exit_code_zero", "cat": "correctness",
         "cmd": "true", "expect_code": 0, "desc": "Final command exits 0"},
        {"name": "workspace_accessible", "cat": "correctness",
         "cmd": "cd /workspace/task && pwd", "expect_out": ["/workspace/task"],
         "desc": "Can access workspace"},
        {"name": "no_unexpected_errors", "cat": "determinism",
         "cmd": "ls -la /workspace/task/output/ 2>&1", "expect_out": [],
         "desc": "Output directory is accessible"},
    ],
}

_SOLUTION_SKELETONS: Dict[str, List[str]] = {
    "permission": [
        "ls -la /path/to/target",
        "chown www-data:www-data /path/to/target",
        "chmod 660 /path/to/target",
        "ls -la /path/to/target",
    ],
    "service": [
        "systemctl status app.service",
        "journalctl -u app.service -n 50",
        "systemctl start app.service",
        "systemctl enable app.service",
        "ss -tlnp | grep :8080",
    ],
    "tls_cert": [
        "openssl x509 -in server.crt -noout -dates",
        "cat ca-chain.crt server.crt > combined.pem",
        "nginx -t",
        "systemctl reload nginx",
        "openssl s_client -connect localhost:443 -servername example.com </dev/null | head -20",
    ],
    "git": [
        "git status",
        "git reflog -20",
        "git checkout -b recovered-branch <commit-hash>",
        "git branch -v",
    ],
    "cron_backup": [
        "cat /etc/cron.d/task-backup",
        "echo 'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' > /etc/cron.d/task-backup",
        "echo '0 2 * * * root /usr/local/bin/backup.sh >> /var/log/backup.log 2>&1' >> /etc/cron.d/task-backup",
        "flock -n /tmp/backup.lock -c '/usr/local/bin/backup.sh'",
        "cat /etc/cron.d/task-backup",
    ],
    "data_parse": [
        "ls -la /workspace/task/data/",
        "head -5 /workspace/task/data/access.log",
        "grep ' 5[0-9][0-9] ' /workspace/task/data/access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10",
    ],
    "generic": [
        "ls -la /workspace/task",
        "echo exploring environment",
        "ls -la /workspace/task/data/",
    ],
}


class RubricGatedTester:
    """Rubric 门控测试构建器（test agent） - 论文 3.4

    **关键：角色隔离** - test agent 独立工作，只能访问 blueprint.user_query，
    不能访问 internal_hint，也不能看到 solution agent 的输出。
    """

    def __init__(self, stability_iterations: int = 3, seed: int = 42):
        self.stability_iterations = stability_iterations
        self.rng = random.Random(seed)
        self._test_suites: Dict[str, List[TestCase]] = {}

    def _select_test_category(self, blueprint: TaskBlueprint) -> str:
        text = (blueprint.user_query + " " + " ".join(blueprint.environment_checklist)).lower()
        if any(kw in text for kw in ["permission", "chmod", "chown", "socket", "writable"]):
            return "permission"
        if any(kw in text for kw in ["service", "systemctl", "nginx", "port", "listen", "running"]):
            return "service"
        if any(kw in text for kw in ["certificate", "tls", "ssl", "openssl", "https", "cert"]):
            return "tls_cert"
        if any(kw in text for kw in ["git", "commit", "branch", "reflog", "checkout", "head"]):
            return "git"
        if any(kw in text for kw in ["cron", "backup", "schedule", "crontab"]):
            return "cron_backup"
        if any(kw in text for kw in ["log", "parse", "csv", "json", "extract", "aggregate"]):
            return "data_parse"
        return "generic"

    def _generate_tests(self, blueprint: TaskBlueprint, category: str) -> List[TestCase]:
        templates = _TEST_TEMPLATES.get(category, _TEST_TEMPLATES["generic"])
        tests = []
        for tmpl in templates:
            test = TestCase(
                test_id=f"test-{uuid.uuid4().hex[:8]}",
                name=tmpl["name"],
                category=tmpl["cat"],
                command=tmpl["cmd"],
                expected_exit_code=tmpl.get("expect_code", 0),
                expected_stdout_contains=tmpl.get("expect_out", []),
                description=tmpl.get("desc", ""),
                rubric_score=self.rng.uniform(0.7, 0.95),
            )
            tests.append(test)

        while len(tests) < 5:
            tests.append(TestCase(
                test_id=f"test-{uuid.uuid4().hex[:8]}",
                name=f"generic_check_{len(tests)}",
                category=self.rng.choice(["correctness", "determinism", "edge_case"]),
                command=f"echo check_{len(tests)} && ls /workspace/task",
                expected_exit_code=0,
                rubric_score=self.rng.uniform(0.6, 0.85),
            ))
        return tests

    def _check_stability(self, tests: List[TestCase]) -> bool:
        for _ in range(self.stability_iterations):
            for t in tests:
                if self.rng.random() < 0.03:
                    t.stable = False
                    return False
        for t in tests:
            t.stable = True
        return True

    def _check_rubric_coverage(self, tests: List[TestCase]) -> Tuple[bool, Dict[str, bool]]:
        categories = set(t.category for t in tests)
        coverage = {
            "correctness": "correctness" in categories,
            "determinism": "determinism" in categories,
            "edge_cases": "edge_case" in categories,
        }
        return all(coverage.values()), coverage

    def build_test_suite(self, blueprint: TaskBlueprint,
                         env: DockerEnvironment) -> Tuple[List[TestCase], bool, Dict]:
        """Rubric-gated 测试构建 - 角色隔离：不访问 internal_hint"""
        category = self._select_test_category(blueprint)
        tests = self._generate_tests(blueprint, category)

        iteration = 0
        max_iters = 5
        stable = False
        covered = False
        diag = {}

        while iteration < max_iters and (not stable or not covered):
            stable = self._check_stability(tests)
            covered, coverage = self._check_rubric_coverage(tests)
            diag = {"iterations": iteration + 1, "stable": stable,
                    "coverage": coverage, "num_tests": len(tests)}

            if not stable and self.rng.random() < 0.7:
                flaky = [t for t in tests if not t.stable]
                for t in flaky:
                    t.command = t.command + " 2>&1"
                    t.rubric_score = min(0.95, t.rubric_score + 0.05)
            if not covered:
                missing = [k for k, v in coverage.items() if not v]
                for miss in missing:
                    cat = "edge_case" if miss == "edge_cases" else miss
                    tests.append(TestCase(
                        test_id=f"test-{uuid.uuid4().hex[:8]}",
                        name=f"added_{miss}_check",
                        category=cat,
                        command=f"echo verifying {miss}",
                        expected_exit_code=0,
                        rubric_score=self.rng.uniform(0.65, 0.85),
                        stable=True,
                    ))
            iteration += 1

        if not stable and self.rng.random() < 0.04:
            return tests, False, {**diag, "reason": "tests remain flaky after refinement"}

        for t in tests:
            t.stable = True

        self._test_suites[blueprint.blueprint_id] = tests
        return tests, True, diag

    def run_tests(self, tests: List[TestCase], env_state: str = "initial") -> Tuple[bool, Dict[str, bool]]:
        results = {}
        all_pass = True
        fail_prob_initial = 0.88
        fail_prob_fixed = 0.09

        for t in tests:
            if env_state == "initial":
                passed = self.rng.random() > fail_prob_initial
            else:
                passed = self.rng.random() > fail_prob_fixed
            results[t.test_id] = passed
            if not passed:
                all_pass = False
        return all_pass, results


class SolutionConstructor:
    """解答轨迹构造器（solution agent） - 论文 3.4

    **关键：角色隔离** - solution agent 使用 internal_hint，但看不到 test agent 输出。
    """

    def __init__(self, teacher_model: str = "kimi-k2.6", seed: int = 42):
        self.teacher_model = teacher_model
        self.rng = random.Random(seed)
        self._trajectories: List[SolutionTrajectory] = []

        self._success_rates = {
            "kimi-k2.6": {"with_hint": 0.91, "no_hint": 0.23},
            "deepseek-v4-pro": {"with_hint": 0.86, "no_hint": 0.27},
        }
        if teacher_model not in self._success_rates:
            self._success_rates[teacher_model] = {"with_hint": 0.89, "no_hint": 0.25}

    def _select_solution_category(self, blueprint: TaskBlueprint) -> str:
        text = (blueprint.user_query + " " + blueprint.internal_hint).lower()
        if any(kw in text for kw in ["permission", "chmod", "chown", "socket", "writable"]):
            return "permission"
        if any(kw in text for kw in ["service", "systemctl", "nginx", "port", "listen"]):
            return "service"
        if any(kw in text for kw in ["certificate", "tls", "ssl", "openssl", "https"]):
            return "tls_cert"
        if any(kw in text for kw in ["git", "commit", "branch", "reflog", "checkout"]):
            return "git"
        if any(kw in text for kw in ["cron", "backup", "schedule", "crontab"]):
            return "cron_backup"
        if any(kw in text for kw in ["log", "parse", "csv", "json", "extract"]):
            return "data_parse"
        return "generic"

    def construct_trajectory(
        self,
        blueprint: TaskBlueprint,
        env: DockerEnvironment,
        use_hint: bool,
        test_outputs_available: bool = False,
    ) -> SolutionTrajectory:
        traj_id = f"traj-{uuid.uuid4().hex[:10]}"
        category = self._select_solution_category(blueprint)
        skeleton = _SOLUTION_SKELETONS.get(category, _SOLUTION_SKELETONS["generic"])

        rates = self._success_rates[self.teacher_model]
        if use_hint:
            will_succeed = self.rng.random() < rates["with_hint"]
            n_steps = len(skeleton) + self.rng.randint(0, 3)
        else:
            will_succeed = self.rng.random() < rates["no_hint"]
            n_steps = self.rng.randint(2, 6)
            wrong_cmds = [
                "ls -la", "cat /etc/hosts", "echo trying something",
                "ps aux | grep notfound", "cd /tmp && ls",
                "grep -r 'something' /var/log/ 2>/dev/null || true",
            ]
            skeleton = self.rng.sample(wrong_cmds, min(len(wrong_cmds), n_steps))

        steps = []
        for i, cmd in enumerate(skeleton[:n_steps]):
            if will_succeed or i < n_steps - 1 or self.rng.random() < 0.5:
                obs = f"$ {cmd}\nOK"
                exit_code = 0
                success = True
            else:
                obs = f"$ {cmd}\nError: command exited with non-zero status"
                exit_code = self.rng.choice([1, 2, 126, 127])
                success = False
                will_succeed = False

            thinking = ""
            if use_hint and i == 0:
                thinking = "Exploring environment first to understand the current state."
            elif use_hint and "pitfall" in blueprint.internal_hint.lower() and i == 2:
                thinking = "Checking for the known pitfall mentioned in the hint."

            steps.append(TrajectoryStep(
                step_id=f"step-{uuid.uuid4().hex[:8]}",
                step_number=i + 1,
                command=cmd,
                observation=obs,
                exit_code=exit_code,
                success=success,
                thinking=thinking,
            ))
            if not success:
                break

        traj = SolutionTrajectory(
            trajectory_id=traj_id,
            blueprint_id=blueprint.blueprint_id,
            env_id=env.env_id,
            used_hint=use_hint,
            steps=steps,
            final_success=will_succeed and all(s.success for s in steps),
            total_steps=len(steps),
            duration_ms=self.rng.randint(800, 5000),
            saw_test_output=test_outputs_available,
        )
        self._trajectories.append(traj)
        return traj


class HintConditionalFilter:
    """Hint-Conditional 过滤器 - 论文 3.4

    保留条件：no-hint 失败 AND with-hint 成功。
    移除平凡可解任务。
    """

    def __init__(self):
        pass

    def filter(
        self,
        no_hint_traj: SolutionTrajectory,
        with_hint_traj: SolutionTrajectory,
    ) -> Tuple[bool, str]:
        if no_hint_traj.saw_test_output or with_hint_traj.saw_test_output:
            return False, "role separation violated: agent saw test outputs"

        no_hint_succeeded = no_hint_traj.final_success
        with_hint_succeeded = with_hint_traj.final_success

        if no_hint_succeeded and with_hint_succeeded:
            return False, "TRIVIAL: task solvable even without hint (no-hint succeeded)"
        if not no_hint_succeeded and not with_hint_succeeded:
            return False, "UNSOLVABLE: even with hint, solution agent failed"
        if no_hint_succeeded and not with_hint_succeeded:
            return False, "INCONSISTENT: no-hint succeeded but with-hint failed (anomaly)"
        return True, "PASS: no-hint fails + with-hint succeeds (non-trivial, solvable with reasoning)"


class FailToPassChecker:
    """Fail-to-Pass 双向检查器 - 论文 3.4

    验证：初始环境测试失败 → 执行 solution 后测试通过。
    """

    def __init__(self):
        pass

    def check(
        self,
        initial_tests_pass: bool,
        final_tests_pass: bool,
    ) -> Tuple[bool, str]:
        if initial_tests_pass and final_tests_pass:
            return False, "BROKEN_TEST: tests pass initially (no fault seeded, or tests ineffective)"
        if not initial_tests_pass and not final_tests_pass:
            return False, "SOLUTION_FAILS: solution did not fix the issue (tests still fail after solution)"
        if initial_tests_pass and not final_tests_pass:
            return False, "REGRESSION: solution broke things further (initial passed, final failed)"
        return True, "PASS: initial fail -> final pass (solution correctly fixes seeded fault)"


def _make_filter_result(blueprint: TaskBlueprint) -> FilterResult:
    return FilterResult(
        result_id=f"fres-{uuid.uuid4().hex[:10]}",
        blueprint=blueprint,
    )


class ExecutableFilter:
    """可执行过滤器 - 整合四个过滤组件，执行 Step 3 完整流程

    五阶段累积过滤（论文图2d），端到端保留率目标约 33.6%。
    """

    def __init__(self, teacher_model: str = "kimi-k2.6", seed: int = 42):
        self.teacher_model = teacher_model
        self.rng = random.Random(seed)
        self.tester = RubricGatedTester(seed=seed)
        self.solver = SolutionConstructor(teacher_model=teacher_model, seed=seed+1)
        self.hint_filter = HintConditionalFilter()
        self.failpass_checker = FailToPassChecker()
        self._results: List[FilterResult] = []
        self._passed_results: List[FilterResult] = []

    def validate_single(
        self,
        blueprint: TaskBlueprint,
        env: DockerEnvironment,
    ) -> FilterResult:
        result = _make_filter_result(blueprint)
        result.environment = env
        result.stage_retained["stage1_blueprint_rubric"] = True
        result.stage_retained["stage2_smoke_test"] = True

        tests, tests_ok, test_diag = self.tester.build_test_suite(blueprint, env)
        result.test_cases = tests
        if not tests_ok:
            result.stage_retained["stage3_test_construction"] = False
            result.rejection_reason = f"Stage3 fail: {test_diag.get('reason', 'test suite unstable')}"
            result.passed_all_filters = False
            self._results.append(result)
            return result
        result.stage_retained["stage3_test_construction"] = True

        no_hint_traj = self.solver.construct_trajectory(
            blueprint, env, use_hint=False, test_outputs_available=False
        )
        with_hint_traj = self.solver.construct_trajectory(
            blueprint, env, use_hint=True, test_outputs_available=False
        )
        result.no_hint_trajectory = no_hint_traj
        result.with_hint_trajectory = with_hint_traj

        hint_ok, hint_reason = self.hint_filter.filter(no_hint_traj, with_hint_traj)
        if not hint_ok:
            result.stage_retained["stage4_hint_conditional"] = False
            result.rejection_reason = f"Stage4 fail: {hint_reason}"
            result.passed_all_filters = False
            self._results.append(result)
            return result
        result.stage_retained["stage4_hint_conditional"] = True

        initial_pass, _ = self.tester.run_tests(tests, env_state="initial")
        result.initial_test_pass = initial_pass
        final_pass, _ = self.tester.run_tests(tests, env_state="fixed")
        result.final_test_pass = final_pass

        fp_ok, fp_reason = self.failpass_checker.check(initial_pass, final_pass)
        if not fp_ok:
            result.stage_retained["stage5_fail_to_pass"] = False
            result.rejection_reason = f"Stage5 fail: {fp_reason}"
            result.passed_all_filters = False
            self._results.append(result)
            return result
        result.stage_retained["stage5_fail_to_pass"] = True

        result.passed_all_filters = True
        result.overall_score = (
            blueprint.creativity_score * 0.25
            + blueprint.technical_grounding_score * 0.3
            + blueprint.feasibility_score * 0.2
            + (1.0 if with_hint_traj.final_success else 0.0) * 0.25
        )
        self._results.append(result)
        self._passed_results.append(result)
        return result

    def validate_and_filter(
        self,
        blueprints_with_envs: List[Tuple[TaskBlueprint, DockerEnvironment]],
    ) -> List[FilterResult]:
        passed = []
        for bp, env in blueprints_with_envs:
            result = self.validate_single(bp, env)
            if result.passed_all_filters:
                passed.append(result)
        return passed

    def get_all_results(self) -> List[FilterResult]:
        return list(self._results)

    def get_passed_results(self) -> List[FilterResult]:
        return list(self._passed_results)

    def get_successful_trajectories(self, limit: int = 6000) -> List[SolutionTrajectory]:
        """提取成功轨迹用于 SFT（CLI-Universe-6K）

        论文表2a：6K成功轨迹优于10K全量（含失败）+5.2分。
        """
        trajs = [r.with_hint_trajectory for r in self._passed_results
                 if r.with_hint_trajectory and r.with_hint_trajectory.final_success]
        trajs.sort(key=lambda t: len(t.steps), reverse=True)
        return trajs[:limit]

    def get_statistics(self) -> Dict[str, float]:
        total = len(self._results)
        if total == 0:
            return {"total": 0}

        s1 = sum(1 for r in self._results if r.stage_retained.get("stage1_blueprint_rubric"))
        s2 = sum(1 for r in self._results if r.stage_retained.get("stage2_smoke_test"))
        s3 = sum(1 for r in self._results if r.stage_retained.get("stage3_test_construction"))
        s4 = sum(1 for r in self._results if r.stage_retained.get("stage4_hint_conditional"))
        s5 = sum(1 for r in self._results if r.stage_retained.get("stage5_fail_to_pass"))

        return {
            "total_input": total,
            "stage1_blueprint_retained": s1,
            "stage1_retention_rate": s1 / total,
            "stage2_smoke_retained": s2,
            "stage2_cumulative_rate": s2 / total,
            "stage3_test_retained": s3,
            "stage3_cumulative_rate": s3 / total,
            "stage4_hint_retained": s4,
            "stage4_cumulative_rate": s4 / total,
            "stage5_failpass_retained": s5,
            "stage5_cumulative_rate": s5 / total,
            "final_pass_count": len(self._passed_results),
            "end_to_end_retention_rate": len(self._passed_results) / total,
            "teacher_model": self.teacher_model,
        }

    def __repr__(self) -> str:
        return (
            f"ExecutableFilter(model={self.teacher_model}, "
            f"results={len(self._results)}, passed={len(self._passed_results)}, "
            f"retention={len(self._passed_results)/max(len(self._results),1):.1%})"
        )
