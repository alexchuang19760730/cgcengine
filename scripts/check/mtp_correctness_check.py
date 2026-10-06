#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtp_correctness_check.py — MTP-on 任务正确性硬证据生成器（lineA）

目的（对应「放弃 bit-identical，但任务结果必须正确」）：
  MTP-on（投机解码）在数学上保持基模输出分布，前提是 verify 步骤是 *sound* 的。
  本脚本用两道机检证明：

    L1 soundness（自动判）：temp=0 + 同 seed 下，MTP-on 输出必须与 MTP-off
      逐字节相同。通过 ⇒ verify 没有接受错误 token（即无 4 秒悬崖类 bug）。
      这是「结果正确」的最强硬证据：贪心下 spec-decode 理论保证与基模等价。

    L2 task（人工/调用方判）：temp>0（真实 25+ 配置）下，对任务型 prompt
      的回答必须正确。本脚本只负责产出可比对文本，正确性由后续评测判定。

机制（已对齐 run_server.sh 的真实变量名）：
  run_server.sh 有**两个独立旋钮**：
    * CGC_SERVER_PROFILE        模型/配置档（如 prod25 = 唯一实测 25.17 t/s 的配置）
    * CGC_SERVER_MTP            显式 0/1，控制 MTP on/off（authoritative）
  所以本脚本：
    - 对每种 mode 设 CGC_SERVER_PROFILE=<profile>（两臂相同，保证 CTX/BUDGET 一致）
    - 用显式 CGC_SERVER_MTP 切 MTP：off→0，on→1（profile 的 [ -z ] 守卫会跳过，不覆盖）
      （★ 不能用 CGC_SERVER_RUNTIME_PROFILE：prod25 等 profile 的
       `[ -z "${CGC_SERVER_MTP+x}" ] && SERVER_MTP=1` 在 runtime profile case 之后执行，
       会强制翻成 1，导致 off 模式也被迫 MTP ON → L1 假阳性 PASS）
    - 起 server → 轮询 /health → 对每条 prompt POST /v1/chat/completions
      （temperature=0, seed=0, stream=false）
    - 对 run_server.sh 发 SIGINT（其内部 trap 优雅 kill llama-server）收尾
    - 额外断言：on 模式 server 日志必须含 "MTP ON"、off 模式必须「不含」"MTP ON"
      → 防止 env 没生效导致的静默假阳性（这正是上一版 prod25 误用的同类坑）

  最后比对 off vs on 文本：L1 要求逐字节相等（另附 strip 后的次级信号）。

清理协议（来自 run_server.sh:2908/2965）：
  server 由 run_server.sh 后台拉起、脚本自身 wait 并 trap SIGINT→kill -INT server。
  ⇒ 我们对 run_server.sh 进程发 SIGINT 即可优雅收尾，切勿直接 kill -9 llama-server。

用法：
  # L1-a（默认）：prod25 profile，MTP-off 与 MTP-on 各跑一遍，temp=0 比对
  python3 scripts/check/mtp_correctness_check.py --port 8081

  # 叠加真实 25+ 的 CLI-parity 路径（early-verify init），再验 L1 是否仍 sound
  python3 scripts/check/mtp_correctness_check.py --port 8081 \
      --extra-env CGC_SERVER_MTP_CLI_PARITY=1

  # 用自定义 prompt 文件（每行一条）
  python3 scripts/check/mtp_correctness_check.py --prompts prompts.txt --port 8081

  # L2：关掉强制 temp0，用采样温度跑（只产文本，正确性人工判）
  python3 scripts/check/mtp_correctness_check.py --temperature 0.3 --no-force-temp0 --port 8081

  # 只打印将要执行的命令/env，不真正起 server（review 用）
  python3 scripts/check/mtp_correctness_check.py --dry-run --port 8081

注意：
  - verify-decode（CGC_VERIFY_DECODE）在 mtp 模式下默认就是开着的（run_server.sh:2664
    默认 SERVER_VERIFY_DECODE=1），所以 L1-a 已站在真实 25+ 路径上；L1-b 额外叠
    CGC_SERVER_MTP_CLI_PARITY=1 是更贴 25+ 的 init 口径。
  - 本地若有代理（如 7897）会拦 127.0.0.1 造成 502；脚本已设 no_proxy=* 绕过。
  - run_server.sh 默认不再替你清场（preflight 不杀别人的 server），但会做
    memory guard 检查；若内存不足起不来，health 超时后脚本会打印 preflight 提示。
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SERVER_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_server.sh")

# 内置任务型 prompt（覆盖数学/事实/逻辑/代码/指令跟随，便于 L2 人工判读）
DEFAULT_PROMPTS = [
    "请计算：17 * 23 - 8 = ? 只需给出最终数字。",
    "法国的首都是哪座城市？一句话回答。",
    "如果所有的猫都是动物，而米娜是一只猫，那么米娜是动物吗？请回答是或否并简述理由。",
    "用 Python 写一个函数 is_prime(n)，返回 n 是否为素数。",
    "请把这句话翻译成英文：今天天气真好，我们去公园散步吧。",
]

HEALTH_TIMEOUT = 420.0      # 模型加载 + 就绪的耐心（秒）
HEALTH_POLL = 2.0
REQUEST_TIMEOUT = 240.0     # 单条 completion 的耐心（秒）
STOP_WAIT = 30.0           # SIGINT 后等 run_server.sh 优雅退出的耐心（秒）

# MTP 开关走显式 CGC_SERVER_MTP（0/1）。
# 注意：不能走 CGC_SERVER_RUNTIME_PROFILE，因为 run_server.sh 的 profile case
# （如 prod25 的 `[ -z "${CGC_SERVER_MTP+x}" ] && SERVER_MTP=1`）在 runtime profile
# case 之后执行，且只看 CGC_SERVER_MTP 有没有设——不设就会被强制翻成 1，
# 导致 off/on 两路都跑 MTP-on（L1 假阳性 PASS）。显式设 CGC_SERVER_MTP 后，
# 两个 case 的 `[ -z ... ]` 守卫都会跳过，SERVER_MTP 保住我们设的值。


def log(msg: str) -> None:
    print(f"[mtp_correctness_check] {msg}", flush=True)


def port_in_use(port: int) -> bool:
    """碰撞面预检：端口是否已被监听（别人的 server 在跑就别抢）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def build_env(mode: str, profile: str, port: int,
              force_temp0: bool, extra_env: list[str]) -> dict:
    env = dict(os.environ)
    # 两臂 CGC_SERVER_PROFILE 相同 → CTX/BUDGET/draft_n_max 完全一致，仅 MTP 不同
    env["CGC_SERVER_PROFILE"] = profile
    # MTP 开关：显式 CGC_SERVER_MTP（off→0 / on→1）。
    # 用显式值而非 runtime profile：profile 的 `[ -z "${CGC_SERVER_MTP+x}" ]` 守卫
    # 看到已设就会跳过，SERVER_MTP 保住我们设的值（不会被 prod25 翻成 1）。
    env["CGC_SERVER_MTP"] = "1" if mode == "on" else "0"
    # denseIQ4X 是 MTP 的 carrier（run_server.sh:89 默认 1），显式设保证两臂一致
    env["CGC_SERVER_DENSE_IQ4X"] = "1"
    # L1：强制 temp=0 + seed=0（run_server.sh 侧 C++ 检测 presence，非值）
    if force_temp0:
        env["CGC_FORCE_TEMP0"] = "1"
    else:
        env.pop("CGC_FORCE_TEMP0", None)
    env["CGC_SERVER_PORT"] = str(port)
    # 绕过本地代理对 127.0.0.1 的拦截（run_server.sh:9 注记的 502 陷阱）
    env["no_proxy"] = "*"
    env["NO_PROXY"] = "*"
    for kv in extra_env:
        if "=" not in kv:
            raise SystemExit(f"--extra-env 必须是 KEY=VAL，收到：{kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v
    return env


def start_server(script: str, env: dict, log_path: str) -> subprocess.Popen:
    # start_new_session 隔离 process group，SIGINT 不会级联到别的东西
    lf = open(log_path, "wb")
    return subprocess.Popen(
        ["bash", script],
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ), lf


def wait_ready(port: int, timeout: float = HEALTH_TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL)
    return False


def assert_mtp_state(mode: str, log_path: str) -> None:
    """防假阳性：确认 env 真的让 MTP 按预期开关。

    on 模式：日志必须含 "MTP ON"；off 模式：必须「不含」"MTP ON"。
    若不符，说明 env 没生效（例如 runtime profile 被某默认分支覆盖），
    此时 L1 即便"PASS"也毫无意义，必须报错。
    """
    try:
        txt = open(log_path, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        raise SystemExit(f"mode={mode}: 读 server 日志失败（{e}），无法断言 MTP 状态")
    has_mtp_on = "MTP ON" in txt
    if mode == "on" and not has_mtp_on:
        raise SystemExit(
            f"mode=on 但 server 日志未出现 'MTP ON'——MTP 实际未开启，"
            f"L1 失去对照意义。日志尾段：\n{txt[-1500:]}"
        )
    if mode == "off" and has_mtp_on:
        raise SystemExit(
            f"mode=off 但 server 日志出现 'MTP ON'——runtime profile 未关掉 MTP，"
            f"off/on 两臂实际都在跑 MTP-on，L1 会产生假阳性 PASS。日志尾段：\n{txt[-1500:]}"
        )
    log(f"mode={mode}: MTP 状态断言 OK（{'ON' if has_mtp_on else 'OFF'}）")


def stop_server(proc: subprocess.Popen, lf) -> None:
    try:
        proc.send_signal(signal.SIGINT)  # run_server.sh trap 会优雅 kill llama-server
    except Exception as e:
        log(f"发 SIGINT 失败：{e}")
    try:
        proc.wait(timeout=STOP_WAIT)
    except subprocess.TimeoutExpired:
        log("优雅退出超时，强制 kill")
        proc.kill()
    try:
        lf.close()
    except Exception:
        pass


def complete(port: int, prompt: str, max_tokens: int,
             temperature: float, seed: int) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        obj = json.loads(r.read().decode("utf-8"))
    return obj["choices"][0]["message"]["content"]


def run_mode(mode: str, args, prompts: list[str]) -> dict:
    env = build_env(mode, args.profile, args.port,
                    args.force_temp0, args.extra_env)
    log_path = os.path.join(
        os.path.dirname(args.out),
        f"mtp_correctness_{mode}.log",
    )
    if args.dry_run:
        log(f"[dry-run] mode={mode} 将起 server，env 关键项：")
        for k in ("CGC_SERVER_PROFILE", "CGC_SERVER_MTP",
                  "CGC_SERVER_DENSE_IQ4X", "CGC_FORCE_TEMP0", "CGC_SERVER_PORT",
                  "CGC_SERVER_MTP_CLI_PARITY", "CGC_VERIFY_DECODE"):
            if k in env:
                log(f"    {k}={env[k]}")
        log(f"[dry-run] 将对 {len(prompts)} 条 prompt 发 /v1/chat/completions "
            f"(temperature={0.0 if args.force_temp0 else args.temperature}, "
            f"max_tokens={args.max_tokens})")
        log(f"[dry-run] server 日志将落盘：{log_path}")
        return {}
    if port_in_use(args.port):
        raise SystemExit(
            f"端口 {args.port} 已被占用（别人的 server？）。换 --port 或等它退出。"
        )
    log(f"mode={mode}: 启动 server (profile={args.profile}, "
        f"mtp={env['CGC_SERVER_MTP']}) ...")
    proc, lf = start_server(args.server_script, env, log_path)
    try:
        if not wait_ready(args.port):
            hint = ""
            try:
                hint = open(log_path, "r", encoding="utf-8", errors="replace").read()[-1500:]
            except Exception:
                pass
            raise SystemExit(
                f"mode={mode}: server 在 {HEALTH_TIMEOUT:.0f}s 内未就绪。\n"
                f"可能原因：memory guard 拒起 / 模型路径错 / 编译期 MTP 未开。\n"
                f"server 日志尾段：\n{hint}"
            )
        log(f"mode={mode}: server 就绪，开始 {len(prompts)} 条 completion ...")
        assert_mtp_state(mode, log_path)
        results = {}
        for i, p in enumerate(prompts):
            try:
                txt = complete(args.port, p, args.max_tokens,
                               0.0 if args.force_temp0 else args.temperature,
                               args.seed)
                results[i] = {"prompt": p, "output": txt}
                log(f"  [{i}] OK ({len(txt)} chars)")
            except Exception as e:
                results[i] = {"prompt": p, "output": None, "error": str(e)}
                log(f"  [{i}] 请求失败：{e}")
        return results
    finally:
        stop_server(proc, lf)


def compare(off: dict, on: dict) -> list[dict]:
    rows = []
    keys = sorted(set(off) | set(on))
    for i in keys:
        o = off.get(i, {})
        n = on.get(i, {})
        o_txt = o.get("output")
        n_txt = n.get("output")
        if o_txt is None or n_txt is None:
            rows.append({"idx": i, "exact": False, "stripped": False,
                         "error": True,
                         "off": o_txt, "on": n_txt})
        else:
            exact = (o_txt == n_txt)
            stripped = (o_txt.strip() == n_txt.strip())
            rows.append({"idx": i, "exact": exact, "stripped": stripped,
                         "error": False,
                         "off": o_txt, "on": n_txt})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="MTP-on 任务正确性硬证据生成器")
    ap.add_argument("--server-script", default=DEFAULT_SERVER_SCRIPT,
                    help="run_server.sh 路径")
    ap.add_argument("--profile", default="prod25",
                    help="CGC_SERVER_PROFILE（默认 prod25 = 唯一实测 25.17 t/s 的配置；"
                         "off/on 两臂共用，仅 MTP 不同）")
    ap.add_argument("--modes", default="off,on",
                    help="逗号分隔，跑哪些 mode（默认 off,on）")
    ap.add_argument("--port", type=int, default=8081,
                    help="server 端口（默认 8081，避免与别人的 8080 撞）")
    ap.add_argument("--prompts", default=None,
                    help="prompt 文件，每行一条；省略则用内置 5 条")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="每条 completion 的最大生成长度")
    ap.add_argument("--seed", type=int, default=0,
                    help="采样种子（L1 下被 CGC_FORCE_TEMP0 覆盖为 0）")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度（L1 默认 0；L2 用 >0 配合 --no-force-temp0）")
    ap.add_argument("--force-temp0", dest="force_temp0", action="store_true",
                    default=True, help="强制 temp=0+seed=0（L1 默认开）")
    ap.add_argument("--no-force-temp0", dest="force_temp0",
                    action="store_false",
                    help="不强制 temp0，用 --temperature 的值（L2 用）")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="叠加 env，可重复，如 --extra-env CGC_SERVER_MTP_CLI_PARITY=1")
    ap.add_argument("--out", default="/tmp/mtp_correctness.json",
                    help="结果 JSON 输出路径")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将执行的命令/env，不起 server")
    args = ap.parse_args()

    if not os.path.exists(args.server_script):
        raise SystemExit(f"找不到 server 脚本：{args.server_script}")

    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        if not prompts:
            raise SystemExit(f"prompt 文件为空：{args.prompts}")
    else:
        prompts = DEFAULT_PROMPTS

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    log(f"modes={modes} profile={args.profile} prompts={len(prompts)} "
        f"force_temp0={args.force_temp0} temperature={args.temperature}")

    collected = {}
    for mode in modes:
        collected[mode] = run_mode(mode, args, prompts)

    if args.dry_run:
        log("[dry-run] 未起 server，结束。")
        return 0

    # 仅当 off 与 on 都有数据时做 L1 比对
    if "off" in collected and "on" in collected and collected["off"] and collected["on"]:
        rows = compare(collected["off"], collected["on"])
        n_exact = sum(1 for r in rows if r["exact"])
        n_err = sum(1 for r in rows if r["error"])
        log("=" * 60)
        log(f"L1 soundness：{n_exact}/{len(rows)} 条逐字节相同"
            + (f"，{n_err} 条请求失败" if n_err else ""))
        for r in rows:
            if r["error"]:
                status = "ERR"
            elif r["exact"]:
                status = "EXACT"
            else:
                status = "DIFF"
            log(f"  [{r['idx']}] {status}")
            if not r["exact"]:
                log(f"      off={r['off']!r}")
                log(f"      on ={r['on']!r}")
        l1_pass = (n_exact == len(rows)) and (n_err == 0)
        log(f"L1 结论：{'PASS（verify sound，MTP-on 结果正确）' if l1_pass else 'FAIL（存在不一致，verify 可能有 bug）'}")
    else:
        l1_pass = None
        log("未同时收集到 off 与 on 数据，跳过 L1 比对。")

    report = {
        "profile": args.profile,
        "modes": modes,
        "force_temp0": args.force_temp0,
        "temperature": args.temperature,
        "l1_pass": l1_pass,
        "prompts": prompts,
        "results": collected,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告已写：{args.out}")
    log(f"server 日志：{os.path.dirname(args.out)}/mtp_correctness_<mode>.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
