#!/usr/bin/env python3
"""
replay_server_profile.py — 應用層 replay benchmark + 質量/速度/內存指標

設計 (2026-09-05):
  - 三個 profile (qa-zh / longform-zh / coding) 一次跑完 (--all-profiles)
  - 每個 profile 收集三個面向:
      1. 質量 (quality): 啟發式 keyword / length / syntax 檢查 (0.0-1.0 score)
      2. 速度 (speed):  prefill t/s + decode t/s (從 llama-server timings)
      3. 內存 (memory): llama-server RSS (psutil, 沒裝 fallback 到 ps -o rss= -p PID)
  - 輸出結構化 JSON 到 --bench-output,供 precommit hook 比較

跟舊版差異:
  - 加 --all-profiles / --bench-output / --reference / --server-pid
  - 加 psutil RSS 監控 (before / peak / after)
  - 加啟發式 quality score (從 --reference 讀 rules)
  - 加 aggregated metrics 摘要 (3 profile 平均 / 中位數)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from urllib import request as urlrequest


# ChatML 模板會 render <im_start>/<im_end> (無 | ) 這兩個 literal marker;
# 不加進 stop 的話, model 重開 assistant turn (echo <im_end><im_start>assistant)
# 時 server 不會停, 造成整段 marker 迴圈輸出。
# Nail GGUF 的 tokenizer 把這兩個字串 tokenize 成多個 sub-token (非 special token),
# 所以 stop 要用純文字 '<im_end>' / '<im_start>' (無 pipe) 才會 match。
CHATML_STOPS = ["<im_end>", "<im_start>"]
LONGFORM_STOP = ["<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_json(url, payload, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url, timeout):
    """HTTP GET, return parsed JSON. Used for /cgc_stats, /health, /metrics."""
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_cgc_stats(base_url, timeout=10):
    """[CGC Three-Factor Measurement 2026-09-07] Get expert cache stats from /cgc_stats endpoint.

    Directly measures the three factors that cause quality degradation:
      Factor 1 (count-cold): fraction of selected experts that are non-resident
      Factor 2 (ZERO-slot pollution): fraction of steps that use the ZERO slot
      Factor 3 (memory pressure): resident memory usage vs pool capacity
    Plus auxiliary metrics: cache hit rate, prefetch stats, load latency, EMA feeds, draft prefetch.

    Returns None if the endpoint is not available (e.g. expert cache not enabled).
    """
    try:
        url = base_url.rstrip("/") + "/cgc_stats"
        return http_get_json(url, timeout)
    except Exception as e:
        return {"error": str(e), "available": False}


def compute_cgc_stats_delta(before, after):
    """[CGC Three-Factor Measurement] Compute the delta (this request's contribution) between two stats snapshots.

    For cumulative counters (n_fast_calls, n_fast_cold, etc.), compute the difference.
    For rates (cold_rate_all_pct, etc.), use the 'after' value directly (they're already rates).
    For memory (resident_mib), use the 'after' value directly.
    """
    if before is None or after is None:
        return {"available": False}
    if "error" in before or "error" in after:
        return {"available": False, "error": before.get("error") or after.get("error")}

    delta = {"available": True}

    # Factor 1: count-cold (rates are already percentages, use 'after' value)
    if "factor1_count_cold" in after:
        f1_after = after["factor1_count_cold"]
        delta["factor1_count_cold"] = {
            "cold_rate_all_pct": f1_after.get("cold_rate_all_pct"),
            "cold_rate_verify_pct": f1_after.get("cold_rate_verify_pct"),
            "cold_rate_draft_pct": f1_after.get("cold_rate_draft_pct"),
        }
        # Compute delta for cumulative counters
        if "factor1_count_cold" in before:
            f1_before = before["factor1_count_cold"]
            for key in ["n_fast_calls", "n_fast_union", "n_fast_cold",
                        "n_fast_draft_calls", "n_fast_draft_union", "n_fast_draft_cold"]:
                if key in f1_after and key in f1_before:
                    delta["factor1_count_cold"][key + "_delta"] = f1_after[key] - f1_before[key]

    # Factor 2: ZERO-slot pollution
    if "factor2_zero_slot" in after:
        f2_after = after["factor2_zero_slot"]
        delta["factor2_zero_slot"] = {
            "zero_slot_enabled": f2_after.get("zero_slot_enabled"),
            "zero_slot_usage_rate_pct": f2_after.get("zero_slot_usage_rate_pct"),
        }

    # Factor 3: memory pressure
    if "factor3_memory_pressure" in after:
        f3_after = after["factor3_memory_pressure"]
        delta["factor3_memory_pressure"] = {
            "resident_mib": f3_after.get("resident_mib"),
            "n_layers": f3_after.get("n_layers"),
            "n_expert_per_layer": f3_after.get("n_expert_per_layer"),
            "total_slots": f3_after.get("total_slots"),
            "slot_utilization_pct": f3_after.get("slot_utilization_pct"),
            "pool_active": f3_after.get("pool_active"),
        }

    # Auxiliary metrics
    if "auxiliary" in after:
        aux_after = after["auxiliary"]
        delta["auxiliary"] = {
            "cache_hit_rate_pct": aux_after.get("cache_hit_rate_pct"),
            "prefetch_drop_rate_pct": aux_after.get("prefetch_drop_rate_pct"),
            "spac_enabled": aux_after.get("spac_enabled"),
            "draft_prefetch_enabled": aux_after.get("draft_prefetch_enabled"),
            "draft_prefetch_hit_rate_pct": aux_after.get("draft_prefetch_hit_rate_pct"),
        }
        # Compute delta for cumulative counters
        if "auxiliary" in before:
            aux_before = before["auxiliary"]
            for key in ["n_requests", "n_hits", "n_misses", "n_prefetch_queued",
                        "n_prefetch_dropped", "n_reads", "spac_feeds",
                        "draft_prefetch_queued", "draft_prefetch_hit", "draft_prefetch_miss"]:
                if key in aux_after and key in aux_before:
                    delta["auxiliary"][key + "_delta"] = aux_after[key] - aux_before[key]

    return delta


def parse_prefetch_drop_breakdown_from_log(log_file_path):
    """[CGC Prefetch Drop Audit 2026-09-08] Parse the prefetch drop breakdown from server log.

    The expert cache destructor outputs a line like:
      llama_expert_cache: prefetch drop breakdown (total=N):
        #1 no_free_slot=X  #2 dbuf_cap_skip=X  #3 drain_cleared=X
        #4 zero_slot_fallback=X  #5 lru_evicted_predicted=X
        #6 bg_reassign_race=X  #7 guard_reject=X
        #8 dbuf2_scratch_invisible=X  #9 one_shot_consumed=X
        #10 fast_wait_expired=X  #11 trigger_too_late=X  #12 collect_skipped=X

    This function parses that line and returns a dict with the 12 drop point counters.
    Returns None if the log file doesn't exist or the breakdown line isn't found.
    """
    import re
    import os

    if not log_file_path or not os.path.exists(log_file_path):
        return None

    try:
        with open(log_file_path, 'r', errors='replace') as f:
            content = f.read()

        # Find the drop breakdown line
        pattern = r'llama_expert_cache: prefetch drop breakdown \(total=(\d+)\):\s*(.*)'
        match = re.search(pattern, content)
        if not match:
            return None

        total = int(match.group(1))
        rest = match.group(2)

        # Parse each #N name=value pair
        result = {"total": total, "available": True}
        drop_points = [
            ("#1", "no_free_slot"),
            ("#2", "dbuf_cap_skip"),
            ("#3", "drain_cleared"),
            ("#4", "zero_slot_fallback"),
            ("#5", "lru_evicted_predicted"),
            ("#6", "bg_reassign_race"),
            ("#7", "guard_reject"),
            ("#8", "dbuf2_scratch_invisible"),
            ("#9", "one_shot_consumed"),
            ("#10", "fast_wait_expired"),
            ("#11", "trigger_too_late"),
            ("#12", "collect_skipped"),
        ]

        for num, name in drop_points:
            # Match "name=value" (allow optional #N prefix)
            pattern = rf'(?:{num}\s+)?{name}=(\d+)'
            m = re.search(pattern, rest)
            if m:
                result[name] = int(m.group(1))
            else:
                result[name] = 0

        # Compute percentages
        if total > 0:
            for num, name in drop_points:
                result[name + "_pct"] = round(100.0 * result[name] / total, 2)
        else:
            for num, name in drop_points:
                result[name + "_pct"] = 0.0

        return result

    except Exception as e:
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-prompt profile definitions (mirror reference v2: 3 prompts per profile).
# Overfit guard: --runs 3 會依序輪詢 3 個 prompts, 驗證 config 不是只對單一 prompt 有效。
# ---------------------------------------------------------------------------
PROFILE_PROMPTS = {
    "qa-zh": [
        "巴黎是哪個國家的首都？請只用一句中文回答。",
        "太陽系有幾個行星？請列出名稱。",
        "請問 15 + 27 等於多少？",
    ],
    "longform-zh": [
        "為什麼巴黎會成為法國的政治與文化中心？請用一段中文說明。",
        "請說明人工智慧的發展歷史、現狀和未來趨勢。",
        "請介紹台灣的地理、文化和經濟特色。",
    ],
    "coding": [
        "用 Python 寫一個函數計算斐波那契數列的第 n 項。",
        "用 Python 寫一個冒泡排序函數。",
        "用 Python 寫一個函數判斷一個字符串是否是回文。",
    ],
    "math": [
        "請計算 123 × 456 等於多少？請顯示計算過程。",
        "請問一個圓的半徑是 5，面積是多少？（π 取 3.14）",
        "請解方程式 2x + 5 = 15，x 等於多少？",
    ],
    "reasoning": [
        "如果所有的貓都是動物，所有的動物都需要水，那麼貓需要水嗎？請解釋推理過程。",
        "一個農場有雞和兔子，總共 30 隻頭，80 隻腳。請問雞和兔子各有多少隻？",
        "請問「所有的程式設計師都喜歡咖啡」和「小明不喜歡咖啡」，可以得出什麼結論？",
    ],
    "writing": [
        "請寫一首關於春天的短詩（4-8 行）。",
        "請寫一段關於日落的散文描寫（100-200 字）。",
        "請寫一封感謝信，感謝老師的教導。",
    ],
    "translation": [
        "請將「Hello, how are you?」翻譯成中文。",
        "請將「人工智慧正在改變世界」翻譯成英文。",
        "請將「Thank you for your help」翻譯成中文。",
    ],
}

# prompt 關鍵字 → coding prefill anchor (prompt-specific, 避免通用前綴過擬合)
CODING_ANCHORS = [
    ("斐波那契", "```python\ndef fibonacci(n):\n    "),
    ("費氏", "```python\ndef fibonacci(n):\n    "),
    ("冒泡", "```python\ndef bubble_sort(arr):\n    "),
    ("回文", "```python\ndef is_palindrome(s):\n    "),
    ("字符串", "```python\ndef is_palindrome(s):\n    "),
]


def _resolve_prompts(profile, reference_rules):
    """優先取 reference rules 的 prompts, 否則用內建 PROFILE_PROMPTS。"""
    rules = (reference_rules or {}).get(profile) or {}
    prompts = rules.get("prompts") or PROFILE_PROMPTS.get(profile)
    return prompts or [""]


def _coding_anchor(prompt):
    for kw, anchor in CODING_ANCHORS:
        if kw in prompt:
            return anchor
    return "```python\n"


def build_payload(profile, model, max_tokens, seed=0, prompt=None, prompt_index=0):
    """固定 seed (預設 0) + temperature 0 (greedy), 讓採樣確定性。
    seed 對 greedy 無作用, 但若 server 端有非 greedy 路徑 (如 MTP draft),
    固定 seed 可消掉殘餘採樣噪聲。multi-prompt: prompt_index 輪詢 PROFILE_PROMPTS。
    """
    prompts = _resolve_prompts(profile, None)
    p = prompt if prompt is not None else prompts[prompt_index % len(prompts)]
    stops_std = ["<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS

    if profile == "qa-zh":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 24,
            "chat_template_kwargs": {"assistant_prefill": "答:"},
            "stop": ["。"] + stops_std,
        }

    if profile == "longform-zh":
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable. Keep the final answer lean. Use plain prose. Never drop correctness.",
                },
                {"role": "user", "content": p},
            ],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 220,
            "stop": LONGFORM_STOP,
        }
        # 2026-09-08 FIX: presence_penalty 套用到所有 longform prompts (不只巴黎)。
        # 量測: IQ3_XXS 在 pp=0 下開頭即 echo thinking/response marker loop
        # (pp=0 -> "\n thinking\n response...", pp=1.5 -> 正常開頭), 與 prefill 無關。
        # 巴黎 prompt 保留已調校的 open-ended 錨定 (2026-09-06 FIX)。
        payload["presence_penalty"] = 1.5
        if "巴黎" in p:
            payload["chat_template_kwargs"] = {
                "assistant_prefill": "答:巴黎之所以成為法國的政治與文化中心,主要因為",
            }
        return payload

    if profile == "coding":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 512,
            # def 錨定進 code-mode (prompt-specific anchor)
            "chat_template_kwargs": {
                "assistant_prefill": _coding_anchor(p),
            },
            # 2026-10-19 FIX: 裸 "```" 會在「開 fence」即命中 stop —— assistant_prefill 開頭就是
            # "```python"（或無 prefill 時模型首 token 即開 fence），於是第一個 token 就 finish=stop、
            # content 空（completion_tokens=1），coding 質量門徹底失效（A/B/C + HEAD 二進制對照見
            # docs/REPLAY_CODING_STOP_FENCE_2026-10-19.md；HEAD 58abe27ad 已復現，與 commit2/工作區 dylib 無關）。
            # 改用閉 fence "\n```"（行首獨佔三反引號、前有換行）：開 fence "```python" 不命中，模型寫完整段
            # 代碼後在閉 fence 停止（實測 87 token 完整 fibonacci，與 baseline dbe5f8eb1 一致）。
            "stop": ["\n```", "<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS,
            # 2026-09-09 FIX: presence_penalty 套用到 coding prompts (與 longform/math/reasoning 一致)。
            # 量測: IQ3_XXS 在 pp=0 下 prefill 後即進入交替循環 ("fibonacci(n):" / "def fibonacci(n):")，
            # pp=1.5 抑制重複 → 正常生成函數體。白皮書 §0.2 明確: 長輸出 profile (longform/coding) 的 pp=1.5 是必要修正。
            "presence_penalty": 1.5,
        }

    if profile == "math":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 300,
            "stop": stops_std,
            # 2026-09-08: IQ3_XXS 裸生成開頭即 echo prompt 迴圈, presence_penalty 必要
            "presence_penalty": 1.5,
        }

    if profile == "reasoning":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 400,
            "stop": stops_std,
            "presence_penalty": 1.5,
        }

    if profile == "writing":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 250,
            "stop": stops_std,
            "presence_penalty": 1.5,
        }

    if profile == "translation":
        return {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens or 120,
            "stop": stops_std,
        }

    raise ValueError(f"unsupported profile: {profile}")




# ---------------------------------------------------------------------------
# 內存監控 (psutil + ps fallback)
# ---------------------------------------------------------------------------

def get_rss_mb(pid):
    """讀 PID 的 RSS (MB),用 psutil (如果裝) 或 ps -o rss=。失敗回 None。"""
    if pid is None or pid <= 0:
        return None
    # 1) psutil
    try:
        import psutil  # type: ignore
        p = psutil.Process(pid)
        return p.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    # 2) ps fallback (macOS / Linux 都有 ps)
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip()) / 1024.0  # KB → MB
    except Exception:
        pass
    return None


def sample_rss_during(fn, pid, sample_interval_s=0.5):
    """在 fn() 執行期間,每 sample_interval_s 採樣 RSS,回傳 (result, samples_list)。"""
    samples = []
    before = get_rss_mb(pid)
    if before is not None:
        samples.append(("before", before))
    import threading
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            rss = get_rss_mb(pid)
            if rss is not None:
                samples.append(("during", rss))
            stop.wait(sample_interval_s)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join(timeout=2)
    after = get_rss_mb(pid)
    if after is not None:
        samples.append(("after", after))
    return result, samples


def summarize_rss(samples):
    """從 [(phase, rss_mb), ...] 算 peak / avg / delta。"""
    rss_values = [rss for _, rss in samples if rss is not None]
    if not rss_values:
        return {"peak_mb": None, "avg_mb": None, "before_mb": None, "after_mb": None, "delta_mb": None}
    before = next((r for p, r in samples if p == "before"), None)
    after = next((r for p, r in samples if p == "after"), None)
    return {
        "peak_mb": round(max(rss_values), 1),
        "avg_mb": round(sum(rss_values) / len(rss_values), 1),
        "before_mb": round(before, 1) if before is not None else None,
        "after_mb": round(after, 1) if after is not None else None,
        "delta_mb": round((after - before), 1) if (after is not None and before is not None) else None,
        "samples": [[p, round(r, 1)] for p, r in samples],
    }


# ---------------------------------------------------------------------------
# 機器狀態分檔量化 (Machine State Tiering 2026-09-07)
# ---------------------------------------------------------------------------

def _run_cmd(cmd, timeout=5):
    """執行 shell 命令,回傳 stdout 字串;失敗回空字串。"""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except Exception:
        return ""


def get_memory_state():
    """[Machine State Tiering] 收集完整記憶體狀態並分檔。
    
    Returns:
        dict with:
          - total_gb, free_gb, active_gb, inactive_gb, speculative_gb
          - compressed_gb, wired_gb, used_pct, free_pct
          - tier: 'A (寬鬆)' / 'B (正常)' / 'C (緊張)' / 'D (危險)'
          - swap_total_mb, swap_used_mb, swap_free_mb
    """
    result = {
        "total_gb": None, "free_gb": None, "active_gb": None,
        "inactive_gb": None, "speculative_gb": None,
        "compressed_gb": None, "wired_gb": None,
        "used_pct": None, "free_pct": None,
        "tier": None,
        "swap_total_mb": None, "swap_used_mb": None, "swap_free_mb": None,
    }
    
    # macOS vm_stat
    vm_out = _run_cmd("vm_stat")
    if vm_out:
        page_size = 16384  # macOS default page size
        stats = {}
        for line in vm_out.splitlines()[1:]:
            parts = line.strip().rstrip('.').split(':')
            if len(parts) == 2:
                try:
                    stats[parts[0].strip()] = int(parts[1].strip())
                except ValueError:
                    pass
        
        total = sum(stats.values())
        free = stats.get('Pages free', 0)
        active = stats.get('Pages active', 0)
        inactive = stats.get('Pages inactive', 0)
        speculative = stats.get('Pages speculative', 0)
        compressed = stats.get('Pages occupied by compressor', 0)
        wired = stats.get('Pages wired down', 0)
        
        if total > 0:
            free_pct = free / total * 100
            used_pct = (total - free - speculative) / total * 100
            
            result.update({
                "total_gb": round(total * page_size / 1024**3, 1),
                "free_gb": round(free * page_size / 1024**3, 2),
                "active_gb": round(active * page_size / 1024**3, 2),
                "inactive_gb": round(inactive * page_size / 1024**3, 2),
                "speculative_gb": round(speculative * page_size / 1024**3, 2),
                "compressed_gb": round(compressed * page_size / 1024**3, 2),
                "wired_gb": round(wired * page_size / 1024**3, 2),
                "used_pct": round(used_pct, 1),
                "free_pct": round(free_pct, 1),
            })
            
            # 分檔
            if free_pct > 30:
                result["tier"] = "A (寬鬆)"
            elif free_pct > 20:
                result["tier"] = "B (正常)"
            elif free_pct > 10:
                result["tier"] = "C (緊張)"
            else:
                result["tier"] = "D (危險)"
    
    # Swap
    swap_out = _run_cmd("sysctl vm.swapusage")
    if swap_out and "=" in swap_out:
        try:
            # vm.swapusage: total = 2048.00M  used = 722.50M  free = 1325.50M
            parts = swap_out.split("=")
            if len(parts) >= 4:
                result["swap_total_mb"] = float(parts[1].strip().rstrip("M").strip())
                result["swap_used_mb"] = float(parts[2].strip().rstrip("M").strip())
                result["swap_free_mb"] = float(parts[3].strip().rstrip("M").strip())
        except (ValueError, IndexError):
            pass
    
    return result


def get_cpu_state():
    """[Machine State Tiering] 收集 CPU 狀態。
    
    Returns:
        dict with:
          - user_pct, sys_pct, idle_pct
          - load_avg_1m, load_avg_5m, load_avg_15m
          - tier: 'A (正常)' / 'B (輕度負載)' / 'C (重度負載)'
    """
    result = {
        "user_pct": None, "sys_pct": None, "idle_pct": None,
        "load_avg_1m": None, "load_avg_5m": None, "load_avg_15m": None,
        "tier": None,
    }
    
    # CPU usage (top -l 1)
    top_out = _run_cmd("top -l 1 -n 0 | grep 'CPU usage'")
    if top_out:
        try:
            # CPU usage: 12.50% user, 11.18% sys, 76.31% idle
            parts = top_out.split(":")[-1].split(",")
            for part in parts:
                part = part.strip()
                if "user" in part:
                    result["user_pct"] = float(part.replace("% user", "").strip())
                elif "sys" in part:
                    result["sys_pct"] = float(part.replace("% sys", "").strip())
                elif "idle" in part:
                    result["idle_pct"] = float(part.replace("% idle", "").strip())
        except (ValueError, IndexError):
            pass
    
    # Load average
    load_out = _run_cmd("sysctl vm.loadavg")
    if load_out and "{" in load_out:
        try:
            # vm.loadavg: { 3.44 4.06 3.78 }
            vals = load_out.split("{")[1].split("}")[0].strip().split()
            if len(vals) >= 3:
                result["load_avg_1m"] = float(vals[0])
                result["load_avg_5m"] = float(vals[1])
                result["load_avg_15m"] = float(vals[2])
        except (ValueError, IndexError):
            pass
    
    # 分檔 (based on idle %)
    if result["idle_pct"] is not None:
        if result["idle_pct"] > 50:
            result["tier"] = "A (正常)"
        elif result["idle_pct"] > 30:
            result["tier"] = "B (輕度負載)"
        else:
            result["tier"] = "C (重度負載)"
    
    return result


def get_process_interference():
    """[Machine State Tiering] 偵測其他進程干擾。
    
    Returns:
        dict with:
          - other_llama_servers: list of PIDs
          - other_quantization_processes: list of PIDs
          - heavy_processes: list of (pid, cpu_pct, mem_pct, command)
          - interference_level: 0 (無) / 1 (輕度) / 2 (嚴重)
    """
    result = {
        "other_llama_servers": [],
        "other_quantization_processes": [],
        "heavy_processes": [],
        "interference_level": 0,
    }
    
    # 找其他 llama-server 進程 (排除自己)
    ps_out = _run_cmd("ps aux | grep 'llama-server' | grep -v grep")
    if ps_out:
        for line in ps_out.splitlines():
            parts = line.split()
            if len(parts) >= 11:
                pid = parts[1]
                result["other_llama_servers"].append(int(pid))
    
    # 找量化進程 (llama-quantize, convert, etc.)
    quant_out = _run_cmd("ps aux | grep -E 'llama-quantize|convert|quantize' | grep -v grep")
    if quant_out:
        for line in quant_out.splitlines():
            parts = line.split()
            if len(parts) >= 11:
                pid = parts[1]
                result["other_quantization_processes"].append(int(pid))
    
    # 找重度進程 (CPU > 50% 或 MEM > 10%)
    heavy_out = _run_cmd("ps aux | sort -k3 -nr | head -10")
    if heavy_out:
        for line in heavy_out.splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 11:
                try:
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    pid = int(parts[1])
                    cmd = " ".join(parts[10:])[:80]
                    if cpu > 50 or mem > 10:
                        result["heavy_processes"].append({
                            "pid": pid, "cpu_pct": cpu, "mem_pct": mem, "command": cmd
                        })
                except (ValueError, IndexError):
                    pass
    
    # 干擾等級
    if len(result["other_llama_servers"]) > 1 or len(result["other_quantization_processes"]) > 0:
        result["interference_level"] = 2  # 嚴重
    elif len(result["heavy_processes"]) > 3:
        result["interference_level"] = 1  # 輕度
    else:
        result["interference_level"] = 0  # 無
    
    return result


def get_metal_gpu_state(log_file=None):
    """[Machine State Tiering] 收集 Metal/GPU 狀態。
    
    Args:
        log_file: optional path to llama-server log file for OOM warning detection
    
    Returns:
        dict with:
          - oom_warnings: count of OOM warnings in log
          - oom_tier: 0 (無) / 1 (加載時) / 2 (decode 時)
          - gpu_available: whether GPU is accessible
    """
    result = {
        "oom_warnings": 0,
        "oom_tier": 0,
        "gpu_available": True,
    }
    
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, "r", errors="ignore") as f:
                content = f.read()
                oom_count = content.lower().count("out of memory") + content.lower().count("oom")
                result["oom_warnings"] = oom_count
                
                if oom_count > 0:
                    # 判斷是加載時還是 decode 時
                    if "decode" in content.lower() and "out of memory" in content.lower():
                        result["oom_tier"] = 2  # decode 時
                    else:
                        result["oom_tier"] = 1  # 加載時
        except Exception:
            pass
    
    return result


def get_system_state(server_pid=None, log_file=None):
    """[Machine State Tiering] 收集完整機器狀態並分檔。
    
    這是 replay gate 的核心指標之一,用於區分「代碼問題」和「機器狀態問題」。
    每次測試前後都會調用,記錄在輸出 JSON 中。
    
    Args:
        server_pid: llama-server PID for process-specific metrics
        log_file: optional path to llama-server log for OOM detection
    
    Returns:
        dict with:
          - timestamp: ISO format timestamp
          - memory: get_memory_state() result
          - cpu: get_cpu_state() result
          - process_interference: get_process_interference() result
          - metal_gpu: get_metal_gpu_state() result
          - server_process: {pid, cpu_pct, mem_pct, rss_mb} if server_pid provided
          - overall_tier: 綜合分檔 'A' / 'B' / 'C' / 'D'
          - expected_decode_tps: 基於機器狀態的預期 decode 速度範圍
    """
    import datetime
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "memory": get_memory_state(),
        "cpu": get_cpu_state(),
        "process_interference": get_process_interference(),
        "metal_gpu": get_metal_gpu_state(log_file),
        "server_process": None,
        "overall_tier": None,
        "expected_decode_tps": None,
    }
    
    # Server process specific metrics
    if server_pid and server_pid > 0:
        try:
            ps_out = _run_cmd(f"ps -p {server_pid} -o %cpu,%mem,rss= 2>/dev/null")
            if ps_out:
                parts = ps_out.split()
                if len(parts) >= 3:
                    result["server_process"] = {
                        "pid": server_pid,
                        "cpu_pct": float(parts[0]),
                        "mem_pct": float(parts[1]),
                        "rss_mb": round(int(parts[2]) / 1024, 1),
                    }
        except (ValueError, IndexError):
            pass
    
    # 綜合分檔 (取最差的那檔)
    tiers = []
    mem_tier = result["memory"].get("tier")
    if mem_tier:
        tiers.append(mem_tier[0])  # 'A', 'B', 'C', 'D'
    cpu_tier = result["cpu"].get("tier")
    if cpu_tier:
        tiers.append(cpu_tier[0])
    if result["process_interference"]["interference_level"] == 2:
        tiers.append("D")
    elif result["process_interference"]["interference_level"] == 1:
        tiers.append("C")
    if result["metal_gpu"]["oom_tier"] == 2:
        tiers.append("D")
    elif result["metal_gpu"]["oom_tier"] == 1:
        tiers.append("C")
    
    if tiers:
        result["overall_tier"] = max(tiers)  # 'D' is worst, 'A' is best
    
    # 基於機器狀態的預期 decode 速度範圍
    overall = result["overall_tier"]
    if overall == "A":
        result["expected_decode_tps"] = "25+ t/s"
    elif overall == "B":
        result["expected_decode_tps"] = "22-25 t/s"
    elif overall == "C":
        result["expected_decode_tps"] = "18-22 t/s"
    elif overall == "D":
        result["expected_decode_tps"] = "< 18 t/s, 可能 OOM"
    else:
        result["expected_decode_tps"] = "unknown"
    
    return result


def system_state_to_summary(state):
    """[Machine State Tiering] 將完整機器狀態壓縮成一行摘要,方便日誌輸出。"""
    if not state:
        return "system_state: unknown"
    
    mem = state.get("memory", {})
    cpu = state.get("cpu", {})
    overall = state.get("overall_tier", "?")
    expected = state.get("expected_decode_tps", "?")
    
    return (
        f"tier={overall} | "
        f"mem_free={mem.get('free_gb', '?')}GB ({mem.get('free_pct', '?')}%) | "
        f"cpu_idle={cpu.get('idle_pct', '?')}% | "
        f"expected={expected}"
    )


# ---------------------------------------------------------------------------
# 質量評分 (啟發式, 從 --reference 讀 rules)
# ---------------------------------------------------------------------------

def detect_phrase_loop(text, min_unit=6, max_unit=300, min_repeat=3):
    """偵測片語級重複迴圈: 相同長度 >= min_unit 的片語連續重複 >= min_repeat 次。

    例: 「從12世紀起就是法國王國的首都,並且匯聚了」連續重複 3 次 → 回傳
    (unit, start, count)。在 raw text 與移除全部空白後的 text 上各檢查一次,
    避免換行/空白打斷連續性 (例如每行重複的長句)。回傳 None 代表沒有片語迴圈。

    i 用 step=1 掃描 (不能用 p 步進), 否則會跳過前面有前綴文字時的真實起點。
    """
    def _find(t):
        n = len(t)
        if n < min_unit * min_repeat:
            return None
        max_len = min(max_unit, n // min_repeat)
        for p in range(min_unit, max_len + 1):
            limit = n - p * min_repeat
            i = 0
            while i <= limit:
                block = t[i:i + p]
                cnt = 1
                j = i + p
                while j + p <= n and t[j:j + p] == block:
                    cnt += 1
                    j += p
                    if cnt >= min_repeat:
                        return (block, i, cnt)
                i += 1
        return None

    if not text:
        return None
    found = _find(text)
    if found:
        return found
    collapsed = "".join(text.split())  # 移除全部空白, 抓跨行/跨模板標記重複
    if collapsed == text:
        return None
    return _find(collapsed)


def evaluate_quality(profile, content, finish_reason, reference_rules, cgc_stats=None):
    """根據 reference rules + CGC three-factor measurement 給 0.0-1.0 score。

    [CGC Three-Factor Measurement 2026-09-07] Directly measures the three factors that cause
    quality degradation, instead of relying on output heuristics alone:
      Factor 1 (count-cold): fraction of selected experts that are non-resident (>10% = warning, >15% = fail)
      Factor 2 (ZERO-slot pollution): fraction of steps that use the ZERO slot (>5% = warning, >10% = fail)
      Factor 3 (memory pressure): resident memory vs pool capacity (>90% utilization = warning)

    These factors are measured directly from the /cgc_stats endpoint, not inferred from output.
    They are combined with the traditional reference-rule checks (length, keywords, loop detection).

    回傳 (score, checks_list)。
    """
    rules = reference_rules.get(profile, {}) if reference_rules else {}
    checks = []

    # [CGC Three-Factor Measurement] Factor checks first (directly measured, most reliable)
    three_factor_score = 1.0
    if cgc_stats and cgc_stats.get("available"):
        # Factor 1: count-cold rate
        f1 = cgc_stats.get("factor1_count_cold", {})
        cold_rate = f1.get("cold_rate_all_pct")
        if cold_rate is not None:
            if cold_rate > 15:
                checks.append({"check": "factor1_count_cold", "rate_pct": cold_rate,
                              "threshold": 15, "result": "fail",
                              "note": "count-cold > 15% = severe quality degradation risk"})
                three_factor_score = min(three_factor_score, 0.3)
            elif cold_rate > 10:
                checks.append({"check": "factor1_count_cold", "rate_pct": cold_rate,
                              "threshold": 10, "result": "warning",
                              "note": "count-cold > 10% = moderate quality degradation risk"})
                three_factor_score = min(three_factor_score, 0.7)
            else:
                checks.append({"check": "factor1_count_cold", "rate_pct": cold_rate,
                              "result": "pass", "note": "count-cold < 10% = acceptable"})

        # Factor 2: ZERO-slot pollution
        f2 = cgc_stats.get("factor2_zero_slot", {})
        zero_enabled = f2.get("zero_slot_enabled", False)
        zero_rate = f2.get("zero_slot_usage_rate_pct")
        if zero_enabled and zero_rate is not None:
            if zero_rate > 10:
                checks.append({"check": "factor2_zero_slot", "rate_pct": zero_rate,
                              "threshold": 10, "result": "fail",
                              "note": "ZERO-slot usage > 10% = severe quality degradation (logits zeroed)"})
                three_factor_score = min(three_factor_score, 0.3)
            elif zero_rate > 5:
                checks.append({"check": "factor2_zero_slot", "rate_pct": zero_rate,
                              "threshold": 5, "result": "warning",
                              "note": "ZERO-slot usage > 5% = moderate quality degradation"})
                three_factor_score = min(three_factor_score, 0.7)
            else:
                checks.append({"check": "factor2_zero_slot", "rate_pct": zero_rate,
                              "result": "pass", "note": "ZERO-slot usage < 5% = acceptable"})
        elif zero_enabled:
            checks.append({"check": "factor2_zero_slot", "result": "skip",
                          "note": "ZERO-slot enabled but rate not available"})

        # Factor 3: memory pressure
        f3 = cgc_stats.get("factor3_memory_pressure", {})
        slot_util = f3.get("slot_utilization_pct")
        resident_mib = f3.get("resident_mib")
        if slot_util is not None:
            if slot_util > 90:
                checks.append({"check": "factor3_memory_pressure", "slot_util_pct": slot_util,
                              "resident_mib": resident_mib, "threshold": 90, "result": "warning",
                              "note": "slot utilization > 90% = memory pressure, fills may stall"})
                three_factor_score = min(three_factor_score, 0.8)
            else:
                checks.append({"check": "factor3_memory_pressure", "slot_util_pct": slot_util,
                              "resident_mib": resident_mib, "result": "pass",
                              "note": "slot utilization < 90% = acceptable"})
    else:
        checks.append({"check": "three_factors", "result": "skip",
                      "note": "CGC stats not available (expert cache not enabled or endpoint not reachable)"})

    # Traditional reference-rule checks
    if not rules:
        # No reference rules: use three-factor score as the primary quality metric
        final_score = three_factor_score
        checks.append({"check": "no_reference", "result": "skip",
                      "note": "no reference rules provided, using three-factor measurement"})
        return round(final_score, 3), checks

    checks = []
    score = 0.0
    weight_sum = 0.0

    # 1) length check
    length_min = rules.get("length_min")
    length_max = rules.get("length_max")
    if length_min is not None or length_max is not None:
        n = len(content or "")
        ok = True
        if length_min is not None and n < length_min:
            ok = False
        if length_max is not None and n > length_max:
            ok = False
        checks.append({"check": "length", "actual": n, "min": length_min, "max": length_max, "result": "pass" if ok else "fail"})
        w = 0.2
        score += (1.0 if ok else 0.0) * w
        weight_sum += w

    # 2) required keyword (any of)
    required_any = rules.get("required_any") or []
    if required_any:
        hit = any(kw in (content or "") for kw in required_any)
        checks.append({"check": "required_any", "keywords": required_any, "result": "pass" if hit else "fail"})
        w = 0.4
        score += (1.0 if hit else 0.0) * w
        weight_sum += w

    # 3) required all
    required_all = rules.get("required_all") or []
    if required_all:
        missing = [kw for kw in required_all if kw not in (content or "")]
        ok = not missing
        checks.append({"check": "required_all", "keywords": required_all, "missing": missing, "result": "pass" if ok else "fail"})
        w = 0.2
        score += (1.0 if ok else 0.0) * w
        weight_sum += w

    # 4) forbidden (any of)
    forbidden = rules.get("forbidden") or []
    if forbidden:
        hit = any(kw in (content or "") for kw in forbidden)
        checks.append({"check": "forbidden", "keywords": forbidden, "result": "pass" if not hit else "fail"})
        w = 0.2
        score += (1.0 if not hit else 0.0) * w
        weight_sum += w

    # 5) finish_reason check
    ok_finish = rules.get("ok_finish_reasons")
    if ok_finish:
        ok = finish_reason in ok_finish
        checks.append({"check": "finish_reason", "actual": finish_reason, "ok": ok_finish, "result": "pass" if ok else "fail"})
        w = 0.2
        score += (1.0 if ok else 0.0) * w
        weight_sum += w

    # 6) loop detection (consecutive identical chars)
    loop_ok = True
    loop_max_run = rules.get("loop_max_run", 8)
    if content:
        max_run = 1
        cur_run = 1
        for i in range(1, len(content)):
            if content[i] == content[i-1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1
        loop_ok = max_run <= loop_max_run
        checks.append({"check": "loop_run", "max_run": max_run, "limit": loop_max_run, "result": "pass" if loop_ok else "fail"})
        w = 0.2
        score += (1.0 if loop_ok else 0.0) * w
        weight_sum += w

    # 7) phrase-level loop detection: 相同 6+ 字元片語連續重複 >= 3 次 = 語意迴圈
    #    舊 loop_run 只抓連續相同字元, 抓不到片語級重複 (如長句迴圈輸出),
    #    造成 loop 內容仍拿 1.0 高分。此 check 權重最高 + 硬閘門壓分。
    phrase_loop = detect_phrase_loop(content) if content else None
    if phrase_loop is not None:
        unit, start, cnt = phrase_loop
    else:
        unit = start = cnt = None
    checks.append({
        "check": "loop_phrase",
        "result": "pass" if phrase_loop is None else "fail",
        "min_unit": 6,
        "max_unit": 300,
        "min_repeat": 3,
        "unit": unit,
        "start": start,
        "count": cnt,
    })
    w = 0.5
    score += (1.0 if phrase_loop is None else 0.0) * w
    weight_sum += w

    if weight_sum == 0:
        # No reference rules: use three-factor score as the primary quality metric
        final = three_factor_score
    else:
        final = score / weight_sum
        # [CGC Three-Factor Measurement] Combine reference-rule score with directly-measured factors.
        # Three-factor score acts as a hard gate: if factors indicate severe quality degradation,
        # the final score cannot exceed the three-factor score (no matter how good the output looks).
        final = min(final, three_factor_score)

    # 硬閘門: 偵測到語意迴圈 (片語重複 或 連續字元爆量) 時壓到 0.3 以下,
    # 不讓迴圈輸出靠 required_any 命中關鍵字而拿到 0.5+ 的假高分。
    if phrase_loop is not None or not loop_ok:
        final = min(final, 0.3)

    # Add three-factor summary to checks
    checks.append({
        "check": "three_factor_combined_score",
        "three_factor_score": round(three_factor_score, 3),
        "reference_rule_score": round(score / weight_sum, 3) if weight_sum > 0 else None,
        "final_score": round(final, 3),
        "result": "pass" if final >= 0.9 else ("warning" if final >= 0.7 else "fail"),
    })

    return round(final, 3), checks


# ---------------------------------------------------------------------------
# 跑單一 profile
# ---------------------------------------------------------------------------

def run_profile(args, profile, reference_rules, prompt_index=0):
    prompts = _resolve_prompts(profile, reference_rules)
    p = prompts[prompt_index % len(prompts)]
    payload = build_payload(profile, args.model, args.max_tokens, prompt=p)
    if args.print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return None

    url = args.base_url.rstrip("/") + "/chat/completions"
    server_pid = args.server_pid
    log_file = getattr(args, "log_file", None)

    # [Machine State Tiering] Get system state BEFORE the request (baseline)
    system_state_before = get_system_state(server_pid=server_pid, log_file=log_file)
    print(f"  [system] before: {system_state_to_summary(system_state_before)}")

    # [CGC Three-Factor Measurement] Get stats BEFORE the request (baseline)
    cgc_stats_before = get_cgc_stats(args.base_url, timeout=5)

    # 跑 + 同時採樣 RSS
    def _do_request():
        return http_json(url, payload, args.timeout)

    if server_pid > 0:
        resp, rss_samples = sample_rss_during(_do_request, server_pid, args.rss_sample_interval)
    else:
        resp = _do_request()
        rss_samples = []

    # [CGC Three-Factor Measurement] Get stats AFTER the request
    cgc_stats_after = get_cgc_stats(args.base_url, timeout=5)

    # [Machine State Tiering] Get system state AFTER the request
    system_state_after = get_system_state(server_pid=server_pid, log_file=log_file)
    print(f"  [system] after:  {system_state_to_summary(system_state_after)}")

    # Compute delta (this request's contribution)
    cgc_stats_delta = compute_cgc_stats_delta(cgc_stats_before, cgc_stats_after)

    content = resp["choices"][0]["message"]["content"]
    finish_reason = resp["choices"][0]["finish_reason"]
    timings = resp.get("timings", {})
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    decode_tps = timings.get("predicted_per_second")
    prompt_tps = timings.get("prompt_per_second")
    draft_n = timings.get("draft_n")
    draft_n_accepted = timings.get("draft_n_accepted")
    accept_pct = None
    if isinstance(draft_n, (int, float)) and draft_n:
        accept_pct = (float(draft_n_accepted or 0) / float(draft_n)) * 100.0

    # 過濾 degenerate decode_tps: 1 token + predicted_ms<5ms 是 stop token 立即觸發,
    # 1000/predicted_ms 會膨脹成 1e6 t/s, 不代表真實 decode 速度 → 設 None
    degenerate_decode = False
    if isinstance(decode_tps, (int, float)) and decode_tps is not None:
        if (predicted_n in (0, 1)) or (isinstance(predicted_ms, (int, float)) and predicted_ms < 5.0):
            degenerate_decode = True
            decode_tps = None

    quality_score, quality_checks = evaluate_quality(
        profile, content, finish_reason, reference_rules, cgc_stats_delta)

    # [Machine State Tiering] 判斷速度是否符合機器狀態預期
    speed_vs_expected = "unknown"
    if decode_tps and system_state_after.get("overall_tier"):
        tier = system_state_after["overall_tier"]
        if tier == "A" and decode_tps >= 25:
            speed_vs_expected = "meets_or_exceeds"
        elif tier == "B" and 22 <= decode_tps <= 25:
            speed_vs_expected = "meets"
        elif tier == "C" and 18 <= decode_tps <= 22:
            speed_vs_expected = "meets"
        elif tier == "D" and decode_tps < 18:
            speed_vs_expected = "meets (expected slow)"
        elif tier in ("A", "B") and decode_tps < 20:
            speed_vs_expected = "below_expected (possible regression)"
        else:
            speed_vs_expected = "within_range"

    return {
        "profile": profile,
        "finish_reason": finish_reason,
        "quality": {
            "score": quality_score,
            "checks": quality_checks,
        },
        "speed": {
            "prefill_tps": round(prompt_tps, 2) if isinstance(prompt_tps, (int, float)) else None,
            "decode_tps": round(decode_tps, 2) if isinstance(decode_tps, (int, float)) else None,
            "decode_tps_degenerate": degenerate_decode,
            "completion_tokens": predicted_n,
            "draft_accept_pct": round(accept_pct, 2) if accept_pct is not None else None,
            "speed_vs_expected": speed_vs_expected,
        },
        "memory": summarize_rss(rss_samples),
        "timings": {
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_n": predicted_n,
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
        },
        # [CGC Three-Factor Measurement] directly measured factors that cause quality degradation
        "three_factors": cgc_stats_delta,
        # [Machine State Tiering] complete system state before/after for regression attribution
        "system_state": {
            "before": system_state_before,
            "after": system_state_after,
            "overall_tier": system_state_after.get("overall_tier"),
            "expected_decode_tps": system_state_after.get("expected_decode_tps"),
            "speed_vs_expected": speed_vs_expected,
        },
        # raw content 放最後,方便人工 debug
        "content": content,
    }


# ---------------------------------------------------------------------------
# Robust 量測: warmup + N runs + median 聚合
# ---------------------------------------------------------------------------

def _median(vals):
    """中位數; 空清單回 None。"""
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _variance(vals):
    """樣本變異數; 少於 2 筆回 0。"""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def run_profile_robust(args, profile, reference_rules):
    """warmup 後跑 N 次 (--runs), 用 median 聚合品質/速度, peak 取 RSS 最大值。

    回傳的 JSON schema 與 run_profile 相容 (quality.score / speed.decode_tps /
    speed.prefill_tps / memory.peak_mb 仍是單一數值), 額外加:
      - quality.scores / quality.variance / quality.runs
      - speed.decode_tps_runs / speed.prefill_tps_runs
      - memory.peak_runs
      - content = 品質最接近 median 的那次輸出 (供 debug)
    """
    # 1) warmup: 丟棄結果, 讓 expert pool 進到該 profile 的典型冷熱狀態
    if args.warmup:
        print(f"[replay] warmup {profile} (discard) ...", file=sys.stderr)
        try:
            run_profile(args, profile, reference_rules, prompt_index=0)
        except Exception as e:
            print(f"[replay] WARN warmup {profile} failed: {e}", file=sys.stderr)

    # 2) N 次量測 (multi-prompt: run i 輪詢 prompt i, 避免過擬合單一 prompt)
    runs = []
    for i in range(args.runs):
        print(f"[replay] run {i + 1}/{args.runs} {profile} (prompt {i % 3}) ...", file=sys.stderr)
        r = run_profile(args, profile, reference_rules, prompt_index=i)
        if r is not None:
            runs.append(r)
    if not runs:
        raise RuntimeError(f"no successful runs for {profile}")

    # 3) median 聚合
    q_scores = [r["quality"]["score"] for r in runs if r["quality"]["score"] is not None]
    d_tps = [r["speed"]["decode_tps"] for r in runs if r["speed"]["decode_tps"] is not None]
    pf_tps = [r["speed"]["prefill_tps"] for r in runs if r["speed"]["prefill_tps"] is not None]
    ap_pct = [r["speed"]["draft_accept_pct"] for r in runs if r["speed"]["draft_accept_pct"] is not None]
    rss_peak = [r["memory"]["peak_mb"] for r in runs if r["memory"]["peak_mb"] is not None]

    med_q = _median(q_scores)
    # 代表樣本 = 品質分數最接近 median 的那一次 (保留其 content/checks 供人工 debug)
    rep = min(runs, key=lambda r: abs(r["quality"]["score"] - med_q) if med_q is not None else 0)

    out = json.loads(json.dumps(rep))
    out["profile"] = profile
    out["runs"] = len(runs)
    out["quality"] = {
        "score": round(med_q, 3) if med_q is not None else None,
        "scores": [round(s, 3) for s in q_scores],
        "variance": round(_variance(q_scores), 3),
        "checks": rep["quality"]["checks"],
    }
    med_d = _median(d_tps)
    med_pf = _median(pf_tps)
    out["speed"] = {
        **rep["speed"],
        "decode_tps": round(med_d, 2) if med_d is not None else None,
        "prefill_tps": round(med_pf, 2) if med_pf is not None else None,
        "draft_accept_pct": round(_median(ap_pct), 2) if ap_pct else None,
        "decode_tps_runs": [round(v, 2) for v in d_tps],
        "prefill_tps_runs": [round(v, 2) for v in pf_tps],
    }
    out["memory"] = {
        **rep["memory"],
        "peak_mb": round(max(rss_peak), 1) if rss_peak else None,
        "peak_runs": [round(v, 1) for v in rss_peak],
    }
    return out


# ---------------------------------------------------------------------------
# Aggregated 摘要 (3 profile 平均 / 中位數)
# ---------------------------------------------------------------------------

def aggregate(profiles_dict):
    """從 3 個 profile 算 speed / memory / system_state 的平均 + 中位數 + min/max,供 precommit hook 看趨勢。"""
    speed_decode = [p["speed"]["decode_tps"] for p in profiles_dict.values() if p["speed"]["decode_tps"] is not None]
    speed_prefill = [p["speed"]["prefill_tps"] for p in profiles_dict.values() if p["speed"]["prefill_tps"] is not None]
    quality_scores = [p["quality"]["score"] for p in profiles_dict.values()]
    mem_peaks = [p["memory"]["peak_mb"] for p in profiles_dict.values() if p["memory"]["peak_mb"] is not None]
    
    # [Machine State Tiering] 聚合機器狀態
    system_tiers = [p.get("system_state", {}).get("overall_tier") for p in profiles_dict.values() if p.get("system_state", {}).get("overall_tier")]
    memory_free_pcts = [p.get("system_state", {}).get("after", {}).get("memory", {}).get("free_pct") for p in profiles_dict.values() if p.get("system_state", {}).get("after", {}).get("memory", {}).get("free_pct") is not None]
    cpu_idle_pcts = [p.get("system_state", {}).get("after", {}).get("cpu", {}).get("idle_pct") for p in profiles_dict.values() if p.get("system_state", {}).get("after", {}).get("cpu", {}).get("idle_pct") is not None]
    speed_vs_expected = [p.get("speed", {}).get("speed_vs_expected") for p in profiles_dict.values() if p.get("speed", {}).get("speed_vs_expected")]

    def _stats(vals):
        if not vals:
            return {"avg": None, "median": None, "min": None, "max": None, "n": 0}
        s = sorted(vals)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 2),
            "median": round(s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2, 2),
            "min": min(s),
            "max": max(s),
            "n": n,
        }
    
    def _tier_stats(tiers):
        """統計機器狀態分檔分佈。"""
        if not tiers:
            return {"worst": None, "best": None, "distribution": {}}
        from collections import Counter
        dist = dict(Counter(tiers))
        return {
            "worst": max(tiers),  # 'D' is worst
            "best": min(tiers),   # 'A' is best
            "distribution": dist,
        }
    
    def _categorical_stats(cats):
        """統計分類變數分佈。"""
        if not cats:
            return {"distribution": {}, "most_common": None}
        from collections import Counter
        dist = dict(Counter(cats))
        most_common = max(dist, key=dist.get) if dist else None
        return {"distribution": dist, "most_common": most_common}

    return {
        "decode_tps": _stats(speed_decode),
        "prefill_tps": _stats(speed_prefill),
        "quality_score": _stats(quality_scores),
        "peak_rss_mb": _stats(mem_peaks),
        # [Machine State Tiering] 機器狀態聚合
        "system_state": {
            "tier": _tier_stats(system_tiers),
            "memory_free_pct": _stats(memory_free_pcts),
            "cpu_idle_pct": _stats(cpu_idle_pcts),
            "speed_vs_expected": _categorical_stats(speed_vs_expected),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay fixed profile payloads against llama-server + collect quality/speed/memory metrics"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--profile", choices=['qa-zh', 'longform-zh', 'coding', 'math', 'reasoning', 'writing', 'translation'],
                        help="Single profile. Use --all-profiles to run all seven.")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Run all profiles in sequence (7 profiles x 3 prompts each), output combined JSON.")
    parser.add_argument("--model", default="test")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--print-payload", action="store_true")
    parser.add_argument("--bench-output", default=None,
                        help="Write structured JSON to this file (used by precommit hook).")
    parser.add_argument("--reference", default=None,
                        help="JSON file with quality check rules per profile.")
    parser.add_argument("--server-pid", type=int, default=0,
                        help="llama-server PID for RSS sampling (0 = skip memory monitoring).")
    parser.add_argument("--rss-sample-interval", type=float, default=0.5,
                        help="RSS sampling interval in seconds during request.")
    parser.add_argument("--schema-version", type=int, default=1)
    parser.add_argument("--commit", default=os.environ.get("CGC_COMMIT", "unknown"),
                        help="Git commit hash (recorded in bench output, default: CGC_COMMIT env or 'unknown').")
    parser.add_argument("--runs", type=int, default=3,
                        help="每個 profile 跑 N 次取 median (default 3; 1 = 舊的單次行為).")
    parser.add_argument("--warmup", action="store_true",
                        help="量測前先送一次 warmup request 暖 expert pool (default off).")
    parser.add_argument("--seed", type=int, default=0,
                        help="固定採樣 seed (default 0), 消除殘餘採樣噪聲.")
    parser.add_argument("--log-file", default=None,
                        help="llama-server log file path for Metal OOM detection (default: auto-detect from Backup/cgc_logs/).")
    parser.add_argument("--db-save", action="store_true",
                        help="[Database 2026-09-07] 自動保存測試結果到 replay benchmark 數據庫 (data/replay_bench/).")
    parser.add_argument("--db-verdict", choices=["pass", "fail", "neutral"], default=None,
                        help="[Database] 測試結論 (pass/fail/neutral), 用於數據庫記錄.")
    parser.add_argument("--db-version", default=None,
                        help="[Database] 版本編號 (如 v1.0.0, release-2026-09-07), 用於數據庫記錄.")
    parser.add_argument("--db-branch", default=None,
                        help="[Database] git branch 名稱, 用於數據庫記錄 (默認自動檢測).")
    return parser.parse_args()


def _auto_detect_log_file():
    """[Machine State Tiering] 自動檢測最新的 llama-server log 文件。"""
    import glob
    # 優先級: 環境變數 > 項目目錄 > /tmp
    candidates = []
    
    # 項目目錄的 log
    project_logs = os.path.expanduser("~/Documents/flashkv-devserver/Backup/cgc_logs/llama_server_*.log")
    candidates.extend(glob.glob(project_logs))
    
    # /tmp 的 log
    candidates.extend(glob.glob("/tmp/cgc_*.log"))
    candidates.extend(glob.glob("/tmp/llama_server_*.log"))
    
    if not candidates:
        return None
    
    # 取最新的
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    args = parse_args()
    if not args.profile and not args.all_profiles:
        print("error: must specify --profile or --all-profiles", file=sys.stderr)
        return 2

    # [Machine State Tiering] 自動檢測 log 文件
    if not args.log_file:
        args.log_file = _auto_detect_log_file()
        if args.log_file:
            print(f"[system] auto-detected log file: {args.log_file}")
    
    # 把 log_file 傳給 run_profile (通過 args 對象)
    # run_profile 會用 getattr(args, "log_file", None) 獲取

    reference_rules = {}
    if args.reference:
        try:
            with open(args.reference) as f:
                reference_rules = json.load(f)
        except Exception as e:
            print(f"warning: failed to load reference {args.reference}: {e}", file=sys.stderr)
            reference_rules = {}

    profiles = ['qa-zh', 'longform-zh', 'coding', 'math', 'reasoning', 'writing', 'translation'] if args.all_profiles else [args.profile]
    results = {}
    for p in profiles:
        r = run_profile_robust(args, p, reference_rules)
        results[p] = r

    # [CGC Prefetch Drop Audit 2026-09-08] Parse prefetch drop breakdown from server log.
    # The expert cache destructor outputs a detailed breakdown when the server shuts down.
    # We parse the latest server log file to extract the 12 drop point counters.
    prefetch_drop_breakdown = None
    try:
        import glob
        # Determine log file path: explicit --server-log > auto-detect latest
        log_file = args.server_log
        if not log_file:
            # Auto-detect: check Backup/cgc_logs/ first, then /tmp/
            candidates = []
            project_logs = os.path.expanduser("~/Documents/flashkv-devserver/Backup/cgc_logs/llama_server_*.log")
            candidates.extend(glob.glob(project_logs))
            candidates.extend(glob.glob("/tmp/llama_server_*.log"))
            candidates.extend(glob.glob("/tmp/cgc_server_*.log"))
            if candidates:
                log_file = max(candidates, key=os.path.getmtime)

        if log_file and os.path.exists(log_file):
            prefetch_drop_breakdown = parse_prefetch_drop_breakdown_from_log(log_file)
            if prefetch_drop_breakdown and prefetch_drop_breakdown.get("available"):
                print(f"[replay] prefetch drop breakdown parsed from {log_file}: total={prefetch_drop_breakdown.get('total')}", file=sys.stderr)
    except Exception as e:
        print(f"[replay] warning: failed to parse prefetch drop breakdown: {e}", file=sys.stderr)

    if args.all_profiles:
        output = {
            "schema_version": args.schema_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": args.commit,
            "server_pid": args.server_pid,
            "base_url": args.base_url,
            "profiles": results,
            "aggregate": aggregate(results),
            "prefetch_drop_breakdown": prefetch_drop_breakdown,
        }
    else:
        output = results[profiles[0]]
        output["prefetch_drop_breakdown"] = prefetch_drop_breakdown

    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)

    if args.bench_output:
        try:
            with open(args.bench_output, "w") as f:
                f.write(text)
            print(f"[replay] wrote bench output to {args.bench_output}", file=sys.stderr)
        except Exception as e:
            print(f"error: failed to write bench output: {e}", file=sys.stderr)
            return 1
    
    # [Database 2026-09-07] 自動保存結果到數據庫
    if args.db_save:
        try:
            import subprocess
            # 先保存到臨時文件
            temp_file = f"/tmp/replay_bench_{int(time.time())}.json"
            with open(temp_file, "w") as f:
                f.write(text)
            
            # 調用數據庫管理腳本
            db_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_bench_database.py")
            cmd = [sys.executable, db_script, "add", "--input", temp_file]
            # 傳遞 commit (從 args.commit 獲取)
            if args.commit and args.commit != "unknown":
                cmd.extend(["--commit", args.commit])
            # 傳遞 branch
            if args.db_branch:
                cmd.extend(["--branch", args.db_branch])
            # 傳遞 version
            if args.db_version:
                cmd.extend(["--version", args.db_version])
            # 傳遞 verdict
            if args.db_verdict:
                cmd.extend(["--verdict", args.db_verdict])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("[database] 測試結果已保存到數據庫", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            else:
                print(f"[database] 保存失敗: {result.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"[database] 保存異常: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
