#!/usr/bin/env python3
"""
48 題裸問品質測試 — 純 user message、無拐杖（no prefill / pp / stop）。

用法：
  python3 scripts/check/bare_48.py --base-url http://127.0.0.1:8080/v1 --model qwen36 --label IQ3_XXS
  python3 scripts/check/bare_48.py --base-url http://127.0.0.1:8080/v1 --model qwen36 --label IQ4_XS --output /tmp/bare_iq4xs.json

對比兩次結果：
  python3 scripts/check/bare_48.py --compare /tmp/bare_iq3xxs.json /tmp/bare_iq4xs.json

自測（不需要 server）：把每題的 prompt 當答案餵回評分器，證明「只回顯題目」在新尺上不再 PASS：
  python3 scripts/check/bare_48.py --selftest-echo

=============================================================================
兩把尺（同時輸出，逐題並排）
=============================================================================
legacy（舊尺）= 與 37/48 基線逐位元相同的評分器，保留作為歷史比對：
    每個 profile 共用一份扁平的 required_any / forbidden / length，
    逐題只換 prompt。這把尺**不動**，所以舊分數仍可重現。

strict（新尺）= 修掉扁平 required_any 之後的評分器：
    1. 每題有自己的答案鍵（bare48_answer_keys_v1.json）。
       鍵的選法保證「把題目回顯一次」無法命中（--selftest-echo 機械驗證）。
    2. 結構性 echo 閘門：答案開頭逐字複述題目（正規化後最長共同前綴 >= 10 字元），
       或答案裡出現題目長度 >= 18 字元的連續片段 → 判 FAIL。
       這抓的是「回顯題目 / 複述題目當作答」這個實際觀測到的行為，
       而不是靠關鍵字猜。
    3. 開放式題目（longform / writing）沒有關鍵字鍵 —— 改用 novelty 規則：
       答案必須含有至少 N 個「題目裡沒出現過」的字元。回顯的 novelty = 0，
       所以這個規則也是回顯證明（echo-proof）。
    4. 誠實遵守 reference 自己宣告、舊評分器卻從未讀取的 ok_finish_reasons。

strict-core = strict 但**不做**每題 length 覆寫，
    用來把「修 required_any」與「修 harness 長度假陽性」兩個改動的貢獻切開。
    逐題 length 覆寫（qa-zh #1 的 200 上限、#4/#7 的 4 下限）本身是 harness 假陰性修正，
    不該跟 required_any 的修正混為一談。

=============================================================================
為什麼要修（實際觀測到的假陽性）
=============================================================================
舊 qa-zh 的 required_any 是 16 個關鍵字共用給 8 題，其中包含「行星」「法國」「等於」：
  - 「太陽系有幾個行星？請列出名稱。」→ 只回顯題目就含「行星」→ PASS
  - 「法國的首都是哪裡？」→ 回顯含「法國」→ PASS
  - 「請問 15 + 27 等於多少？」→ 回顯含「等於」→ PASS
舊 longform-zh 同理（巴黎／文化／歷史都在 prompt[0] 裡）。
舊 reasoning[3]「請回答是或否」的鍵含「是」→ 任何含「是」的中文都 PASS。

新尺下這些回顯全部 FAIL，且由 --selftest-echo 機械證明（48/48 回顯零分）。
=============================================================================
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

# 复用 replay harness 的评分逻辑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

REFERENCE_FILE = os.path.join(SCRIPT_DIR, "replay_bench_reference_v2.json")
KEYS_FILE = os.path.join(SCRIPT_DIR, "bare48_answer_keys_v1.json")

# 死伺服器防護。伺服器一旦不在，/v1/chat/completions 會「瞬間」回空——
# 之後每一題都會被計為 0 分，看起來像品質崩壞，其實只是沒有人在聽。
# 2026-09-13 的 cap6 對照就是這樣：伺服器在第 26 個請求後收到 signal 結束，
# 剩下的 22 題全部瞬間失敗，而報告只寫「0/48」。
DEAD_SERVER_STREAK = 5


def load_reference():
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_answer_keys():
    with open(KEYS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_loop(text, min_len, min_repeat):
    """Same phrase repeated >= min_repeat times (each >= min_len chars) -> loop.
    與 bare_48_local2 相同（散布重複也算），確保與 37/48 基線同評分器。"""
    if len(text) < min_len * min_repeat:
        return False
    for length in range(min_len, min(40, len(text) // min_repeat) + 1):
        i = 0
        while i + length <= len(text):
            phrase = text[i:i + length]
            if text.count(phrase) >= min_repeat:
                return True
            i += 1
    return False


# ---------------------------------------------------------------------------
# strict 尺：結構性回顯偵測
# ---------------------------------------------------------------------------
# 正規化：拿掉空白與標點，只留可比較的字元。題目與答案都走同一套。
_ECHO_STRIP = re.compile(r"[\s\u3000，。！？、：；「」『』（）()\[\]【】,.!?:;\"'`~～…·\-—_*#>|]+")
ECHO_LCP_MIN = 10   # 答案開頭逐字複述題目 >= 10 個正規化字元 → 回顯
ECHO_SPAN = 18      # 答案含題目的一段 >= 18 正規化字元 → 回顯
ECHO_MAX_SCAN = 4000  # 只在答案前段找題目片段；回顯/迴圈都出現在前面


def _norm(s):
    return _ECHO_STRIP.sub("", s or "").lower()


def detect_echo(prompt, content, lcp_min=ECHO_LCP_MIN, span=ECHO_SPAN):
    """答案是否在複述題目（而不是作答）。

    兩種證據，都是結構性的、與關鍵字無關：
      lcp  — 答案開頭與題目逐字相同的前綴長度 >= lcp_min
             （觀測到的行為：「15+27 等於多少？請只输出答案」被原樣吐回）
      span — 題目中一段長度 >= span 的連續文字原封不動出現在答案裡
             （抓「題目被複述到答案中段」與「題目被重複多次」）
    回傳 (bool, reason)。
    """
    p, c = _norm(prompt), _norm(content)
    if not p or not c:
        return False, None
    if len(p) >= lcp_min:
        n = 0
        while n < min(len(p), len(c)) and p[n] == c[n]:
            n += 1
        if n >= lcp_min:
            return True, f"lcp{n}"
    if len(p) >= span:
        head = c[:ECHO_MAX_SCAN]
        for i in range(len(p) - span + 1):
            if p[i:i + span] in head:
                return True, f"span{span}"
    return False, None


def novelty_chars(prompt, content):
    """答案中「題目沒有出現過」的字元數。

    回顯題目 → 0。這是開放式題目在沒有關鍵字鍵時的回顯證明替代規則：
    要嘛答案帶來新內容，要嘛就是回顯／空轉。
    """
    pset = set(_norm(prompt))
    pset.discard("")
    return sum(1 for ch in _norm(content) if ch not in pset)


def _hit(key, content):
    """ASCII 鍵不分大小寫比對（H2O / AI is ...）；非 ASCII 直接子字串比對。"""
    if not key:
        return False
    if key.isascii():
        return key.lower() in content.lower()
    return key in content


# ---------------------------------------------------------------------------
# 兩把尺
# ---------------------------------------------------------------------------
def score_legacy(profile_rules, content, finish):
    """舊尺：與 37/48 基線逐位元相同（扁平 required_any，逐題不變）。"""
    rules = profile_rules
    checks = []
    if detect_loop(content, rules.get("loop_phrase_min_len", 6), rules.get("loop_phrase_min_repeat", 3)):
        checks.append("loop_phrase")
    ln = len(content)
    if not (rules.get("length_min", 0) <= ln <= rules.get("length_max", 10 ** 9)):
        checks.append("length")
    if rules.get("required_any"):
        if not any(k in content for k in rules["required_any"]):
            checks.append("required_any")
    if rules.get("required_all"):
        if not all(k in content for k in rules["required_all"]):
            checks.append("required_all")
    if rules.get("forbidden"):
        if any(k in content for k in rules["forbidden"]):
            checks.append("forbidden")
    if rules.get("code_block_check"):
        if content.count("```") > rules.get("code_block_max_repeat", 2) * 2:
            checks.append("code_block")
    return len(checks) == 0, checks


def score_strict(profile_rules, key, prompt, content, finish, apply_length_override=True):
    """新尺：每題答案鍵 + 結構性回顯閘門 + 開放式 novelty 規則。

    `key` 為該題的答案鍵（dict，可為 None）。apply_length_override=False 時
    忽略每題的 length 覆寫 → 這就是 strict-core。
    """
    rules = profile_rules
    key = key or {}
    checks = []

    # 1) 結構性回顯閘門（與關鍵字無關，抓「把題目回顯當作答」）
    echoed, why = detect_echo(prompt, content)
    if echoed:
        checks.append(f"echo:{why}")

    # 2) 迴圈（沿用舊尺規則，語義不變）
    if detect_loop(content, rules.get("loop_phrase_min_len", 6), rules.get("loop_phrase_min_repeat", 3)):
        checks.append("loop_phrase")

    # 3) 長度：每題可覆寫 profile 值（strict-core 忽略覆寫）
    if apply_length_override:
        lmin = key.get("length_min", rules.get("length_min", 0))
        lmax = key.get("length_max", rules.get("length_max", 10 ** 9))
    else:
        lmin = rules.get("length_min", 0)
        lmax = rules.get("length_max", 10 ** 9)
    ln = len(content)
    if not (lmin <= ln <= lmax):
        checks.append("length")

    # 4) 內容要求：每題答案鍵，沒有鍵的開放式題目改用 novelty
    groups = key.get("required_all_groups")
    req_any = key.get("required_any")
    if groups:
        for gi, group in enumerate(groups):
            if not any(_hit(k, content) for k in group):
                checks.append(f"required_group{gi + 1}")
    elif req_any:
        if not any(_hit(k, content) for k in req_any):
            checks.append("required_any")
    else:
        nmin = key.get("novelty_min", 20)
        if novelty_chars(prompt, content) < nmin:
            checks.append(f"novelty<{nmin}")

    # 5) forbidden / code block（沿用舊尺）
    if rules.get("forbidden"):
        if any(k in content for k in rules["forbidden"]):
            checks.append("forbidden")
    if rules.get("code_block_check"):
        if content.count("```") > rules.get("code_block_max_repeat", 2) * 2:
            checks.append("code_block")

    # 6) finish_reason：reference 自己宣告了 ok_finish_reasons，
    #    舊評分器從未讀取它。新尺誠實遵守（宣告的規則就該執行）。
    ok_finish = rules.get("ok_finish_reasons")
    if ok_finish and finish not in ok_finish:
        checks.append(f"finish:{finish}")

    return len(checks) == 0, checks


# Sampling sent with every bare-ask request. This used to be hardcoded to 0.0 (greedy),
# which made the sampling axis UNMEASURABLE from this harness: the request-level value
# overrides the server's --temp, so launching with CGC_SERVER_TEMP=0.4 changed nothing while
# this stayed 0.0. The 37/48 baseline recipe (scripts/run_knifeedge_sweep.sh) ran at
# --temp 0.4 --top-p 0.8, so a comparison against that number has to be able to set it.
# Defaults reproduce the historic behaviour exactly (greedy, no top-p).
TEMPERATURE = 0.0
TOP_P = None


def chat_completion(base_url, model, prompt, timeout=120):
    """裸問：純 user message、temp=TEMPERATURE、無拐杖"""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 2048,
        # 故意不設 presence_penalty / stop → 裸問；top_p 只在明確要求時送
    }
    if TOP_P is not None:
        payload["top_p"] = TOP_P
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"content": "", "finish_reason": f"http_{e.code}", "elapsed": time.time() - t0, "error": str(e)}
    except Exception as e:
        return {"content": "", "finish_reason": "error", "elapsed": time.time() - t0, "error": str(e)}

    choice = body.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    finish = choice.get("finish_reason", "unknown")
    return {"content": content, "finish_reason": finish, "elapsed": time.time() - t0}


def run_suite(base_url, model, label, max_per_profile=None, dump_full=False):
    ref = load_reference()
    keys = load_answer_keys()
    profiles = ref["_profiles"]
    results = {}
    total_pass = 0
    total_count = 0
    totals = {"legacy": 0, "core": 0, "strict": 0}
    removed_echo = 0
    removed_other = 0
    added = 0
    dead_streak = 0
    dead_abort = None

    print(f"\n{'='*100}")
    print(f"裸問 48 題測試 — {label}   （兩把尺並排：old=legacy  new=strict）")
    print(f"Endpoint: {base_url} | Model: {model}")
    print(f"{'='*100}\n")

    for profile in profiles:
        profile_rules = {k: v for k, v in ref[profile].items() if k != "prompts"}
        prompt_keys = keys.get(profile, [])
        prompts = ref[profile]["prompts"]
        if len(prompt_keys) != len(prompts):
            print(f"  !! 答案鍵數量不符：{profile} keys={len(prompt_keys)} prompts={len(prompts)}"
                  f"（缺口以 novelty 規則處理）")
        if max_per_profile:
            prompts = prompts[:max_per_profile]

        n_prompts = len(prompts)
        prof_pass = {"legacy": 0, "core": 0, "strict": 0}
        profile_results = []

        for i, prompt in enumerate(prompts):
            resp = chat_completion(base_url, model, prompt)
            content = resp["content"]
            finish = resp["finish_reason"]
            elapsed = resp["elapsed"]
            key = prompt_keys[i] if i < len(prompt_keys) else None

            lg_ok, lg_checks = score_legacy(profile_rules, content, finish)
            core_ok, core_checks = score_strict(profile_rules, key, prompt, content, finish,
                                                apply_length_override=False)
            st_ok, st_checks = score_strict(profile_rules, key, prompt, content, finish,
                                            apply_length_override=True)

            # 變化歸因
            if lg_ok and not st_ok:
                if any(c.startswith("echo") for c in st_checks):
                    removed_echo += 1
                else:
                    removed_other += 1
            if st_ok and not lg_ok:
                added += 1

            for k, v in (("legacy", lg_ok), ("core", core_ok), ("strict", st_ok)):
                if v:
                    prof_pass[k] += 1

            flags = []
            if not content.strip():
                flags.append("EMPTY")
            if "```" in content and content.count("```") >= 4:
                flags.append("FENCE_LOOP")
            echoed, echo_why = detect_echo(prompt, content)
            if echoed:
                flags.append(f"ECHO({echo_why})")
            nv = novelty_chars(prompt, content)

            rec = {
                "idx": i,
                "prompt": prompt,
                "old": {"passed": lg_ok, "fail": lg_checks},
                "strict_core": {"passed": core_ok, "fail": core_checks},
                "new": {"passed": st_ok, "fail": st_checks},
                "finish_reason": finish,
                "elapsed_s": round(elapsed, 1),
                "output_len": len(content),
                "novel_chars": nv,
                "flags": flags,
                "output_preview": content[:120].replace("\n", " "),
            }
            if dump_full:
                # 沒有存全文，就沒有辦法在評分器改版後回頭重評——這正是本次任務
                # 無法用舊結果重跑雙尺的原因。新結果一律帶全文。
                rec["output_full"] = content
            profile_results.append(rec)

            verdict = "=" if lg_ok == st_ok else "!"
            if st_ok and not lg_ok:
                why = " (new-only)"
            elif lg_ok and not st_ok:
                why = f" [removed: {','.join(st_checks)}]"
            else:
                why = f" [{','.join(st_checks)}]" if st_checks else ""
            flag_str = f" {','.join(flags)}" if flags else ""
            print(f"  {verdict} [{profile}] #{i+1:2d} old={'PASS' if lg_ok else 'FAIL'}"
                  f" new={'PASS' if st_ok else 'FAIL'} {finish} {elapsed:.1f}s"
                  f" len={len(content)} nov={nv}{flag_str}{why}")
            if lg_ok != st_ok:
                print(f"      prompt: {prompt[:80]}")
                print(f"      output: {content[:110].replace(chr(10), ' ')}")

            # 連續「瞬間空回應」= 伺服器已死，後面的分數都是假的。
            if not content.strip() and elapsed < 1.0:
                dead_streak += 1
            else:
                dead_streak = 0
            if dead_streak >= DEAD_SERVER_STREAK:
                dead_abort = (
                    f"連續 {dead_streak} 題瞬間空回應（elapsed<1s）——伺服器已不在，"
                    f"中止於 {profile} #{i+1}；此報告的通過率無效"
                )
                print(f"\n!! {dead_abort}\n")
                break

        results[profile] = {
            "old_pass": prof_pass["legacy"],
            "core_pass": prof_pass["core"],
            "new_pass": prof_pass["strict"],
            "total": n_prompts,
            "old_rate": round(prof_pass["legacy"] / n_prompts, 3) if n_prompts else 0,
            "new_rate": round(prof_pass["strict"] / n_prompts, 3) if n_prompts else 0,
            "details": profile_results,
        }
        total_pass += prof_pass["legacy"]
        total_count += len(profile_results)
        for k in totals:
            totals[k] += prof_pass[k]
        print(f"\n  → {profile}: old {prof_pass['legacy']}/{n_prompts}  "
              f"new {prof_pass['strict']}/{n_prompts}  "
              f"(strict-core {prof_pass['core']}/{n_prompts})\n")

        if dead_abort:
            break

    print(f"\n{'='*100}")
    print(f"{'profile':<16} {'old (legacy)':<14} {'new (strict)':<14} {'strict-core':<14} {'delta'}")
    print("-" * 76)
    for p in results:
        r = results[p]
        print(f"{p:<16} {r['old_pass']}/{r['total']:<12} {r['new_pass']}/{r['total']:<12} "
              f"{r['core_pass']}/{r['total']:<12} {r['new_pass'] - r['old_pass']:+d}")
    print("-" * 76)
    print(f"{'TOTAL':<16} {totals['legacy']}/{total_count:<12} {totals['strict']}/{total_count:<12} "
          f"{totals['core']}/{total_count:<12} {totals['strict'] - totals['legacy']:+d}")
    print()
    print("歸因（old=PASS 但 new=FAIL 的假陽性被移除）:")
    print(f"  回顯閘門移除: {removed_echo}")
    print(f"  其他（關鍵字/長度/finish）移除: {removed_other}")
    print(f"  new-only 新增通過: {added}")
    print(f"  舊尺保留作為歷史比對；新尺才是判定「有回答」的那把。")
    print(f"{'='*100}\n")

    summary = {
        "label": label,
        "base_url": base_url,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "full_outputs_stored": dump_full,
        "scales": {
            "old": "legacy flat required_any (37/48-baseline ruler, unchanged)",
            "strict_core": "per-prompt keys + echo gate + novelty (no per-prompt length override)",
            "new": "strict_core + per-prompt length overrides for harness artifacts",
        },
        "old_total_pass": totals["legacy"],
        "strict_core_total_pass": totals["core"],
        "new_total_pass": totals["strict"],
        "total_count": total_count,
        "old_total_rate": round(totals["legacy"] / total_count, 3) if total_count else 0,
        "new_total_rate": round(totals["strict"] / total_count, 3) if total_count else 0,
        "false_positives_removed_echo": removed_echo,
        "false_positives_removed_other": removed_other,
        "new_only_passes": added,
        "profiles": results,
        "invalid": dead_abort is not None,
        "invalid_reason": dead_abort,
    }
    return summary


def selftest_echo():
    """機械證明：把每題的 prompt 當答案餵回去，新尺必須 0 PASS。

    這是「只回顯題目不再 PASS」的可驗證證據，不需要 server、不需要模型。
    同時列出舊尺在同樣輸入下會放行幾題（= 舊尺的假陽性）。
    """
    ref = load_reference()
    keys = load_answer_keys()
    variants = {
        "prompt": lambda p: p,
        "prompt 2x": lambda p: p + p,
        "prompt+nl 3x": lambda p: (p + "\n") * 3,
        "prompt+答：": lambda p: p + "\n答：",
    }
    n_total = 0
    n_old_pass = 0
    n_new_pass = 0
    failures = []

    print(f"\n{'='*100}")
    print("ECHO 自測：把每題 prompt 當答案餵回評分器（證明回顯不再 PASS）")
    print(f"{'='*100}\n")
    print(f"{'profile':<16} {'q':<4} {'variant':<14} {'old':<6} {'new':<6} fail-checks")
    print("-" * 100)

    for profile in ref["_profiles"]:
        rules = {k: v for k, v in ref[profile].items() if k != "prompts"}
        prompts = ref[profile]["prompts"]
        pk = keys.get(profile, [])
        for i, prompt in enumerate(prompts):
            key = pk[i] if i < len(pk) else None
            for vname, fn in variants.items():
                content = fn(prompt)
                n_total += 1
                old_ok, _ = score_legacy(rules, content, "stop")
                new_ok, new_checks = score_strict(rules, key, prompt, content, "stop")
                if old_ok:
                    n_old_pass += 1
                if new_ok:
                    n_new_pass += 1
                    failures.append((profile, i + 1, vname, new_checks))
                mark = "" if not new_ok else "  <== LEAK"
                print(f"{profile:<16} #{i+1:<3} {vname:<14} {'PASS' if old_ok else 'FAIL':<6} "
                      f"{'PASS' if new_ok else 'FAIL':<6} {','.join(new_checks)}{mark}")

    print("-" * 100)
    print(f"餵入樣本: {n_total}")
    print(f"舊尺放行: {n_old_pass}  ({n_old_pass/n_total*100:.1f}%)  <- 這些全是回顯假陽性")
    print(f"新尺放行: {n_new_pass}")
    if failures:
        print("\n!! 新尺仍有回顯洩漏:")
        for f in failures:
            print(f"   {f}")
        print()
        return 1
    print("\nPASS：新尺在 48 題 × 4 種回顯變體上全部 FAIL —— 「只回顯題目」不再得分。\n")
    print("=" * 100 + "\n")
    return 0


def compare_results(file_a, file_b):
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    print(f"\n{'='*70}")
    print(f"對比: {a['label']} vs {b['label']}")
    print(f"{'='*70}\n")
    print(f"{'Profile':<16} {a['label']:<12} {b['label']:<12} {'差距':<8}")
    print("-" * 50)

    for profile in a["profiles"]:
        pa = a["profiles"][profile]
        pb = b["profiles"][profile]
        diff = pb["new_pass"] - pa["new_pass"]
        sign = "+" if diff > 0 else ""
        print(f"{profile:<16} {pa['new_pass']}/{pa['total']:<8} {pb['new_pass']}/{pb['total']:<8} {sign}{diff}")

    print("-" * 50)
    print(f"{'總計':<16} {a['new_total_pass']}/{a['total_count']:<8} "
          f"{b['new_total_pass']}/{b['total_count']:<8} {b['new_total_pass']-a['new_total_pass']:+d}")
    print()


def main():
    parser = argparse.ArgumentParser(description="48 題裸問品質測試（雙尺：legacy vs strict）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="API base URL")
    parser.add_argument("--model", default="qwen36", help="model name")
    parser.add_argument("--label", default="local", help="標籤（用於輸出）")
    parser.add_argument("--output", default=None, help="結果輸出 JSON 路徑")
    parser.add_argument("--max-per-profile", type=int, default=None, help="每 profile 最多題數（除錯用）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="取樣溫度（預設 0.0 = greedy，與歷史行為逐位元相同）；"
                             "設 0.4 才能對照 37/48 基線配方")
    parser.add_argument("--top-p", type=float, default=None,
                        help="top-p（預設不送；37/48 基線用 0.8）")
    parser.add_argument("--dump-full", action="store_true",
                        help="把每題完整輸出寫進結果 JSON。沒有全文就無法在評分器改版後回頭重評，建議開。")
    parser.add_argument("--selftest-echo", action="store_true",
                        help="不需 server：把每題 prompt 當答案餵回兩把尺，斷言新尺 0 PASS")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"), help="對比兩個結果檔")
    args = parser.parse_args()

    global TEMPERATURE, TOP_P
    TEMPERATURE, TOP_P = args.temperature, args.top_p

    if args.selftest_echo:
        sys.exit(selftest_echo())

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    summary = run_suite(args.base_url, args.model, args.label, args.max_per_profile,
                        dump_full=args.dump_full)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存: {args.output}")


if __name__ == "__main__":
    main()
