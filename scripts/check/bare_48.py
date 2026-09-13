#!/usr/bin/env python3
"""
48 題裸問品質測試 — 純 user message、無拐杖（no prefill / pp / stop）、temp=0。

用法：
  python3 scripts/check/bare_48.py --base-url http://127.0.0.1:8080/v1 --model qwen36 --label IQ3_XXS
  python3 scripts/check/bare_48.py --base-url http://127.0.0.1:8080/v1 --model qwen36 --label IQ4_XS --output /tmp/bare_iq4xs.json

對比兩次結果：
  python3 scripts/check/bare_48.py --compare /tmp/bare_iq3xxs.json /tmp/bare_iq4xs.json

評分 = 與 37/48 基線相同（bare_48_local2 all-pass 規則）：每個 reference
rule 都要通過（length / required_any / required_all / forbidden / code_block /
loop_phrase），空輸出因 length_min>0 必然 fail。早期版本誤用
replay_server_profile.evaluate_quality 且傳錯參數 → 空輸出也拿 1.0/0.60
（no_reference 分支），所有早期 bare_48 分數因此失真。
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

# 死伺服器防護。伺服器一旦不在，/v1/chat/completions 會「瞬間」回空——
# 之後每一題都會被計為 0 分，看起來像品質崩壞，其實只是沒有人在聽。
# 2026-09-13 的 cap6 對照就是這樣：伺服器在第 26 個請求後收到 signal 結束，
# 剩下的 22 題全部瞬間失敗，而報告只寫「0/48」。
DEAD_SERVER_STREAK = 5


def load_reference():
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
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


def score_one(profile, rules, content, finish):
    """All-pass 評分：所有 rule 通過才算 PASS（與 37/48 基線 bare_48_local2 相同）。
    回傳 (passed: bool, fail_checks: list[str])。"""
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
        n_open = content.count("```")
        if n_open > rules.get("code_block_max_repeat", 2) * 2:
            checks.append("code_block")
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


def run_suite(base_url, model, label, max_per_profile=None):
    ref = load_reference()
    profiles = ref["_profiles"]
    results = {}
    total_pass = 0
    total_count = 0
    dead_streak = 0
    dead_abort = None

    print(f"\n{'='*70}")
    print(f"裸問 48 題測試 — {label}")
    print(f"Endpoint: {base_url} | Model: {model}")
    print(f"{'='*70}\n")

    for profile in profiles:
        rules = ref[profile]
        prompts = rules["prompts"]
        if max_per_profile:
            prompts = prompts[:max_per_profile]

        profile_pass = 0
        profile_results = []

        for i, prompt in enumerate(prompts):
            resp = chat_completion(base_url, model, prompt)
            content = resp["content"]
            finish = resp["finish_reason"]
            elapsed = resp["elapsed"]

            # 评分：all-pass 規則（與 37/48 基線 bare_48_local2 相同）
            rules = {k: v for k, v in ref[profile].items() if k != "prompts"}
            passed, fail_checks = score_one(profile, rules, content, finish)
            score = 1.0 if passed else 0.0

            # 额外检测：空输出 / echo / fence loop 标记
            flags = []
            if not content.strip():
                flags.append("EMPTY")
            if content.strip().startswith(prompt[:20]) if len(prompt) > 20 else False:
                flags.append("ECHO")
            if "```" in content and content.count("```") >= 4:
                flags.append("FENCE_LOOP")

            profile_results.append({
                "idx": i,
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "score": round(score, 3),
                "passed": passed,
                "finish_reason": finish,
                "elapsed_s": round(elapsed, 1),
                "output_len": len(content),
                "flags": flags,
                "output_preview": content[:120].replace("\n", " "),
            })

            status = "PASS" if passed else "FAIL"
            flag_str = f" [{','.join(flags)}]" if flags else ""
            why = f" [{','.join(fail_checks)}]" if fail_checks else ""
            print(f"  [{profile}] #{i+1:2d} {status} score={score:.2f} {finish} {elapsed:.1f}s len={len(content)}{flag_str}{why}")
            if not passed:
                print(f"         prompt: {prompt[:80]}")
                print(f"         output: {content[:100].replace(chr(10), ' ')}")

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

            if passed:
                profile_pass += 1
            total_count += 1

        total_pass += profile_pass
        results[profile] = {
            "pass": profile_pass,
            "total": len(prompts),
            "rate": round(profile_pass / len(prompts), 3) if prompts else 0,
            "details": profile_results,
        }
        print(f"\n  → {profile}: {profile_pass}/{len(prompts)} ({profile_pass/len(prompts)*100:.0f}%)\n")

        if dead_abort:
            break

    summary = {
        "label": label,
        "base_url": base_url,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pass": total_pass,
        "total_count": total_count,
        "total_rate": round(total_pass / total_count, 3) if total_count else 0,
        "profiles": results,
        "invalid": dead_abort is not None,
        "invalid_reason": dead_abort,
    }

    print(f"\n{'='*70}")
    if dead_abort:
        print(f"總計: {total_pass}/{total_count} (無效 — {dead_abort})")
    else:
        print(f"總計: {total_pass}/{total_count} ({total_pass/total_count*100:.1f}%)")
    print(f"{'='*70}\n")

    return summary


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
        diff = pb["pass"] - pa["pass"]
        sign = "+" if diff > 0 else ""
        print(f"{profile:<16} {pa['pass']}/{pa['total']:<8} {pb['pass']}/{pb['total']:<8} {sign}{diff}")

    print("-" * 50)
    print(f"{'總計':<16} {a['total_pass']}/{a['total_count']:<8} {b['total_pass']}/{b['total_count']:<8} +{b['total_pass']-a['total_pass']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="48 題裸問品質測試")
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
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"), help="對比兩個結果檔")
    args = parser.parse_args()

    global TEMPERATURE, TOP_P
    TEMPERATURE, TOP_P = args.temperature, args.top_p

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    summary = run_suite(args.base_url, args.model, args.label, args.max_per_profile)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存: {args.output}")


if __name__ == "__main__":
    main()
