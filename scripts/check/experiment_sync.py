#!/usr/bin/env python3
"""experiment_sync — 把「跑前立項 charter → mindmap 實驗節點 → harness bench 產物 → 節點更新」自動化。

為什麼需要它
    兩條開發線各自跑實驗，常見三個問題：實驗沒在 mindmap 上掛節點（看不到進展）、繞過生產級
    腳本自拼參數（口徑不一致）、跑完結果不回寫（子目標永遠卡在 doing）。本工具把流程釘死成：

        ① 寫一張 charter（scripts/check/charters/_TEMPLATE.yaml：現狀/目標/假設/驗收）
        ② python3 experiment_sync.py init --charter <x.yaml> --sub S
               → 在 mindmap 建一個 tier=3a 的「實驗階段」節點（含初始子目標）
        ③ 用生產級腳本跑：harness.py bench --charter <x.yaml> --json Backup/.../x.json
        ④ python3 experiment_sync.py sync --artifact Backup/.../x.json
               → 把運行設置／可點 log／結果／thermal／swap 追加進節點，更新結果與子目標

    init／sync 完成後都會自動重建 mindmap 總圖（index.html）＋ 全部逐條白皮書 briefs（html/md）。

用法
    python3 experiment_sync.py init  --charter scripts/check/charters/x.yaml --sub S
    python3 experiment_sync.py sync  --artifact Backup/.../x.json [--artifact ...]
    python3 experiment_sync.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mindmap_build as MB          # noqa: E402  (總圖構建／校驗)
import mindmap_brief_build as MBB   # noqa: E402  (逐條白皮書)

MM_JSON = ROOT / "docs/mindmap/mindmap.json"
EXP_BACKUP = ROOT / "Backup/exp_runs"
SUBS = ("S", "M", "both", "C", "na")


# ───────────────────────── 純邏輯（不碰檔案，selftest 測這一層）─────

def resolve_sub(charter: dict, sub: str | None) -> str:
    """實驗節點要掛在哪個軸：--sub 優先，其次 charter.axis；都沒有就 fail-closed。"""
    s = sub or charter.get("axis")
    if s not in SUBS:
        raise ValueError(f"必須指定軸 --sub（{list(SUBS)}），或在 charter 加 axis 欄位")
    return s


def short_name(question: str, fallback: str, n: int = 34) -> str:
    q = " ".join(str(question).split())
    if not q:
        return fallback
    return q[:n] + "…" if len(q) > n else q


def build_entry(charter: dict, sub: str, name: str | None = None) -> dict:
    """charter → tier=3a 實驗節點（純 dict）。"""
    cid = str(charter.get("id", "")).strip()
    if not cid:
        raise ValueError("charter 缺 id")
    base = charter.get("baseline") or {}
    hyp = charter.get("hypothesis") or {}
    acc = charter.get("acceptance") or {}
    raw_targets = charter.get("targets")
    targets = {k: v for k, v in (raw_targets.items() if isinstance(raw_targets, dict) else [])
               if k in ("decode_tps", "prefill_tps") and isinstance(v, (int, float))}
    charter_rel = None
    cp = charter.get("_path")
    if cp and _is_relative_to(Path(cp), ROOT):
        charter_rel = os.path.relpath(cp, ROOT)
    subtasks = [
        {"id": "st-charter", "text": "跑前立項（現狀/目標/假設/驗收）", "status": "done",
         **({"log": charter_rel} if charter_rel else {})},
        {"id": "st-run", "text": "用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log",
         "status": "doing"},
        {"id": "st-accept", "text": f'驗收：{str(acc.get("success", "—"))}', "status": "todo"},
        {"id": "st-falsify", "text": f'否證條件：{str(acc.get("falsify", "—"))}', "status": "todo"},
    ]
    for sg in charter.get("subgoals") or []:
        sid = str(sg.get("id", "")).strip()
        if not sid:
            continue
        subtasks.append({
            "id": sid, "text": str(sg.get("text", sid)), "status": "todo",
            "expect": sg.get("expect"), "how": sg.get("how"),
            "contrib": None, "contribs": [],
        })
    return {
        "id": cid,
        "name": name or short_name(charter.get("question", ""), cid),
        "theme": "實驗",
        "tier": "3a",
        "goal": str(charter.get("question", "")).strip(),
        "crit": str(acc.get("success", "—")),
        "res": RES_PLACEHOLDER,
        "evid": str(base.get("source", "—")),
        "sub": sub,
        "note": f'假設：{str(hyp.get("mechanism", "—"))}',
        "runs": [],
        "targets": targets,
        "best": None,
        "target_gap": target_gap(targets, None),
        "subtasks": subtasks,
    }


def charter_id(arm: dict) -> str | None:
    """從產物 arm 取出它隸屬的 charter id（無立項／豁免 → None）。"""
    ch = arm.get("charter")
    if isinstance(ch, dict):
        return None if ch.get("waived") else ch.get("id")
    if isinstance(ch, str) and ch not in ("waived", "none", ""):
        return ch
    return None


def row_ts(rows: list, kind: str):
    """從 rows 抓 prefill（pp，n_prompt>0）或 decode（tg，n_gen>0 且無 prompt）的 avg_ts。"""
    for r in rows:
        if kind == "pp" and r.get("n_prompt", 0) > 0:
            return r.get("avg_ts")
        if kind == "tg" and r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) > 0:
            return r.get("avg_ts")
    return None


def _mb(v):
    return f"{v:.0f} MiB" if isinstance(v, (int, float)) else "?"


def _clean_attribution(arm) -> tuple:
    """(clean, why) —— 判準從 `memory_pressure.is_clean()` 來，**不在这里重寫一遍**。

    讀不到那支工具就 fail-closed（不乾淨）：證明不了不是乾淨。

    [CGC 2026-10-03 線A · 熱口徑] 參數是**臂**（不是 `arm["attribution"]`）：熱條件要看
    **量測窗**（`arm["thermal_windows"]`），而那是**臂層級**的欄位。依據就是本檔自己那句
    「本 repo 的驗收寫的是**量測期** NOMINAL」（見 `_arm_on_board` 的註解）——
    意的本來就是計時段，只是 `is_clean` 以前只拿得到整臂取樣的 worst。
    ⚠ 舊產物沒有 `thermal_windows` ⇒ 行為一字不變。
    """
    att = arm.get("attribution") if isinstance(arm, dict) else None
    try:
        import memory_pressure as _mp
        wins = arm.get("thermal_windows") if isinstance(arm, dict) else None
        return _mp.is_clean(att, thermal_windows=wins)
    except Exception as e:  # noqa: BLE001  判不了就不准上成績面
        return False, f"無法判定歸因（{e}）"


class SyncRefused(Exception):
    """第三扇門擋下了一次同步。帶 `code`（4 = 量具 UNBOUND、5 = 缺成對 log）與 `gate` 判定。

    用例外而不是回空 list：回空 list 會跟「這支產物沒有 charter」那個正常的空結果混在一起，
    而兩者的下一步完全不同（一個要修量具／補 log，一個要加 --entry）。
    """

    def __init__(self, code: int, gate: dict):
        self.code = code
        self.gate = gate
        super().__init__(gate.get("why") or "同步被量具閘擋下")


def _spec_draft_dead(arm) -> bool:
    """[CGC 2026-10-06] spec draft 死亡的判準——唯一出處在產物端（llama_bench_matrix）。
    匯入失敗**不放行**（fail-closed）：判不了不等於健康。"""
    import llama_bench_matrix as _lbm
    return _lbm.spec_draft_dead(arm)


def sync_gate(path, waive_reason=None) -> dict:
    """第三扇門：**同步本身**也要過量具閘與成對閘。

    WHY 要這一扇：`harness bench` 的 rc=4／rc=5 只擋住「經由 harness 的那條路」—— 人（或任何腳本）
    直接 `experiment_sync sync <artifact>` 時，那道閘根本不會被執行，而 sync 就是寫進決策面那一步。
    把閘放在 `sync_artifact_file` 裡（手動 sync 與 harness 的 auto-sync 都必經它），
    兩條路就不再可能一個有一個沒有。

    回 `{ok, code, pair, arms[], why, waived}`。

    兩個半邊的策略差別是刻意的，與 harness 兩扇門一致：
    * **成對**不可豁免：沒有 log 時，你不知道自己錯過了什麼。
    * **spec draft 死亡**（逐 rep `drafted=0`）不可豁免：那一輪量的是純解碼，不是 spec；
      它的 t/s 與健康的 tg 讀數長得一樣，留下來只會被誤引用（2026-10-06 `p_chain`）。
    * **UNBOUND 可豁免**（理由記進產物）：那是「我知道這輪儀器壞了，還是要留下這一筆」。
      豁免只讓它進得來，不讓它變乾淨 —— `merge_run`／`mindmap_void_check` 照樣把它標成不可引用。
    """
    p = str(path)
    pair = _pair_of(p)
    arms, why = [], None
    try:
        import instrument_binding as _ib
    except Exception as e:  # noqa: BLE001  判不了就只判成對那一半（不静默放行）
        return {"ok": bool(pair.get("paired", True)), "code": 0, "pair": pair, "arms": [],
                "why": f"（instrument_binding 不可用：{e}）" if pair.get("paired", True) else None,
                "waived": waive_reason}
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "code": 0, "pair": pair, "arms": [],
                "why": f"產物讀不出來（{e}）—— 交給後面的跳過邏輯", "waived": waive_reason}
    arm_list = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    for arm in arm_list:
        if not isinstance(arm, dict) or "rows" not in arm:
            continue
        rec = _instrument_of(arm, p)
        if not isinstance(rec, dict):
            continue
        ok_i, why_i = _ib.is_bound(rec)
        arms.append({"tag": arm.get("tag"), "ok": bool(ok_i),
                     "verdict": rec.get("verdict"), "why": why_i})
    unbound = [a for a in arms if not a["ok"]]
    if pair and not pair.get("paired", True):
        return {"ok": False, "code": 5, "pair": pair, "arms": arms, "waived": waive_reason,
                "why": pair.get("why") or "缺成對的 stderr log"}
    # 【CGC 2026-10-06】第六道閘門延伸到 sync：spec 臂的 draft 死過（逐 rep drafted=0）⇒ 拒寫。
    # 與成對閘同類：**沒有豁免** —— draft 死亡的那一輪量的是純解碼（emit ≡ 1.0），
    # 它的 t/s 與健康的 tg 讀數長得一樣，但量的不是 spec 路徑。`harness bench` 的 rc=6 只擋住
    # 經由 harness 的那條路；手動 `sync` 不經過它，而 sync 才是寫進決策面那一步。
    _dead = []
    for arm in arm_list:
        if not isinstance(arm, dict):
            continue
        # [CGC 2026-10-06] 與 harness／排行榜共用**同一個**判準（fail-closed）。先前這裡只擋
        # 字面上的 "INVALID" ⇒ NO_STATE／打錯字會穿過 sync 直接寫進決策面。
        # ⚠ 刻意**不**採用產物裡的 `spec_draft_gate`：那是 harness 依同一條判準**算出來後寫下**的
        # 記錄（output），拿它當 input 會讓「舊記錄較嚴」的臂永遠翻不了身，而判準的唯一出處是
        # `spec_draft.state`。三個消費端因此逐格相同。
        if _spec_draft_dead(arm):
            _dead.append(arm)
    if _dead:
        return {"ok": False, "code": 6, "pair": pair, "arms": arms, "waived": waive_reason,
                "why": "spec draft 死亡（drafted=0 ⇒ 那一輪量的是純解碼，不是 spec）：%s"
                       % "；".join("%s（%s）" % (a.get("tag"), ((a.get("spec_draft") or {}).get("why")
                                                                 or "見 spec_draft_gate"))
                                   for a in _dead[:3])}
    if unbound and not waive_reason:
        return {"ok": False, "code": 4, "pair": pair, "arms": arms, "waived": None,
                "why": "量具 UNBOUND：%s" % "；".join("%s（%s）" % (a["tag"], a["why"])
                                                    for a in unbound[:3])}
    return {"ok": True, "code": 0, "pair": pair, "arms": arms, "waived": waive_reason,
            "why": "" if not unbound else "已豁免：%d 臂量具 UNBOUND" % len(unbound)}


def _instrument_of(arm: dict, log_rel: str):
    """這一輪的**量具綁定**判定（見 `instrument_binding.py`）。

    新產物（`llama_bench_matrix` 2026-09-29 之後）自己記了 `instrument`，直接用它；舊產物則由臂的
    env 推「對誰許過承諾」，再讀**並存**的 stderr log（`X.json` ↔ `X.stderr.log`）算。

    讀不到 log ⇒ `UNVERIFIABLE`（沒有證據 ≠ 證據說沒綁）。判不了（工具不在）⇒ 回 None，
    由 `_instrument_ok` 當「無此欄位」處理 —— 不會因此把一輪汙成 VOID。
    """
    rec = arm.get("instrument")
    if isinstance(rec, dict) and rec.get("verdict"):
        return rec
    try:
        import instrument_binding as _ib
    except Exception:  # noqa: BLE001  判不了就不記（缺席不是「未綁上」）
        return None
    return _ib.report_for_artifact(arm, artifact_path=(str(log_rel) or None), repo_root=ROOT)


def _pair_of(log_rel) -> dict:
    """跑出來的產物 ↔ stderr log 的成對判定（判準在 `instrument_binding.pair_status`，不重寫）。"""
    if not log_rel or not str(log_rel).endswith(".json"):
        return {}
    p = str(log_rel) if os.path.isabs(str(log_rel)) else os.path.join(ROOT, str(log_rel))
    if not os.path.exists(p):
        return {}
    try:
        import instrument_binding as _ib
        is_run, _d, _w = _ib.load_run_artifact(p)
        if not is_run:
            return {}          # 不是在跑的產物（文件類）⇒ 成對不適用
        return _ib.pair_status(p)
    except Exception:  # noqa: BLE001  判不了就不記（缺席不是「沒成對」）
        return {}


def _pair_ok(run: dict) -> tuple:
    """(ok, why) —— 這一輪的產物旁邊有沒有它的 stderr log。

    缺 ⇒ **不乾淨（不可引用）**：「沒有 log 所以驗不了」不該是一種狀態。這條對**舊 run 也成立**
    （`run["log"]` 在舊產物上也有），所以它不是「政策不回溯」那一類的：回溯的那條講的是
    「欄位還不存在時不要偷改判定」，而這條問的是「檔案在不在」，那個問題在任何年代都有答案。
    """
    pair = run.get("pair")
    if not isinstance(pair, dict) or not pair:
        pair = _pair_of(run.get("log"))       # 舊 run：當場算
    if not pair:
        return True, ""
    if pair.get("paired"):
        return True, ""
    return False, "缺成對的 stderr log（不可引用）：%s" % (pair.get("why") or "")


def _instrument_ok(run: dict) -> tuple:
    """(ok, why)。`BOUND`／`N/A` 通過；`UNBOUND` 不通過；
    `UNVERIFIABLE`（產物旁邊沒有並存的 stderr log）**預設只記不翻**，
    要 fail-closed 就設 `CGC_INSTRUMENT_STRICT=1`（判準與 strict 開關都在 `instrument_binding.py`）。

    沒有 `instrument` 欄位的舊 run 一律回 True，這是刻意的：它不代表那一輪綁上了，代表「這個欄位
    還不存在時的政策不該回頭改寫整份語料」。要把它清出來的人用 `experiment_sync instrument-audit`
    （唯讀）—— 看得見，而不是静默地翻掉。
    """
    inst = run.get("instrument")
    if not isinstance(inst, dict):
        return True, ""
    try:
        import instrument_binding as _ib
    except Exception as e:  # noqa: BLE001
        return True, f"（量具判定不可用：{e}）"
    v = str(inst.get("verdict") or "")
    if v in _ib.BOUND_VERDICTS:
        return True, ""
    if v == _ib.UNVERIFIABLE and not _ib.strict_mode():
        return True, "（量具未驗：%s）" % (inst.get("why") or "")
    return False, "量具未綁上：%s" % (inst.get("why") or v or "未記錄")


def _run_clean(run: dict) -> tuple:
    """(clean, why)。新產物看 `run['clean']`；舊產物沒有那個欄位時，由 `run['verdict']` 的字首判。

    兩條路都要再過一次**量具綁定**（`_instrument_ok`）：歸因乾淨說的是「這一輪的 t/s 沒被熱／swap
    汙染」，量具綁定說的是「這一輪的計數器真的有在動」。
    2026-09-29 的教訓：两者可以一個真一個假 —— 一個 swap 乾淨、但 MM 回讀全部印 0 的局，
    它的 misses/nsel 一樣不可引用。
    """
    whys = []
    iok, iwhy = _instrument_ok(run)
    if not iok:
        whys.append(iwhy)
    # 成對（2026-09-29）：歸因乾淨與量具綁上都成立，還得看「那一場的 log 還在不在」——
    # 沒有 log，前兩者的判定根本無從檢查。
    pok, pwhy = _pair_ok(run)
    if not pok:
        whys.append(pwhy)
    if "clean" in run:
        cl = bool(run["clean"])
        if not cl:
            whys.insert(0, str(run.get("void_why") or ""))
    else:
        verdict = str(run.get("verdict") or "")
        cl = verdict.split("：", 1)[0].strip() == "none"
        if not cl:
            whys.insert(0, verdict or "未記錄歸因")
    return (cl and iok and pok), "；".join(w for w in whys if w)


def build_run(arm: dict, log_rel: str, when: str) -> dict:
    """產物 arm → 一條 run（運行設置＋log＋結果＋thermal＋swap）。"""
    rows = arm.get("rows") or []
    sb, sa = arm.get("sys_before") or {}, arm.get("sys_after") or {}
    pp, tg = row_ts(rows, "pp"), row_ts(rows, "tg")
    th = arm.get("thermal") or {}
    att = arm.get("attribution") or {}
    thermal = ((sa.get("thermal") or {}).get("label")) or th.get("worst") or "?"
    r0 = rows[0] if rows else {}
    pp_row = next((r for r in rows if r.get("n_prompt", 0) > 0), None)
    tg_row = next((r for r in rows if r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) > 0), None)
    n_prompt = (pp_row or r0).get("n_prompt", "?")
    n_gen = tg_row.get("n_gen") if tg_row else r0.get("n_gen", "?")
    cmd_desc = (f'llama-bench（{arm.get("profile", "?")}）-p {n_prompt} '
                f'-n {n_gen} --warm-skip {arm.get("warm_skip", "?")}')
    inst = _instrument_of(arm, log_rel)
    _c_ok, c_why = _clean_attribution(arm)
    _i_ok, i_why = _instrument_ok({"instrument": inst})
    result = {}
    if pp is not None:
        result["pp"] = round(pp, 2)
    if tg is not None:
        result["tg"] = round(tg, 2)
    return {
        "when": when,
        "arm": str(arm.get("tag") or arm.get("profile") or "?"),
        "profile": arm.get("profile"),
        "log": log_rel,
        # 成對（2026-09-29）：寫入時就定下來，這樣「那一場的 log 還在不在」不必等到引用時才發現。
        # 只留非數值欄位：`server_window.MEASURED_KEYS` 把帶 `arms` 的 dict 當「量到的東西」，
        # 而這個區塊是**關於檔案**的，不是量到的數值 —— 混進去會讓 provenance_gate 多一個 offender。
        "pair": {k: v for k, v in (_pair_of(log_rel) or {}).items()
                 if k in ("paired", "source", "canonical", "why")},
        "cmd": cmd_desc + "（由產物參數重建）",
        "result": result,
        "thermal": thermal,
        "swap": {"before": _mb(sb.get("swap_used_mb")), "after": _mb(sa.get("swap_used_mb"))},
        "verdict": f'{att.get("verdict", "?")}：{att.get("why", "")}'.rstrip("："),
        # 2026-09-28：歸因是不是乾淨，要在**寫入時**就定下來（見 _recompute_result 的 docstring）。
        # 2026-09-29：再加上**量具綁定**與**成對 log** —— 乾淨的歸因不足以保證計數器真的有在動，
        # 而沒有 log 時根本無從檢查那兩件事。
        "clean": _c_ok and _i_ok,
        "void_why": "；".join(x for x in (c_why, i_why) if x),
        "instrument": inst,
        "subgoal_contrib": arm.get("subgoal_contrib"),
    }


def merge_run(entry: dict, run: dict) -> bool:
    """把 run 併進 entry（**同 log ＋ 同臂**去重）；把 st-run 標 done，並重算成績面。回是否新增。

    成績面（`res`／`best`／`target_gap`）**只有乾淨的 run 能影響**，見 `_recompute_result`。

    [CGC 2026-10-03] 去重鍵加上 `arm`。`sync_artifact_file` 是**逐臂**呼叫本函式的（意圖就是每臂
    各記一條），但一個**多臂產物**（例如 off／on 配對的 `harness bench --arm A --arm B`）的每一臂
    都指向**同一份產物路徑**（`ensure_log_in_repo` 對 repo 內的產物直接回它的相對路徑）
    ⇒ 只用 `log` 去重會**靜默吃掉第二臂之後的每一臂**，而那正好是配對實驗的對照臂。
    加上 `arm` 之後：同一產物重複 sync 仍冪等（同 log ＋ 同臂 ⇒ 去重），不同臂各記一條。
    """
    entry.setdefault("runs", [])
    if any(r.get("log") == run["log"] and r.get("arm") == run.get("arm")
           for r in entry["runs"]):
        # 去重的那一條也要重算：否則一個在本規則之前就寫壞的節點，再 sync 幾次都不會自己好。
        _recompute_result(entry)
        return False
    entry["runs"].append(run)
    for t in entry.setdefault("subtasks", []):
        if t.get("id") == "st-run" and run.get("result"):
            t["status"] = "done"
    for sid, val in (run.get("subgoal_contrib") or {}).items():
        st = next((t for t in entry["subtasks"] if t.get("id") == sid), None)
        if st is None:
            continue
        st.setdefault("contribs", []).append(val)
        nums = [x for x in st["contribs"] if isinstance(x, (int, float))]
        st["contrib"] = round(sum(nums), 4) if nums else st["contribs"][-1]
        st["status"] = "doing"
    _recompute_result(entry)
    return True


def compute_best(runs: list) -> dict | None:
    """從**乾淨**的 runs 挑成績最高者：以 decode(tg) 為第一指標；無任何 tg 才退而以 prefill(pp) 為準。

    「乾淨」= `memory_pressure.is_clean()`（唯一判準，`mindmap_void_check.py` 用同一個）。
    build_run 已把 clean 標在 run 上，但**舊產物**沒有那個欄位 ⇒ `_run_clean` 兩條路都判。
    """
    runs = [r for r in runs if isinstance(r, dict) and _run_clean(r)[0]]
    def tg_of(r):
        return (r.get("result") or {}).get("tg")

    def pp_of(r):
        return (r.get("result") or {}).get("pp")

    tg_cands = [r for r in runs if isinstance(tg_of(r), (int, float))]
    if tg_cands:
        r = max(tg_cands, key=tg_of)
        return {"metric": "tg", "tg": tg_of(r), "pp": pp_of(r),
                "arm": r.get("arm"), "log": r.get("log"), "when": r.get("when")}
    pp_cands = [r for r in runs if isinstance(pp_of(r), (int, float))]
    if pp_cands:
        r = max(pp_cands, key=pp_of)
        return {"metric": "pp", "tg": None, "pp": pp_of(r),
                "arm": r.get("arm"), "log": r.get("log"), "when": r.get("when")}
    return None


TS_IN_RES = re.compile(r"\d+(?:\.\d+)?\s*t/s")

# build_entry 一開始放的那一句。機器寫的預設值可以被機器更新，人寫的判詞不行 —— 差別就在這裡。
RES_PLACEHOLDER = "實驗進行中（尚無讀數）"


def _res_claims_throughput(entry: dict) -> bool:
    """現在的 `res` 是不是一個**吞吐主張**（會被抓去做節點成績、也會被 brief 放大）。"""
    return bool(TS_IN_RES.search(str(entry.get("res") or "")))


def _machine_owns_res(entry: dict) -> bool:
    """機器可以改寫 `res` 嗎：它現在是一個吞吐主張，或者還是 build_entry 放的預設句。"""
    res = str(entry.get("res") or "")
    return _res_claims_throughput(entry) or res.strip() in ("", RES_PLACEHOLDER)


def _recompute_result(entry: dict) -> None:
    """把 `res`／`best`／`target_gap` 從**整個** run set 重算（idempotent）。

    WHY THIS EXISTS（2026-09-28）：一次實驗跑完後，一個 **swap 歸因的 9.94 t/s** 變成節點的
    `res`，並讓 `target_gap` 算出「距 decode 20 目標 49.7%」—— 一個依 §5.1 不得引用的吞吐
    兩次上了決策面（`docs/PREMISE_B_CHURN_DELIVERY_2026-09-28.md` §8）。抓它的是
    `scripts/check/mindmap_void_check.py`；但檢查器是**事後**的，所以判準也在這裡擋一次：

      * 只有乾淨的 run 可以把吞吐放上 `res`／`best`；
      * 一個乾淨的都沒有時，`res` 必須說「尚無可引用讀數」並帶 `no_throughput_claim`（含為什麼），
        而不是放一個 VOID 數字；
      * 後來有乾淨的 run 進來 ⇒ `res` 恢復成讀數，並把標記拿掉。

    重算而不是「只在新增時更新」，是為了讓它自我修復：一個在本次修正之前寫壞的節點，
    下次 sync 就會收斂回誠實狀態。
    """
    runs = [r for r in (entry.get("runs") or []) if isinstance(r, dict)]
    entry["best"] = compute_best(runs)
    entry["target_gap"] = target_gap(entry.get("targets") or {}, entry["best"])
    best = entry["best"]
    if best:
        if best.get("metric") == "pp":
            entry["res"] = f"prefill {best['pp']:.2f} t/s"
        else:
            entry["res"] = f"decode {best['tg']:.2f} t/s"
        entry.pop("no_throughput_claim", None)
        return
    if not runs:
        return  # 還沒有 run：res 留給 build_entry 的初始值
    void = sorted({(_run_clean(r)[1] or "?").split("：", 1)[0] for r in runs if not _run_clean(r)[0]})
    if _machine_owns_res(entry):
        # 只在 res 是**吞吐主張**或**機器的預設句**時才改寫：那個 VOID 數字就是這條規則要拿掉的東西。
        # 反過來，人寫的判詞（例如「主 context ntok=1 的 churn 非 0」）不是吞吐主張，不動它 ——
        # 用機器句子蓋掉人寫的判詞也是一種資訊損失。
        entry["res"] = (f"尚無可引用讀數：{len(runs)} 個 run 全部非乾淨（{'、'.join(void)}）")
    entry["no_throughput_claim"] = (
        f"本節點 {len(runs)} 個 run 全部非乾淨歸因（{'、'.join(void)}）⇒ 依 MEASUREMENT_CONTRACT §5.1 "
        "吞吐不得引用；節點只記判詞。乾淨 = memory_pressure.is_clean()。")


def target_gap(targets: dict, best: dict | None) -> dict:
    """量化目標 vs 當前最佳：回每個目標的 current/gap/進度%（tg→decode_tps、pp→prefill_tps）。"""
    cur_map = {"decode_tps": (best or {}).get("tg"), "prefill_tps": (best or {}).get("pp")}
    out: dict = {}
    for key in ("decode_tps", "prefill_tps"):
        tgt = targets.get(key) if isinstance(targets, dict) else None
        if not isinstance(tgt, (int, float)):
            continue
        cur = cur_map[key]
        if isinstance(cur, (int, float)):
            out[key] = {"current": round(cur, 3), "target": tgt,
                        "gap": round(cur - tgt, 3), "pct": round(cur / tgt * 100, 1)}
        else:
            out[key] = {"current": None, "target": tgt, "gap": None, "pct": None}
    return out


# ───────────────────────── 檔案 IO ────────────────────────────────

def _is_relative_to(p: Path, base: Path) -> bool:
    try:
        p.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


def load_charter(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"charter 不存在: {path}（相對於 repo root 解析）")
    try:
        import yaml
    except ImportError:
        raise SystemExit("需要 PyYAML（pip install pyyaml）")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("charter 頂層必須是 mapping")
    data["_path"] = str(p)
    return data


def load_mm() -> dict:
    return json.loads(MM_JSON.read_text(encoding="utf-8"))


def save_mm(data: dict) -> None:
    MM_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_log_in_repo(artifact: Path, cid: str, stamp: str) -> str:
    """產物若在 repo 內，回其相對路徑；否則複製進 Backup/exp_runs/（保證 brief 的連結點得到）。"""
    if _is_relative_to(artifact, ROOT):
        return os.path.relpath(artifact.resolve(), ROOT)
    EXP_BACKUP.mkdir(parents=True, exist_ok=True)
    dest = EXP_BACKUP / f"{cid}_{stamp}.json"
    shutil.copy2(artifact, dest)
    return str(dest.relative_to(ROOT))


def rebuild() -> None:
    """重建 mindmap 總圖（權威流程 MB.main）＋ 全部逐條白皮書 briefs。"""
    if MB.main([]) != 0:
        raise SystemExit("mindmap 總圖構建失敗（見上）")
    data, mapping = MBB.load()
    problems = MBB.build(data, mapping)
    if problems:
        print("!! briefs 構建問題：", problems[:5])
        raise SystemExit(2)


# ───────────────────── 成績排行榜（leaderboard）────────────────────
LEADERBOARD_ID = "score-leaderboard"
LEADER_BACKUP = ROOT / "Backup/leaderboard"


def _th_worst_label(a: dict) -> str:
    t = a.get("thermal")
    if isinstance(t, dict):
        return (t.get("worst") or {}).get("label", "?")
    return "?"


def _skip_scan_path(p: Path) -> bool:
    """這個檔不該被掃進來：**自己的輸出目錄**。

    WHY：`persist_leader_checks` 會把 /tmp 的來源复制進 `Backup/leaderboard/`，而掃描的 glob 包含
    `Backup/**` ⇒ 上一輪的副本會被當成新的來源收進來（同一筆成績在榜上出現兩次，且一輪一輪累積）。
    實測 2026-09-28：榜首前十有一半是副本。掃自己寫的東西就是一個會自我增長的輸入。
    """
    try:
        return _is_relative_to(p, LEADER_BACKUP)
    except Exception:  # noqa: BLE001  路徑比不出來就不排除（寧可多收也不要靜默漏）
        return False


def scan_artifact_arms():
    """掃所有產物（Backup＋/tmp，**排除自己的輸出目錄**），yield (arm, Path)。"""
    seen = set()
    for pat in ("Backup/**/*.json", "/tmp/**/*.json", "/tmp/*.json"):
        for f in glob.glob(pat, recursive=True):
            if f in seen:
                continue
            seen.add(f)
            p = Path(f)
            if _skip_scan_path(p):
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            arms = d if isinstance(d, list) else [d] if isinstance(d, dict) else []
            for a in arms:
                if isinstance(a, dict) and "rows" in a:
                    yield a, p


def _lb_entry(v, std, a, n, kind, src, row_warm=0):
    th = _th_worst_label(a)
    bad = bool(a.get("incomplete") or a.get("error"))
    if kind == "tg":
        steady = (not bad) and n >= 100 and (
            bool(a.get("warm_skip_applied")) or row_warm > 0
        ) and th in ("NOMINAL", "?")
    else:
        steady = (not bad) and n >= 1000 and th in ("NOMINAL", "?")
    return {
        "v": round(v, 3),
        "std": round(std, 3) if isinstance(std, (int, float)) else None,
        "arm": a.get("tag") or "?", "prof": a.get("profile") or "?",
        "n": n, "th": th, "spec": a.get("spec_type") or "-",
        "steady": bool(steady), "src": None, "_src_path": str(src),
    }


def _arm_on_board(a: dict, path=None) -> tuple:
    """(能不能上榜, 不能的原因)。榜規：**只能乾淨歸因**（與 `merge_run` 同一份判準），再加四條：
    09-29 兩條 —— **量具不是 UNBOUND**、**產物有它的成對 log**；09-30 兩條 ——
    **臂上沒有「不得引用吞吐」的量具（R5）**、**輸出有見證（R6）**（兩份清單向 `quote_gate` 要，不重寫）。

    前兩條是「寫不進決策面」的第三扇門延伸到排行榜：排行榜是決策面最容易被人抄一條數字的地方，
    而它在 09-29 就示範過那個漏洞 —— 榜首 17.669 t/s 出自一支沒有成對 log 的產物（見 §10）。
    後兩條是 09-30 的同一個故事再演一次：`quote_gate` 已把 `CGC_EB_NOFILL`（R5）與 `CGC_SEG_BATCH`（R6）
    判 DIRTY，排行榜卻還是把它們放在榜首 —— 一個判準有兩份實作，就會有一份落後。
    `path=None` 時判歸因與 R5／R6（臂身分不需要檔案）；量具與成對那兩條向 `instrument_binding` 要，不重寫。

    抽成一個純函式就是為了可以被測：`build_leaderboard` 要讀真實檔案，這條規則卻不該只有
    「跑一次排行榜看看」一種驗法。
    """
    # R5／R6（2026-09-30）：把 `quote_gate` 那兩份清單接上來。它們只問**臂的身分**（不看檔案），
    # 所以放在 `path is not None` 之外 —— 不該因為「這個檔判不了」就讓一條假數字上榜。
    # 實測（接之前）：decode 榜首是 `CGC_SEG_BATCH` 的 27.338（R6 單次提交臂，輸出未見證）
    # 與 `CGC_EB_NOFILL` 的 13.75（R5 診斷臂，量到的是終點上界）。兩筆在 `quote_gate` 都已是 DIRTY，
    # 只有排行榜這一格沒跟上。清單不重寫，向 `quote_gate` 要（單一來源）。
    try:
        import quote_gate as _qg
        _void = _qg.arm_instruments(a)
        _unver = _qg.unverified_output_arms(a)
    except Exception:  # noqa: BLE001  判不了就不因此刷掉（缺席不是「有問題」）
        _void, _unver = {}, {}
    if _void:
        return False, "R5 量具臂 %s" % "/".join(sorted(_void))
    if _unver:
        return False, "R6 輸出未見證臂 %s" % "/".join(sorted(_unver))
    # 【CGC 2026-10-06】第六條：spec 臂的 draft 死過 ⇒ 不得上榜。與 R5／R6 同一個理由只問**臂自己的
    # 產物欄位**（不看檔）：draft 死亡那一輪的 t/s 是純解碼的，不該因為「這個檔判不了」而爬上榜首。
    if _spec_draft_dead(a):
        return False, "spec draft 死亡"
    att = a.get("attribution")
    if path is not None:
        try:
            import instrument_binding as _ib
            rec = _instrument_of(a, str(path))
            if isinstance(rec, dict) and not _ib.is_bound(rec)[0]:
                return False, "量具 %s" % (rec.get("verdict") or "?")
            if not _ib.pair_status(str(path)).get("paired"):
                return False, "缺成對 log"
        except Exception:  # noqa: BLE001  判不了就不因此刷掉（缺席不是「有問題」）
            pass
    ok, _why = _clean_attribution(a)
    if ok:
        return True, ""
    if not isinstance(att, dict):
        return False, "沒有 attribution"
    v = str(att.get("verdict") or "未記錄")
    if v == "none":
        # `none` 是 `attribute()` 的「不是 HEAVY、沒有 swap/wired 成長、沒有別人」——但它**包含**
        # thermal MODERATE。那個仍然不是認可的引用條件（本 repo 的驗收寫的是量測期 NOMINAL），
        # 所以它不上榜；但理由要寫清楚，不能只報一個令人困惑的 `none`。
        th = att.get("thermal_worst")
        if th is not None and str(th) != "NOMINAL":
            return False, "none／thermal=%s" % th
    return False, v


def _merged_res(node: dict, auto: str) -> str:
    """把「自動刷新行」與「人工仲裁」合起來。純函式（`node` 只在第一次收編時被動到）。

    WHY 要有這一條（2026-09-30 實測）：`score-leaderboard` 的 `res` 是 09-29 手寫的降級裁定
    （「17.669 不得作『目前最好』」），而 `cmd_leaderboard` 是整段覆寫 —— 跑一次刷新，
    那個節點上唯一的「為什麼」就從決策面上消失了。裁定比刷新行重要，所以：
    第一次遇到非自動的 `res` 就把它收進 `res_ruling`，之後每輪都以 `res_ruling` 為準
    ⇒ 人工的部分永遠只有一份，也不會一輪一輪愈長。
    """
    prev = str(node.get("res") or "")
    ruling = str(node.get("res_ruling") or "")
    if not ruling and prev and not prev.startswith("排行榜更新（"):
        node["res_ruling"] = prev
        ruling = prev
    return auto if not ruling else ("%s\n\n%s" % (ruling, auto))


def build_leaderboard(top_n: int = 10) -> dict:
    """跨所有產物聚合；每指標分 steady（穩態生產口徑）與 all（全部，含實驗變體）兩榜。

    2026-09-28：**只有乾淨歸因的臂能上榜**（`memory_pressure.is_clean`，與 `merge_run`／
    `mindmap_void_check.py` 同一份判準）。2026-09-29 再收緊：量具 UNBOUND 或**缺成對 log**
    的臂也不上榜（`_arm_on_board`）—— 前者「開關設了但量具沒動」，後者連驗都沒得驗。
    原來的掃描把每一個帶 `rows` 的臂都收進來，
    於是榜首是 `prod25-stream`（MTP on）的 13.099 與 seg-batch 家族的 374.3 —— 全部是 §5.1
    不得引用的讀數，而節點的 `res` 把它們當成績寫出去（`mindmap_void_check` 現在會為此開單）。
    被刷掉的臂數與各自的 verdict 一併回傳（`scan`），**不是靜默丟掉**。
    """
    dec, pre = [], []
    scan = {"arms_total": 0, "arms_clean": 0, "skipped": {}}
    for a, p in scan_artifact_arms():
        scan["arms_total"] += 1
        ok, why = _arm_on_board(a, p)
        if not ok:
            scan["skipped"][why] = scan["skipped"].get(why, 0) + 1
            continue
        scan["arms_clean"] += 1
        for r in a.get("rows", []):
            if not isinstance(r, dict) or r.get("avg_ts") is None:
                continue
            if r.get("n_gen", 0) > 0:
                e = _lb_entry(r["avg_ts"], r.get("stddev_ts"), a,
                              r.get("n_gen"), "tg", p, r.get("warm_skip", 0))
                dec.append((e["v"], e))
            if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
                e = _lb_entry(r["avg_ts"], r.get("stddev_ts"), a,
                              r.get("n_prompt"), "pp", p)
                pre.append((e["v"], e))

    def split(lst):
        lst.sort(key=lambda x: x[0], reverse=True)
        all_e = [e for _, e in lst[:top_n]]
        std_l = sorted((x for x in lst if x[1]["steady"]),
                       key=lambda x: x[0], reverse=True)
        return {"steady": [e for _, e in std_l[:top_n]], "all": all_e}

    return {"decode": split(dec), "prefill": split(pre), "scan": scan}


def persist_leader_checks(lb: dict, stamp: str) -> dict:
    """檢驗檔持久化：源在 repo 內用相對路徑；/tmp 複製進 Backup/leaderboard；同源去重。"""
    LEADER_BACKUP.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = {}

    def check_for(src: Path) -> str:
        sp = str(src)
        if sp in cache:
            return cache[sp]
        if _is_relative_to(src, ROOT):
            rel = os.path.relpath(src.resolve(), ROOT)
        else:
            dest = LEADER_BACKUP / f"{stamp}_{src.name}"
            shutil.copy2(src, dest)
            rel = str(dest.relative_to(ROOT))
        cache[sp] = rel
        return rel

    for key in ("decode", "prefill"):
        for sub in ("steady", "all"):
            for e in lb[key][sub]:
                sp = e.pop("_src_path", None)
                if sp is None:
                    continue  # 同一 dict 已在另一榜處理（src 已設）
                e["src"] = check_for(Path(sp))
    return lb


def ensure_leaderboard_node(data: dict) -> dict:
    node = next((e for e in data["entries"] if e["id"] == LEADERBOARD_ID), None)
    if node is None:
        node = {
            "id": LEADERBOARD_ID,
            "name": "成績排行榜（最高配置＋檢驗檔）",
            "theme": "里程碑",
            "tier": "2",
            "goal": "跨所有實驗的 prefill/decode 最高成績排名；每條鏈到可複核檢驗檔，穩態口徑標 ✓。",
            "crit": "上榜須帶檢驗檔；穩態可信優先，非穩態口徑標註、不當生產能力。",
            "res": "由 experiment_sync leaderboard 機械生成",
            "evid": "Backup/leaderboard/",
            "sub": "both",
            "note": "綠＝穩態生產口徑；灰＝非穩態／異常口徑（短測量、熱壓、未驗正確性）。",
            "runs": [], "leaderboard": {"decode": [], "prefill": []},
        }
        data["entries"].append(node)
    return node


def cmd_leaderboard(args) -> int:
    """刷新成績排行榜。它也是一扇門：上榜的臂必須乾淨歸因＋量具綁上＋有成對 log
    （`_arm_on_board`），否則那條數字會從決策面上被抄走。"""
    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    lb = persist_leader_checks(build_leaderboard(top_n=args.top), stamp)
    data = load_mm()
    node = ensure_leaderboard_node(data)
    node["leaderboard"] = lb
    ds = lb["decode"]["steady"] or lb["decode"]["all"]
    ps = lb["prefill"]["steady"] or lb["prefill"]["all"]
    d0 = ds[0]["v"] if ds else "—"
    p0 = ps[0]["v"] if ps else "—"
    sc = lb.get("scan") or {}
    # 榜首要帶口徑：過濾掉不乾淨的臂之後，榜首仍然可能是 §5.0 **不認可的口徑**（實測 2026-09-28：
    # 過濾後 decode 榜首是 prod25 的 17.669）。不寫出來，讀者會把它當「我們的 decode」。
    dp = ds[0].get("prof", "?") if ds else "?"
    pp = ps[0].get("prof", "?") if ps else "?"
    flag_d = "" if dp == "prod-new" else f"（{dp}；**非 §5.0 認可口徑**）"
    flag_p = "" if pp == "prod-new" else f"（{pp}；**非 §5.0 認可口徑**）"
    node["res"] = _merged_res(
        node,
        f"排行榜更新（{stamp}）：decode 最高 {d0} t/s{flag_d}／prefill 最高 {p0} t/s{flag_p}"
        f"（僅乾淨歸因：{sc.get('arms_clean', '?')}/{sc.get('arms_total', '?')} 個臂）")
    save_mm(data)
    rebuild()
    print("成績排行榜已更新：", LEADERBOARD_ID)
    return 0


# ───────────────────────── 子命令 ────────────────────────────────

def cmd_instrument_audit(args) -> int:
    """唯讀：哪些產物武裝過受檢量具、它們綁上了嗎；以及哪些跑的產物缺成對 log。**不寫任何檔案**。

    WHY 要有這支：`_instrument_ok` 對「沒有 instrument 欄位的舊 run」一律放行（政策不回溯改寫語料）。
    放行的代價就是那些舊產物不再被自動看住 —— 所以清理它們必須是**看得見的一步**，也就是這支。
    rc：0 = 沒有 UNBOUND；1 = 至少一個 UNBOUND；2 = 工具不可用。
    """
    try:
        import instrument_binding as _ib
    except Exception as e:  # noqa: BLE001
        print(f"!! instrument_binding 不可用：{e}")
        return 2
    rows = _ib.scan(args.pattern)
    print("量具綁定稽核（唯讀；受檢開關：CGC_MISS_MASK_DBG、CGC_SEG_BATCH、CGC_SPAC_DBG）")
    print("  strict=%s（CGC_INSTRUMENT_STRICT=1 時，UNVERIFIABLE 也會被當不通過）"
          % _ib.strict_mode())
    if not rows:
        print("  沒有任何產物武裝過受檢量具 ⇒ N/A（不是通過）")
        return 0
    for row in rows:
        mark = {"BOUND": "✅", "N/A": "·", "UNVERIFIABLE": "⚠"}.get(row["verdict"], "⛔")
        print("  %s %-12s %-46s %s" % (mark, row["verdict"], row["artifact"],
                                       (row["why"] or "")[:80]))
    bad = [r for r in rows if r["verdict"] == "UNBOUND"]
    unver = [r for r in rows if r["verdict"] == "UNVERIFIABLE"]
    print("  小計：%d 列（BOUND=%d、UNBOUND=%d、UNVERIFIABLE=%d）"
          % (len(rows), sum(1 for r in rows if r["verdict"] == "BOUND"), len(bad), len(unver)))
    if unver and not _ib.strict_mode():
        print("  ⚠ UNVERIFIABLE = 產物旁邊沒有並存的 stderr log（沒有證據，不是證據說沒綁）；"
              "sync 預設不會因此翻掉那一輪。")
    # 成對那一半（2026-09-29）：上面的掃描問的是「武裝過的量具綁上了嗎」，這裡問的是
    # 更前面的一件事 —— 這一場的產物旁邊有沒有 log。沒有 ⇒ 不可引用（`_pair_ok` 也這樣判）。
    prows = _ib.pairs_audit(args.pattern)
    pmiss = [r for r in prows if not r["paired"]]
    print("成對稽核（唯讀）：%d 支跑出來的產物，%d 支缺成對 log" % (len(prows), len(pmiss)))
    for r in pmiss:
        print("  ⛔ %-52s %s" % (r["artifact"], r["why"][:70]))
        for h in (r.get("workdir_log_candidates") or [])[:3]:
            print("        · 可能是同一場的 log（需人工比對）：%s" % h)
    return 1 if (bad or pmiss) else 0


def cmd_init(args) -> int:
    charter = load_charter(args.charter)
    cid = str(charter.get("id", "")).strip()
    sub = resolve_sub(charter, args.sub)
    data = load_mm()
    byid = {e["id"]: e for e in data["entries"]}
    if cid in byid and not args.update:
        raise SystemExit(f"已存在節點 {cid}（加 --update 可刷新描述、保留歷史 runs）")
    entry = build_entry(charter, sub, args.name)
    if cid in byid:
        old = byid[cid]
        for k in ("name", "theme", "tier", "goal", "crit", "res", "evid", "sub", "note", "targets"):
            old[k] = entry[k]
        old.setdefault("runs", [])
        old.setdefault("subtasks", entry["subtasks"])
        old["best"] = compute_best(old["runs"])
        old["target_gap"] = target_gap(old.get("targets") or {}, old["best"])
        print(f"已更新實驗節點 {cid}（軸 {sub}，歷史 runs 保留）")
    else:
        data["entries"].append(entry)
        print(f"已建立實驗節點 {cid}（軸 {sub}）")
    save_mm(data)
    rebuild()
    return 0


def sync_artifact_file(path, cid_default=None, do_rebuild=True, stamp=None, when=None,
                       waive_reason=None):
    """把一支產物檔（dict 或 arm list）回寫到對應節點：build_run＋merge_run、save、選擇性 rebuild。
    獨立成函式，讓 harness bench 跑完可直接調用（D 自動回寫），cmd_sync 與 harness 走同一條路徑。

    第三扇門就在這裡（`sync_gate`）：量具 UNBOUND ⇒ 拒寫（除非豁免）；缺成對 log ⇒ 拒寫（無豁免）。
    放在這一層而不是 cmd_sync 裡，是因為 harness 的 auto-sync 也走這條路 —— 閘門不能只在其中一扇門上。
    """
    p = Path(path)
    if not p.exists():
        print(f"  跳過：產物不存在 {path}")
        return []
    gate = sync_gate(p, waive_reason=waive_reason)
    if not gate["ok"]:
        print("!! 量具閘（同步）FAIL — **拒寫**：這支產物不會進決策面（mindmap）。")
        print(f"   {gate['why']}")
        for a in gate["arms"]:
            print("   · 臂 %s：量具 %s（%s）" % (a["tag"], a["verdict"], a["why"]))
        if gate["code"] == 5:
            print("   補救：先把 log 放到產物旁邊（`instrument_binding --pair-backfill <artifact> "
                  "--from <那份 log>`）—— 這一條沒有豁免。")
        elif gate["code"] == 6:
            print("   補救：修掉 draft 為什麼死（例：CGC_SPEC_TREE=0 的純鏈缺 draft-ctx mirror trim）"
                  "再重跑 —— 這一條沒有豁免：draft 死掉那一輪的 t/s 是純解碼的 t/s。")
        else:
            print("   補救：修好量具重跑，或明知不可引用而要留下這一筆：sync --waive-instrument \"理由\"")
        raise SyncRefused(gate["code"], gate)
    if gate.get("waived"):
        print(f"   ⚠ 量具閘已豁免（{gate['waived']}）：這一筆會被標成不可引用，但不阻止它留在節點上")
    if gate.get("why"):
        print(f"   [sync-gate] {gate['why']}")
    now = dt.datetime.now()
    stamp = stamp or now.strftime("%Y%m%d_%H%M%S")
    when = when or now.strftime("%Y-%m-%d %H:%M")
    data = load_mm()
    byid = {e["id"]: e for e in data["entries"]}
    arms = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(arms, dict):
        arms = [arms]
    touched = []
    for arm in arms:
        cid = charter_id(arm) or cid_default
        if not cid:
            print(f"  跳過：{p.name} 的 arm '{arm.get('tag')}' 無 charter（用 --entry 指定）")
            continue
        entry = byid.get(cid)
        if not entry:
            print(f"  跳過：找不到節點 {cid}（先 init）")
            continue
        log_rel = ensure_log_in_repo(p, cid, stamp)
        run = build_run(arm, log_rel, when)
        if merge_run(entry, run):
            touched.append(cid)
            print(f"  {cid}：追加 run（{run.get('result') or '無結果'}）")
    if touched:
        save_mm(data)
        if do_rebuild:
            rebuild()
    return sorted(set(touched))


def cmd_sync(args) -> int:
    arts = [a for a in args.artifact if Path(a).exists()]
    for m in args.artifact:
        if not Path(m).exists():
            print(f"  跳過：產物不存在 {m}")
    touched = []
    for i, a in enumerate(arts):
        try:
            touched += sync_artifact_file(a, cid_default=args.entry,
                                          do_rebuild=(i == len(arts) - 1),
                                          waive_reason=getattr(args, "waive_instrument", None))
        except SyncRefused as e:
            # 第三扇門擋下：rc 沿用 harness 兩個門的編號（5 = 缺成對、4 = 量具 UNBOUND）。
            print(f"已中止：{os.path.basename(str(a))} 未寫入（rc={e.code}）")
            return e.code
    if not touched:
        print("沒有任何節點被更新")
        return 1
    print("已更新節點：", touched)
    return 0


# ───────────────────────── selftest ──────────────────────────────

def selftest() -> bool:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    charter = {
        "id": "exp-demo", "owner": "agent-x",
        "question": "把 X 並行化之後 decode 能否提升超過 3%？",
        "baseline": {"metric": "decode t/s", "cell": "prod-new 穩態", "source": "docs/abc.md"},
        "hypothesis": {"mechanism": "X 與 GPU 重疊", "expected": "+8%", "basis": "分解"},
        "acceptance": {"success": "decode ≥ 13.2 t/s", "falsify": "配對中位 < +3%",
                       "on_fail": "收攤"},
        "arms": ["prod-new"],
    }
    entry = build_entry(charter, "S")
    chk("build_entry：tier=3a、軸 S", entry["tier"] == "3a" and entry["sub"] == "S")
    chk("build_entry：goal=question、crit=success",
        entry["goal"].startswith("把 X") and entry["crit"] == "decode ≥ 13.2 t/s")
    chk("build_entry：4 條初始 subtasks、首條 done",
        len(entry["subtasks"]) == 4 and entry["subtasks"][0]["status"] == "done")
    chk("build_entry：runs 初始為空", entry["runs"] == [])

    try:
        resolve_sub(charter, None)
        chk("resolve_sub：無軸應拒", False)
    except ValueError:
        chk("resolve_sub：無軸應拒", True)
    chk("resolve_sub：charter.axis 可用", resolve_sub({**charter, "axis": "M"}, None) == "M")
    chk("short_name：過長截斷", short_name("q" * 50, "fb").endswith("…"))

    arm = {
        "tag": "prod-new", "profile": "prod-new", "warm_skip": 64,
        "charter": {"id": "exp-demo"},
        "rows": [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 280.1},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 12.3},
        ],
        "sys_before": {"swap_used_mb": 100.0, "thermal": {"label": "NOMINAL"}},
        "sys_after": {"swap_used_mb": 130.0, "thermal": {"label": "NOMINAL"}},
        "thermal": {"worst": "NOMINAL"},
        "attribution": {"verdict": "none", "why": "thermal=NOMINAL", "thermal_worst": "NOMINAL"},
    }
    chk("charter_id：dict 取 id", charter_id(arm) == "exp-demo")
    chk("charter_id：waived→None", charter_id({"charter": {"waived": True}}) is None)
    chk("row_ts：pp=280.1", row_ts(arm["rows"], "pp") == 280.1)
    chk("row_ts：tg=12.3", row_ts(arm["rows"], "tg") == 12.3)

    run = build_run(arm, "Backup/x.json", "2026-09-27 12:00")
    chk("build_run：result 含 pp/tg", run["result"] == {"pp": 280.1, "tg": 12.3})
    chk("build_run：swap before→after",
        run["swap"]["before"] == "100 MiB" and run["swap"]["after"] == "130 MiB")
    chk("build_run：thermal 與 verdict",
        run["thermal"] == "NOMINAL" and run["verdict"].startswith("none"))

    fresh = build_entry(charter, "S")
    chk("merge_run：首次新增、res 更新",
        merge_run(fresh, run) and fresh["res"].startswith("decode 12.30"))
    st_run = next(t for t in fresh["subtasks"] if t["id"] == "st-run")
    chk("merge_run：st-run 標 done", st_run["status"] == "done")
    chk("merge_run：同 log 去重（不重複）", merge_run(fresh, run) is False)

    # ── 非乾淨歸因不得上成績面（2026-09-28：swap 的 9.94 t/s 曾變成 node 的 res）──
    void_arm = {**arm, "attribution": {"verdict": "swap", "why": "swap_growth=1579.0 MiB",
                                     "thermal_worst": "NOMINAL"}}
    void_run = build_run(void_arm, "Backup/void.json", "2026-09-28 23:12")
    chk("build_run：swap 的 run 被標成不乾淨", void_run["clean"] is False and "swap" in void_run["void_why"])
    chk("build_run：乾淨的 run 標成乾淨", run["clean"] is True)
    e_v = build_entry(charter, "S")
    merge_run(e_v, void_run)
    chk("merge_run：非乾淨的 run **不得**寫出 t/s 的 res：" + str(e_v["res"]),
        "t/s" not in str(e_v["res"]) and "尚無可引用讀數" in str(e_v["res"]))
    chk("merge_run：並且留下為什麼（no_throughput_claim）",
        "swap" in str(e_v.get("no_throughput_claim")))
    # 人寫的判詞（不是吞吐主張）不得被機器句子蓋掉 —— 這是這個規則的邊界，要有測試釘住。
    e_prose = build_entry(charter, "S")
    e_prose["res"] = "主 context ntok=1 的 churn 非 0（publish 級 3026/14976）"
    merge_run(e_prose, void_run)
    chk("merge_run：人寫的判詞 res 不被蓋掉、但 no_throughput_claim 仍要種下",
        e_prose["res"].startswith("主 context ntok=1") and e_prose.get("no_throughput_claim"))
    chk("compute_best：全部非乾淨 ⇒ 沒有最好的（而非挑最高的）", e_v["best"] is None)

    # ── 量具沒綁上 ⇒ 也不能上成績面（2026-09-29：MM 回讀全部印 0 的局）──
    unbound_run = build_run({**arm, "instrument": {"verdict": "UNBOUND",
                                                 "why": "武裝了 mm_pub_n_leaf、mm_pub_wrote，但全部是 0"}},
                            "Backup/unbound.json", "2026-09-29 01:30")
    chk("build_run：量具 UNBOUND ⇒ run 不乾淨",
        unbound_run["clean"] is False and "量具未綁上" in unbound_run["void_why"])
    e_i = build_entry({**charter, "targets": {"decode_tps": 20.0}}, "S")
    merge_run(e_i, unbound_run)
    chk("merge_run：量具 UNBOUND 的 run 不得寫出 t/s 的 res：" + str(e_i["res"]),
        "t/s" not in str(e_i["res"]) and e_i["best"] is None)
    chk("  且 no_throughput_claim 要點名量具",
        "量具未綁上" in str(e_i.get("no_throughput_claim")))
    # ── 成對 log：缺 ⇒ 不乾淨（2026-09-29：log 是唯一能證明「量具真的動了」的東西）──
    chk("judge：缺成對 log ⇒ 不乾淨（沒有 log 所以驗不了，不該是一種狀態）",
        _pair_ok({"pair": {"paired": False, "why": "找不到 x.stderr.log"}})[0] is False)
    chk("judge：成對 ⇒ 通過", _pair_ok({"pair": {"paired": True, "why": "成對"}})[0])
    chk("judge：不是在跑的產物（推不出）⇒ 這條不適用，不得誤殺",
        _pair_ok({"log": "Backup/docs/not_a_run.md"})[0])
    chk("judge：沒有 pair 欄位的舊 run 當場算（檔案不在 ⇒ 不適用，不誤殺）",
        _pair_ok({"log": "Backup/nope_missing.json"})[0])
    chk("_run_clean：歸因乾淨＋量具綁上＋成對 ⇒ 乾淨",
        _run_clean({"clean": True, "instrument": {"verdict": "BOUND"},
                    "pair": {"paired": True}})[0])
    _pclean = _run_clean({"clean": True, "instrument": {"verdict": "BOUND"},
                          "pair": {"paired": False, "why": "找不到 x.stderr.log"}})
    chk("_run_clean：只缺成對 ⇒ 不乾淨，且理由點名成對：" + str(_pclean[1]),
        _pclean[0] is False and "成對" in _pclean[1])
    # ── 第三扇門：**同步**本身（手動 sync 與 harness auto-sync 都走 sync_artifact_file）──
    import tempfile
    _sd = tempfile.mkdtemp(prefix="es_sync_")
    try:
        _att = {"verdict": "none", "why": "thermal=NOMINAL", "thermal_worst": "NOMINAL"}

        def _art(name, verdict, rows=True):
            p = os.path.join(_sd, name)
            a = {"tag": "prod-new", "attribution": dict(_att)}
            if rows:
                a["rows"] = [{"n_prompt": 0, "n_gen": 64}]
            if verdict:
                a["instrument"] = {"verdict": verdict, "why": "測試"}
            with open(p, "w", encoding="utf-8") as f:
                json.dump([a], f)
            return p

        _doca = _art("doc.json", None, rows=False)          # 不是在跑的產物 ⇒ 成對不適用
        _bare = _art("bare.json", "BOUND")                    # 跑的產物、量具 BOUND、但沒有 log
        _unb = _art("unb.json", "UNBOUND")
        # draft 死亡的產物：量具 BOUND、也有成對 log，唯一的問題是 spec_draft=INVALID
        _dead = _art("dead.json", "BOUND")
        with open(_dead, encoding="utf-8") as fh:
            _dj = json.load(fh)
        _dj[0]["spec_draft"] = {"state": "INVALID", "n_reps": 5, "dead": [
            {"phase": "timed", "rounds": 128, "drafted": 0, "emit_per_round": "1.0000"}],
            "why": "4/5 輪 rep 的 draft 死亡"}
        with open(_dead, "w", encoding="utf-8") as fh:
            json.dump(_dj, fh)
        import instrument_binding as _ibx   # 寫 log 用同一份規則（不在測試裡自己拼檔名）
        _ibx.write_pair_bundle(_unb, [("prod-new", "p0_n128", "x\n")])
        chk("sync gate：不是在跑的產物 ⇒ 不適用（不得誤殺）", sync_gate(_doca)["ok"])
        g = sync_gate(_bare)
        chk("sync gate：缺成對 log ⇒ 拒寫（rc=5），且理由說出不可引用",
            (g["ok"], g["code"], "不可引用" in (g["why"] or "")) == (False, 5, True))
        g = sync_gate(_unb)
        chk("sync gate：量具 UNBOUND ⇒ 拒寫（rc=4）", (g["ok"], g["code"]) == (False, 4))
        chk("... 且點名是哪一臂的哪一個判定",
            g["arms"] and g["arms"][0]["verdict"] == "UNBOUND" and not g["arms"][0]["ok"])
        g = sync_gate(_unb, waive_reason="急件：先留下這一筆")
        chk("sync gate：UNBOUND 可豁免（理由進產物）", (g["ok"], g["waived"]) == (True, "急件：先留下這一筆"))
        import instrument_binding as _iby
        _iby.write_pair_bundle(_dead, [("prod-new", "p0_n128", "x\n")])
        _gd = sync_gate(_dead)
        chk("sync gate：spec draft 死亡 ⇒ 拒寫（rc=6），即使量具 BOUND、成對也齊全",
            (_gd["ok"], _gd["code"]) == (False, 6))
        chk("... 且理由說出「純解碼」與是哪幾輪：" + str(_gd["why"]),
            "純解碼" in (_gd["why"] or "") and "draft" in (_gd["why"] or ""))
        _gdw = sync_gate(_dead, waive_reason="急件")
        chk("sync gate：draft 死亡**不可豁免**（豁免只給量具）",
            (_gdw["ok"], _gdw["code"]) == (False, 6))
        chk("sync gate：豁免不讓它變乾淨（run 仍非乾淨）",
            _run_clean({"clean": True, "instrument": {"verdict": "UNBOUND"},
                        "pair": {"paired": True}})[0] is False)
        chk("sync gate：SyncRefused 帶著編號（4/5 才有下一步）",
            (SyncRefused(5, {"why": "x"}).code, str(SyncRefused(5, {"why": "x"}))) == (5, "x"))
        # 排行榜也是決策面：量具 UNBOUND 與缺成對 log 的臂不得上榜
        _clean_arm = {"tag": "prod-new", "attribution": dict(_att),
                      "instrument": {"verdict": "BOUND", "why": "x"}}
        chk("leaderboard：量具 UNBOUND 不得上榜", _arm_on_board(
            {"tag": "p", "attribution": dict(_att),
             "instrument": {"verdict": "UNBOUND", "why": "x"}}, _bare)[0] is False)
        chk("leaderboard：缺成對 log 不得上榜（且理由點名成對）",
            _arm_on_board(_clean_arm, _bare)[1] == "缺成對 log")
        _ibx.write_pair_bundle(_bare, [("prod-new", "p0_n128", "y\n")])
        chk("leaderboard：補上成對 log 且量具 BOUND ⇒ 可以上榜",
            _arm_on_board(_clean_arm, _bare)[0])
        chk("leaderboard：spec draft 死亡不得上榜（不需要檔案，看臂自己的欄位）",
            _arm_on_board(dict(_clean_arm, spec_draft={"state": "INVALID", "why": "drafted=0"}))[0] is False)
        chk("leaderboard：draft 健康的臂不受這一條影響",
            _arm_on_board(dict(_clean_arm, spec_draft={"state": "OK", "why": "5/5 drafted>0"}))[0])
        chk("leaderboard：path=None 時判歸因與 R5／R6（臂身分不需要檔案）",
            _arm_on_board({"tag": "p", "attribution": dict(_att)})[0])
        # 09-30：一個判準有兩份實作 ⇒ 一份會落後。把 `quote_gate` 的 R5／R6 接上來測。
        chk("leaderboard：R5 診斷臂不得上榜（CGC_EB_NOFILL 量到的是終點上界）",
            _arm_on_board({"tag": "prod-new:CGC_EB_TIMER=1;CGC_EB_NOFILL=1",
                           "attribution": dict(_att)})[0] is False)
        chk("leaderboard：R6 輸出未見證臂不得上榜（CGC_SEG_BATCH 是單次提交臂）",
            _arm_on_board({"tag": "prod-new:CGC_SEG_BATCH=1;CGC_B_SCHEME=1",
                           "attribution": dict(_att)})[0] is False)
        chk("leaderboard：R5／R6 不得誤殺乾淨臂（同一個 tag 少掉那一支）",
            _arm_on_board({"tag": "prod-new:CGC_EB_TIMER=1;CGC_FILL_SPLIT=1",
                           "attribution": dict(_att)})[0])
        chk("leaderboard：`env`／`extra_env` 裡的 R5 也要看到（不只 tag）",
            _arm_on_board({"tag": "prod-new", "extra_env": {"CGC_SPAC_DBG": "1"},
                           "attribution": dict(_att)})[0] is False)
        # 刷新不得吃掉人工仲裁
        _n = {"res": "仲裁：17.669 降級為分布尾端的一次抽樣"}
        _n["res"] = _merged_res(_n, "排行榜更新（x）：decode 最高 9 t/s")
        chk("leaderboard：人工仲裁收進 res_ruling（不是靜默消失）",
            _n.get("res_ruling") == "仲裁：17.669 降級為分布尾端的一次抽樣")
        chk("leaderboard：res 同時保留仲裁與自動行",
            _n["res"].startswith("仲裁：17.669 降級") and "排行榜更新（x）" in _n["res"])
        _n["res"] = _merged_res(_n, "排行榜更新（y）：decode 最高 8 t/s")
        chk("leaderboard：多輪刷新不堆疊（人工一份、自動一行）",
            (_n["res"].count("排行榜更新（"), _n["res"].count("仲裁：17.669 降級")) == (1, 1))
        chk("leaderboard：沒有仲裁時 res 就是自動行",
            _merged_res({}, "排行榜更新（z）") == "排行榜更新（z）")
    finally:
        shutil.rmtree(_sd, ignore_errors=True)

    chk("judge：N/A（沒武裝）不算不通過", _instrument_ok({"instrument": {"verdict": "N/A"}})[0])
    chk("judge：BOUND 通過", _instrument_ok({"instrument": {"verdict": "BOUND"}})[0])
    chk("judge：沒有 instrument 欄位的舊 run 不得因政策而翻（要翻得看 instrument-audit）",
        _instrument_ok({"clean": True})[0])
    chk("judge：UNVERIFIABLE 預設只記不翻",
        _instrument_ok({"instrument": {"verdict": "UNVERIFIABLE",
                                         "why": "找不到並存的 stderr log"}})[0])
    _old_strict = os.environ.get("CGC_INSTRUMENT_STRICT")
    os.environ["CGC_INSTRUMENT_STRICT"] = "1"
    try:
        chk("judge：CGC_INSTRUMENT_STRICT=1 時 UNVERIFIABLE 也擋",
            _instrument_ok({"instrument": {"verdict": "UNVERIFIABLE",
                                             "why": "找不到並存的 stderr log"}})[0] is False)
    finally:
        if _old_strict is None:
            os.environ.pop("CGC_INSTRUMENT_STRICT", None)
        else:
            os.environ["CGC_INSTRUMENT_STRICT"] = _old_strict
    e_v_tg = build_entry({**charter, "targets": {"decode_tps": 20.0}}, "S")
    merge_run(e_v_tg, void_run)
    chk("target_gap：沒有可引用的 best ⇒ current 空著（不拿 VOID 數字算進度；那 49.7% 的形狀）",
        e_v_tg["target_gap"]["decode_tps"]["current"] is None
        and e_v_tg["target_gap"]["decode_tps"]["pct"] is None)
    # 這才是 2026-09-28 真正發生過的形狀：一個 swap 的 9.94 被寫成 res 又算成 49.7% 目標達成率
    e_v2 = build_entry({**charter, "targets": {"decode_tps": 20.0}}, "S")
    merge_run(e_v2, build_run({**void_arm, "rows": [{"n_prompt": 0, "n_gen": 64, "avg_ts": 9.94}]},
                              "Backup/void2.json", "2026-09-28 23:12"))
    chk("回歸：swap 的 9.94 不得再變成 'decode 9.94 t/s（swap）'：" + str(e_v2["res"]),
        "9.94" not in str(e_v2["res"]))
    # 後來有乾淨的 run 進來 ⇒ 恢復讀數、把標記拿掉
    merge_run(e_v, build_run(arm, "Backup/clean2.json", "2026-09-29 09:00"))
    chk("merge_run：後來的乾淨 run 讓 res 恢復讀數：" + str(e_v["res"]),
        str(e_v["res"]).startswith("decode 12.30"))
    chk("...並且把 no_throughput_claim 拿掉", "no_throughput_claim" not in e_v)
    chk("compute_best：在好壞混合的 run set 裡只採乾淨的",
        (e_v["best"] or {}).get("tg") == 12.3 and (e_v["best"] or {}).get("log") == "Backup/clean2.json")

    # ── A＋B：targets 解析 / best-config / 目標差距 ─────────────────────
    ch_t = {**charter, "targets": {"decode_tps": 15.0, "prefill_tps": 250.0}}
    e_t = build_entry(ch_t, "S")
    chk("build_entry：targets 解析、best 初始 None",
        e_t["targets"] == {"decode_tps": 15.0, "prefill_tps": 250.0} and e_t["best"] is None)
    chk("build_entry：target_gap 初始 current 空、target 在",
        e_t["target_gap"]["decode_tps"]["current"] is None
        and e_t["target_gap"]["decode_tps"]["target"] == 15.0)

    # 這三筆手做 fixture 測的是「挑最高」；歸因要明說乾淨，否則 compute_best 會（正確地）不採它們
    # —— 2026-09-28 加的 fail-closed 規則：證明不了乾淨就不上成績面。
    _ok = {"verdict": "none：thermal=NOMINAL"}
    r1 = {"result": {"tg": 11.0, "pp": 270.0}, "arm": "a1", "log": "l1", "when": "t1", "verdict": _ok["verdict"]}
    r2 = {"result": {"tg": 12.8, "pp": 260.0}, "arm": "a2", "log": "l2", "when": "t2", "verdict": _ok["verdict"]}
    r3 = {"result": {"tg": 12.1, "pp": 290.0}, "arm": "a3", "log": "l3", "when": "t3", "verdict": _ok["verdict"]}
    b = compute_best([r1, r2, r3])
    chk("compute_best：取 tg 最高 r2、帶 arm/log/pp",
        b["tg"] == 12.8 and b["pp"] == 260.0 and b["arm"] == "a2" and b["log"] == "l2")
    chk("compute_best：空 runs → None", compute_best([]) is None)
    bpp = compute_best([{"result": {"pp": 280.0}, "arm": "p", "log": "lp", "when": "tp",
                        "verdict": "none：thermal=NOMINAL"}])
    chk("compute_best：僅 pp → metric=pp、tg None",
        bpp["metric"] == "pp" and bpp["pp"] == 280.0 and bpp["tg"] is None)

    g = target_gap({"decode_tps": 15.0, "prefill_tps": 250.0}, b)
    chk("target_gap：decode gap=-2.2 pct=85.3",
        g["decode_tps"]["current"] == 12.8 and g["decode_tps"]["gap"] == -2.2
        and g["decode_tps"]["pct"] == 85.3)
    chk("target_gap：prefill 達標 gap=10.0 pct=104.0",
        g["prefill_tps"]["current"] == 260.0 and g["prefill_tps"]["gap"] == 10.0
        and g["prefill_tps"]["pct"] == 104.0)

    e2 = build_entry(ch_t, "S")
    merge_run(e2, {"result": {"tg": 11.0}, "arm": "a1", "log": "l1", "when": "t1",
                  "thermal": "NOMINAL", "swap": {}, "verdict": "none：thermal=NOMINAL"})
    merge_run(e2, {"result": {"tg": 12.8, "pp": 260.0}, "arm": "a2", "log": "l2",
              "when": "t2", "thermal": "NOMINAL", "swap": {},
              "verdict": "none：thermal=NOMINAL"})
    chk("merge_run：best 取最高 12.8、target_gap 同步 pct=85.3",
        e2["best"]["tg"] == 12.8 and e2["best"]["arm"] == "a2"
        and e2["target_gap"]["decode_tps"]["pct"] == 85.3)
    # 歸因**沒記錄**（verdict 空字串）也算不乾淨：缺席不是乾淨。
    e_unk = build_entry(ch_t, "S")
    merge_run(e_unk, {"result": {"tg": 12.5}, "arm": "a9", "log": "l9", "when": "t9",
                      "thermal": "NOMINAL", "swap": {}, "verdict": ""})
    chk("merge_run：verdict 沒記錄（空）⇒ 不上成績面（fail-closed）",
        e_unk["best"] is None and "t/s" not in str(e_unk["res"]))

    # [CGC 2026-10-03] 多臂產物：每一臂都指向**同一份** log（`ensure_log_in_repo` 對 repo 內的產物
    # 直接回它的相對路徑）⇒ 去重鍵少了 `arm` 就會靜默吃掉第二臂之後的每一臂，而那正好是
    # off/on 配對實驗的對照臂。突變檢查：把去重鍵退回只用 `log`，下面兩條都會紅。
    e_multi = build_entry(ch_t, "S")
    merge_run(e_multi, {"result": {"tg": 8.7}, "arm": "on", "log": "same.json", "when": "t1",
                        "thermal": "NOMINAL", "swap": {}, "verdict": "none：thermal=NOMINAL"})
    merge_run(e_multi, {"result": {"tg": 11.5}, "arm": "off", "log": "same.json", "when": "t1",
                        "thermal": "NOMINAL", "swap": {}, "verdict": "none：thermal=NOMINAL"})
    chk("merge_run：同一 log 的兩個臂各記一條（多臂產物不吞臂）",
        len(e_multi["runs"]) == 2 and [r["arm"] for r in e_multi["runs"]] == ["on", "off"])
    chk("merge_run：再 sync 同一 (log, arm) 仍冪等（不重複）",
        merge_run(e_multi, {"result": {"tg": 8.7}, "arm": "on", "log": "same.json",
                            "when": "t1", "thermal": "NOMINAL", "swap": {},
                            "verdict": "none：thermal=NOMINAL"}) is False
        and len(e_multi["runs"]) == 2)

    # ── 排行榜：只有乾淨歸因的臂能上榜（2026-09-28）───────────────
    chk("leaderboard：乾淨的臂可以上榜", _arm_on_board({"attribution": {"verdict": "none"}})[0] is True)
    chk("leaderboard：swap／both／thermal／contention 都不上榜",
        all(_arm_on_board({"attribution": {"verdict": v}})[0] is False
            for v in ("swap", "both", "thermal", "contention")))
    chk("leaderboard：沒有 attribution 欄位的臂也不上榜（缺席不是乾淨）",
        _arm_on_board({"rows": []})[0] is False
        and _arm_on_board({"attribution": {"verdict": "swap"}})[1] == "swap")
    chk("leaderboard：verdict=none 但 thermal 非 NOMINAL 也不上榜，且理由寫清楚",
        _arm_on_board({"attribution": {"verdict": "none", "thermal_worst": "MODERATE"}})
        == (False, "none／thermal=MODERATE"))
    chk("leaderboard：不掃自己的輸出目錄（否則副本會一輪一輪累積）",
        _skip_scan_path(LEADER_BACKUP / "x.json") is True
        and _skip_scan_path(ROOT / "Backup" / "y.json") is False)

    # ── C：技術子目標 subgoals 生成 / 實測貢獻回填與累加 ───────────────
    ch_sg = {**charter, "subgoals": [
        {"id": "sg-gap", "text": "gap 42→<15", "expect": "-27 ms", "how": "GPU_TIMING"},
        {"id": "sg-cb", "text": "cb 重疊", "expect": "+1.5 t/s", "how": "DECPROF"},
    ]}
    e_sg = build_entry(ch_sg, "S")
    sg_ids = [t["id"] for t in e_sg["subtasks"]]
    chk("C：追加 2 技術子目標（4 流程 → 6 subtasks）",
        len(e_sg["subtasks"]) == 6 and "sg-gap" in sg_ids and "sg-cb" in sg_ids)
    sg_gap0 = next(t for t in e_sg["subtasks"] if t["id"] == "sg-gap")
    chk("C：子目標帶 expect/how、contrib None、status todo",
        sg_gap0["expect"] == "-27 ms" and sg_gap0["how"] == "GPU_TIMING"
        and sg_gap0["contrib"] is None and sg_gap0["status"] == "todo")

    merge_run(e_sg, {"result": {"tg": 12.3}, "arm": "a", "log": "sg-l1", "when": "t",
                     "thermal": "NOMINAL", "swap": {}, "verdict": "",
                     "subgoal_contrib": {"sg-gap": -24, "sg-cb": 1.2}})
    sg_gap1 = next(t for t in e_sg["subtasks"] if t["id"] == "sg-gap")
    sg_cb1 = next(t for t in e_sg["subtasks"] if t["id"] == "sg-cb")
    chk("C：貢獻回填、status 轉 doing",
        sg_gap1["contrib"] == -24 and sg_gap1["status"] == "doing"
        and sg_cb1["contrib"] == 1.2 and sg_cb1["status"] == "doing")
    merge_run(e_sg, {"result": {"tg": 12.4}, "arm": "a", "log": "sg-l2", "when": "t",
                     "thermal": "NOMINAL", "swap": {}, "verdict": "",
                     "subgoal_contrib": {"sg-gap": -3}})
    sg_gap2 = next(t for t in e_sg["subtasks"] if t["id"] == "sg-gap")
    chk("C：同子目標數值貢獻累加（-24 + -3 = -27）", sg_gap2["contrib"] == -27)

    # build_run 透傳 subgoal_contrib（產物 arm 帶 → run 帶）
    arm_sg = {**arm, "subgoal_contrib": {"sg-gap": -24}}
    chk("C：build_run 透傳 subgoal_contrib",
        build_run(arm_sg, "Backup/sg.json", "t")["subgoal_contrib"] == {"sg-gap": -24})
    return ok


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="charter ↔ mindmap 實驗節點同步")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="從 charter 建實驗節點")
    p_init.add_argument("--charter", required=True)
    p_init.add_argument("--sub", choices=SUBS)
    p_init.add_argument("--name")
    p_init.add_argument("--update", action="store_true")

    p_sync = sub.add_parser("sync", help="從 harness bench 產物更新節點")
    p_sync.add_argument("--artifact", required=True, nargs="+")
    p_sync.add_argument("--entry", help="產物無 charter 時，指定要寫入的節點 id")
    p_sync.add_argument("--waive-instrument", default=None,
                        help="理由字串：明知量具 UNBOUND 而仍要留下這一筆（會標成不可引用，不變乾淨）。"
                             "缺成對 log 那一條不能豁免")

    p_lb = sub.add_parser("leaderboard", help="掃描所有產物，刷新成績排行榜節點")
    p_lb.add_argument("--top", type=int, default=10)

    p_ia = sub.add_parser("instrument-audit",
                          help="唯讀：列出所有武裝過受檢量具的產物及其綁定判定（不寫任何東西）")
    p_ia.add_argument("--pattern", action="append",
                      default=["Backup/**/*.json"], help="產物 glob（可多次）")

    args = ap.parse_args(argv)
    if args.selftest:
        print("════ experiment_sync selftest ════")
        ok = selftest()
        print(f"\n{'SELFTEST OK' if ok else 'SELFTEST FAIL'}")
        return 0 if ok else 1
    if args.cmd == "init":
        return cmd_init(args)
    if args.cmd == "sync":
        return cmd_sync(args)
    if args.cmd == "leaderboard":
        return cmd_leaderboard(args)
    if args.cmd == "instrument-audit":
        return cmd_instrument_audit(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
