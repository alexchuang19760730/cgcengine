#!/usr/bin/env python3
"""cell_contract.py — prod-new 生產 cell 的單一口徑校驗。

從測試卡（docs/PROD_NEW_TEST_CARD_*.md §2.5）讀 machine-readable CELL，在 llama-bench
**真正執行前**，把「實際要用的 cell 維度」與權威 block 對照：

- 嚴格權威維度（cell 裡、不在 runtime_adjustable）不一致 ⇒ mismatch（**fail-closed，拒跑**）
- runtime_adjustable 維度不一致 ⇒ declared（允許、但需記錄在產物）
- 讀不到 / 壞 block / 缺維度 ⇒ ContractError（fail-closed，不把缺席當預設值）

臂專用開關（P0/P1/P2、MTP、SPAC_HOT…）改的是記憶體／填充行為、**不改 cell 形狀**，
因此它們不豁免任何嚴格維度。

接入點：llama_bench_matrix.run_arm() 拼出最終 llama-bench cmd 後、subprocess 執行前。
matrix 是三個入口（harness bench / commit_bench / 直跑 matrix）唯一真正拼 llama-bench
的地方，在該處校驗即覆蓋全部入口。

用法：
    from cell_contract import check_cell
    rep = check_cell(actual, arm_env=extra_env)
    if not rep.ok: raise SystemExit(rep.render())

    python3 scripts/check/cell_contract.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


class ContractError(RuntimeError):
    pass


def find_card() -> Path:
    cards = sorted((ROOT / "docs").glob("PROD_NEW_TEST_CARD_*.md"),
                   key=lambda p: p.stat().st_mtime)
    if not cards:
        raise ContractError("找不到 docs/PROD_NEW_TEST_CARD_*.md")
    return cards[-1]


def load_contract(card: Path | None = None) -> dict:
    card = card or find_card()
    text = card.read_text(encoding="utf-8")
    m = re.search(r"```json\n(\{.*?\})\n```", text, re.S)
    if not m:
        raise ContractError(f"{card.name} 沒有 machine-readable JSON block")
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ContractError(f"{card.name} JSON 解析失敗: {e}") from e
    for k in ("schema", "cell"):
        if k not in d:
            raise ContractError(f"{card.name} block 缺 {k}")
    return d


def cell_names(contract: dict) -> list[str]:
    """The selectable names: the default cell plus every named one."""
    return ["(default)"] + sorted((contract.get("cells") or {}).keys())


def retired_cells(contract: dict) -> dict:
    """已退役的格名 -> 墓碑（`{"prompt": N, "retired_at": ..., "why": ...}`）。

    WHY THE NAMES SURVIVE THE CELLS (2026-10-03). Operator 裁定的原文是「把所有的格移除」——
    移除的是**可選的格**，不是紀錄裡的名字。三件東西還要靠那些名字才讀得懂：
    `caliber_gate` 要分得出「未知格名（錯誤）」與「已退役（歷史）」；`quote_gate` 的 R2
    pp-less 拆欄要墓碑裡的 `prompt`（`_declared_prompt`）；審計要能回答「這一筆當年是哪一格」。
    所以墓碑只保留名字與 `prompt`／日期／理由 —— 它**不是**格定義，`resolve_cell` 一律拒收。
    """
    rc = contract.get("retired_cells")
    if rc is not None and not isinstance(rc, dict):
        raise ContractError("`retired_cells` 必須是 mapping（name -> 墓碑），得到 %s"
                            % type(rc).__name__)
    return rc or {}


def retired_names(contract: dict) -> list[str]:
    return sorted(retired_cells(contract).keys())


def cell_spec(contract: dict, name: str) -> dict | None:
    """唯讀查格：預設格／在役具名格／退役墓碑都找得到；找不到回 None。

    與 `resolve_cell` 的差別是**它不判可選性**：稽核要讀一個舊格的 `prompt`，那件事與
    「這一格還能不能再跑」無關。
    """
    if name == "(default)":
        return contract.get("cell")
    live = (contract.get("cells") or {}).get(name)
    if isinstance(live, dict):
        return live
    dead = retired_cells(contract).get(name)
    return dead if isinstance(dead, dict) else None


def resolve_cell(contract: dict, cell_name: str | None) -> tuple[str, dict]:
    """(label, cell) for the cell the caller asked for.

    WHY A MISSING NAME IS FAIL-CLOSED RATHER THAN A FALLBACK: a second cell had to exist at all
    because `prod_profile.py` could not measure the delivery cell -- the contract declared ONE
    cell (the prod-new host-preFill cell: batch/ubatch 5632, prompt 2048) and refused everything
    else. Adding the delivery cell must not turn "the cell you named is not declared" into
    "here is a different cell's bar": that is how a naming mistake becomes a plausible number
    from the wrong cell. Set default unchanged: `cell_name=None` still means `contract["cell"]`.

    2026-10-03: 所有具名格退役 ⇒ 指名一個退役格得到的是一個**指名原因的拒絕**（不是 fallback、
    也不是和「從來沒有這一格」同一句話）——「那格沒了」與「你打錯字」是兩種病。
    """
    # The type check lives HERE, not only in `load_contract`: a validation in the loader is
    # skipped by every caller that passes an explicit `contract=`, which is exactly how the
    # selftest caught it. Validate where the value is used.
    cells = contract.get("cells")
    if cells is not None and not isinstance(cells, dict):
        raise ContractError("`cells` 必須是 mapping（name -> cell），得到 %s" % type(cells).__name__)
    # The rep-split declaration is validated on EVERY path (including the default cell), for the
    # same reason the type check above lives here: a validation that only one caller runs is not a
    # contract. A broken declaration must not be readable as "this cell has no session shape".
    errs = validate_rep_split(contract) + validate_reps_twins(contract)
    if errs:
        raise ContractError("測試卡 §2.5 的孿生宣告有問題（fail-closed）：\n   - "
                            + "\n   - ".join(errs))
    if cell_name is None or cell_name == "(default)":
        # [CGC 2026-10-07] `cell_names()` has always LISTED "(default)" as selectable, so a caller
        # that did the only reasonable thing -- read the list, then pass back the name it was given
        # -- was rejected by the line below. `hotkv-d512` (the other declared name) worked, which
        # made it look like `(default)` specifically was unavailable rather than unnamed-vs-named
        # being two spellings of one cell. `None` and "(default)" are the same request; treating
        # them differently is a trap in the selector, not a decision about the caliber.
        return "(default)", contract["cell"]
    if cell_name in retired_cells(contract):
        _t = retired_cells(contract).get(cell_name) or {}
        raise ContractError(
            "測試卡 §2.5 的 cell %r 已退役（retired_at=%s）。operator 裁定：移除所有具名格，"
            "prod-new + harness bench 貼齊 llama-bench 出廠形狀 ⇒ 它只存在於紀錄裡、不可再選；"
            "現在唯一可選的是 '(default)'。" % (cell_name, _t.get("retired_at") or "?"))
    named = (cells or {}).get(cell_name)
    if named is None:
        raise ContractError(
            "測試卡 §2.5 沒有 cell %r；已宣告的是 %s。不退回預設——退回會讓一次命名錯誤"
            "變成另一個 cell 的數字。" % (cell_name, ", ".join(cell_names(contract))))
    return cell_name, named


# ── rep-split cells：一次啟動一個 rep 的孿生 cell ────────────────────────────────────────
#
# WHY A TWIN CELL AND NOT "reps becomes adjustable": `reps` is a STRICT dimension of both
# authoritative cells and it has to stay that way. What was measured is that the reps inside ONE
# llama-bench launch are not interchangeable -- the draft chain degrades from the second measured
# rep on (measured 2026-09-28: `X = 642` vs `Y = 512` at the rep boundary, reps 2/3 measured plain
# decode while the arm still said k=3, docs/MTP_DRAFT_FIRST_STEP_2026-09-28.md). Making `reps`
# adjustable would relax the bar for every caller, including the ones that never needed it.
#
# So the shape is DECLARED instead. A rep-split cell is a twin of a declared cell, measured as N
# single-rep launches with a cooldown between them, and it may differ from its base in exactly two
# keys: `reps` (= 1) and `rep_split` itself. `launches x 1` must equal the base cell's `reps`, so a
# "3-rep" claim still costs three launches -- the twin cannot be a way to measure fewer reps.
REP_SPLIT_KEYS = ("of", "launches", "cool_to", "cool_max_s")

COOL_TO_CHOICES = ("NOMINAL",)


def validate_rep_split(contract: dict) -> list[str]:
    """Every problem with every `rep_split` block, as sentences. Empty list = the declaration is
    sound.

    Returned rather than raised so `resolve_cell` can report ALL of them at once: a half-fixed
    contract should not take three round trips to describe.
    """
    cells = contract.get("cells") or {}
    default = contract.get("cell") or {}
    errs: list[str] = []

    for name, cell in cells.items():
        rs = cell.get("rep_split")
        if rs is None:
            continue
        if not isinstance(rs, dict):
            errs.append(f"{name}: rep_split 必須是 mapping，得到 {type(rs).__name__}")
            continue

        unknown = sorted(k for k in rs if k not in REP_SPLIT_KEYS)
        if unknown:
            errs.append(f"{name}: rep_split 有不認識的鍵 {unknown}（只接受 {list(REP_SPLIT_KEYS)}）")

        # 1) it must name a base cell that EXISTS (a missing base is the same failure as naming a
        #    missing cell: it silently means "some other cell's bar").
        base_name = rs.get("of")
        if base_name is None:
            errs.append(f"{name}: rep_split 缺 of —— 必須指名它是哪個 cell 的孿生")
            base = None
        elif base_name == "(default)":
            base = default
        else:
            base = cells.get(base_name)
            if base is None:
                errs.append(f"{name}: rep_split.of={base_name!r} 不存在；已宣告的是 "
                            f"{['(default)'] + sorted(cells)}")

        # 2) the session shape.
        launches = rs.get("launches")
        if not isinstance(launches, int) or isinstance(launches, bool) or launches < 1:
            errs.append(f"{name}: rep_split.launches 必須是 >=1 的整數，得到 {launches!r}")
        cool_to = rs.get("cool_to")
        if cool_to not in COOL_TO_CHOICES:
            errs.append(f"{name}: rep_split.cool_to 只接受 {list(COOL_TO_CHOICES)}（本 repo 唯一被"
                        f"定義的量測水平；其它值等於宣告一個沒人會執行的冷卻），得到 {cool_to!r}")
        cool_max = rs.get("cool_max_s")
        if not isinstance(cool_max, (int, float)) or isinstance(cool_max, bool) or cool_max <= 0:
            errs.append(f"{name}: rep_split.cool_max_s 必須是 >0 的數字，得到 {cool_max!r}")

        # 3) `reps` must BE 1 -- that is the definition of the cell, not a detail of it.
        if _norm(cell.get("reps")) != 1:
            errs.append(f"{name}: rep-split cell 的 reps 必須是 1（得到 {cell.get('reps')!r}）—— "
                        f"「一次啟動一個 rep」就是這個 cell 的定義")

        # 4) the twin may differ from its base ONLY in reps/rep_split, and its launch count must add
        #    up to exactly the base's reps. This is what stops the twin from becoming a new shape.
        if base is not None:
            want = {k: v for k, v in base.items() if k != "rep_split"}
            want["reps"] = 1
            got = {k: v for k, v in cell.items() if k != "rep_split"}
            diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
            if diff:
                errs.append(f"{name}: 孿生 cell 只准與 {base_name!r} 差 reps/rep_split，"
                            f"但 {diff} 也不同")
            else:
                base_reps = _norm(base.get("reps"))
                if isinstance(launches, int) and base_reps is not None and launches * 1 != base_reps:
                    errs.append(f"{name}: launches={launches} x reps=1 != "
                                f"{base_name!r} 的 reps={base_reps} —— 孿生不可用來量比較少的 rep")
    return errs


# ── reps 孿生（twin_of）：只把 n 加大的同形 cell ─────────────────────────────────────────────
#
# WHY A DECLARED TWIN AND NOT "reps becomes adjustable" (2026-10-01, L20-10): the pair
# `delivery` / `delivery-ws192` was finally measured in one clean window -- both arms quotable --
# and the verdict was still REFUSE: the effect (+4.8%, 3.989 ms/step) was SMALLER than the two
# arms' own per-rep spread (7.515 ms). `reps` is a strict dimension of both cells and stays strict
# (it is part of the cell's identity); a RESOLUTION problem needs a declared cell with a bigger n,
# not a looser bar for every caller. So a cell may name its base with `twin_of`: it is then a reps
# twin that may differ from that base in exactly ONE key, `reps`, and only upward -- a "fewer
# reps" twin would be a way to measure less, the same thing the rep_split rule forbids.
TWIN_KEY = "twin_of"


def validate_reps_twins(contract: dict) -> list[str]:
    """Every problem with every `twin_of` declaration, as sentences. Empty list = sound.

    Returned rather than raised so `resolve_cell` can report ALL of them at once: a half-fixed
    contract should not take three round trips to describe.
    """
    cells = contract.get("cells") or {}
    default = contract.get("cell") or {}
    errs: list[str] = []

    if default.get(TWIN_KEY) is not None:
        errs.append(f"(default): cell 本體不能宣告 {TWIN_KEY}（孿生是命名 cell 的事；預設格是"
                    f"所有人的共同底，不是誰的孿生）")

    for name, cell in cells.items():
        raw = cell.get(TWIN_KEY)
        if raw is None:
            continue
        if not isinstance(raw, str) or not raw.strip():
            errs.append(f"{name}: {TWIN_KEY} 必須是基底 cell 的名稱（字串），得到 {raw!r}")
            continue
        base_name = raw.strip()
        base = default if base_name == "(default)" else cells.get(base_name)
        if base is None:
            errs.append(f"{name}: {TWIN_KEY}={base_name!r} 不存在；已宣告的是 "
                        f"{['(default)'] + sorted(cells)}")
            continue

        # Compare on normalized values (`"512"` == 512), leaving the two declaration keys out:
        # `rep_split` has its own validator above, and `twin_of` is what makes this a twin --
        # requiring it to equal the base's would make the declaration impossible to write.
        want = {k: _norm(v) for k, v in base.items() if k not in ("rep_split", TWIN_KEY)}
        got = {k: _norm(v) for k, v in cell.items() if k not in ("rep_split", TWIN_KEY)}
        diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        if not diff:
            errs.append(f"{name}: 與基底 {base_name!r} 逐字相同 —— 那是一個別名，不是孿生；"
                        f"孿生要宣告它改了什麼（只准 `reps`）")
            continue
        if diff != ["reps"]:
            errs.append(f"{name}: reps 孿生只准與 {base_name!r} 差 `reps` 一項，"
                        f"但差的是 {diff} —— 別形狀要走別的宣告")
            continue

        reps, base_reps = got.get("reps"), want.get("reps")
        if not isinstance(reps, int) or isinstance(reps, bool) or reps < 2:
            errs.append(f"{name}: reps 孿生的 reps 必須是 >=2 的整數，得到 {cell.get('reps')!r}")
        elif isinstance(base_reps, int) and not isinstance(base_reps, bool) and reps <= base_reps:
            errs.append(f"{name}: reps 孿生只能加大（它是為了解析度 ⇒ 更多樣本），"
                        f"得到 {reps} ≤ 基底 {base_name!r} 的 {base_reps}")
    return errs


def rep_split_plan(contract: dict, cell_name: str) -> dict | None:
    """The session shape for `cell_name`, or None when that cell is not rep-split.

    The orchestrator (`rep_split.py`) is the only caller: it needs the numbers, and it must not
    re-derive them from the cell dict (two readers of one declaration is how they drift).
    """
    label, cell = resolve_cell(contract, cell_name)
    rs = cell.get("rep_split")
    if not rs:
        return None
    return {"name": label, "of": rs["of"], "launches": int(rs["launches"]),
            "reps_per_launch": 1, "cool_to": rs["cool_to"], "cool_max_s": float(rs["cool_max_s"])}


@dataclass
class Report:
    ok: bool
    matches: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    declared: list[str] = field(default_factory=list)
    # WHICH cell was checked. Without it, two different cells' verdicts read identically in the
    # artifact -- and a number is only comparable to another number from the same cell.
    cell: str = "(default)"

    def render(self) -> str:
        lines = []
        if self.mismatches:
            lines.append(f"⛔ cell 口徑與測試卡權威 block 不一致（cell={self.cell}）— 拒跑（fail-closed）：")
            lines += [f"   - {x}" for x in self.mismatches]
            lines.append("   請走生產入口（harness bench / commit_bench）的預設，或顯式對齊測試卡 §2.5。")
        if self.declared:
            lines.append("（可運行調整、需記錄在產物、不阻塞）：")
            lines += [f"   · {x}" for x in self.declared]
        if not self.mismatches:
            lines.append(f"cell 口徑校驗通過（cell={self.cell}，嚴格維度 {len(self.matches)} 項一致）。")
        return "\n".join(lines)


def _norm(v):
    # 數字字串以 int 比較（"5632" == 5632）；其餘原值；None 保留。
    if isinstance(v, str):
        s = v.strip()
        if s and s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass
        return s
    return v


def check_cell(actual: dict, arm_env: dict | None = None,
               contract: dict | None = None, cell_name: str | None = None) -> Report:
    contract = contract or load_contract()
    if "cell" not in contract:
        raise ContractError("block 缺 cell")
    label, cell = resolve_cell(contract, cell_name)
    adjustable = set(contract.get("runtime_adjustable", []))
    matches, mismatches, declared = [], [], []

    for key, expected in cell.items():
        if key in ("rep_split", TWIN_KEY):
            # NOT dimensions. `rep_split` declares how many launches the cell is measured in and
            # `twin_of` which cell this one is a twin of; the validators above enforce both.
            # Comparing either against `actual` would reject every legitimate launch for a key
            # llama-bench never sees.
            continue
        if key not in actual:
            # 缺維度 ⇒ 嚴格 fail-closed，不把缺席當成預設。
            mismatches.append(f"{key}: 實際值缺席（block 要求 {expected!r}）")
            continue
        av, ex = _norm(actual[key]), _norm(expected)
        if av == ex:
            matches.append(key)
        elif key in adjustable:
            declared.append(f"{key}: {ex!r} → {av!r}")
        else:
            mismatches.append(f"{key}: 實際 {av!r} ≠ 權威 {ex!r}")

    return Report(ok=not mismatches, matches=matches,
                  mismatches=mismatches, declared=declared, cell=label)


# ───────────────────── selftest ─────────────────────

def selftest() -> int:
    chk: list[tuple[str, bool]] = []

    def c(n: str, ok: bool) -> None:
        chk.append((n, bool(ok)))

    ctr = load_contract()
    cell = ctr["cell"]

    r = check_cell(dict(cell), contract=ctr)
    c("完全一致 -> ok 且無 mismatch", r.ok and not r.mismatches)

    # 由格自己宣告的值推導（2026-10-03：預設格的 warm_skip 由 64 改為 0，
    # 寫死 64 的自測會跟著紅 —— 自測要驗「偏離就拒」，不是驗某個歷史數值）。
    _ws = int(cell["warm_skip"])
    run3 = dict(cell); run3["warm_skip"] = _ws + 64
    r3 = check_cell(run3, contract=ctr)
    c("warm_skip 偏離格宣告 -> 拒、mismatch 含 warm_skip",
      not r3.ok and any("warm_skip" in m for m in r3.mismatches))

    h = dict(cell); h["reps"] = 1
    rh = check_cell(h, contract=ctr)
    c("reps=1（harness 舊預設）-> 拒",
      not rh.ok and any("reps" in m for m in rh.mismatches))

    cx = dict(cell); cx["ctx_size"] = 8192
    rc = check_cell(cx, contract=ctr)
    c("ctx_size=8192 -> 放行、declared",
      rc.ok and any("ctx_size" in d for d in rc.declared))

    fs = dict(cell); fs["fixed_fill_seed"] = 123
    rf = check_cell(fs, contract=ctr)
    c("fixed_fill_seed=123 -> 放行、declared",
      rf.ok and any("fixed_fill_seed" in d for d in rf.declared))

    rp0 = check_cell(dict(cell), arm_env={"CGC_EXPERT_SKIP_READRAW": "1"}, contract=ctr)
    c("P0 arm + cell 一致 -> ok（臂開關不改 cell）", rp0.ok)

    b = dict(cell); b["batch"] = 4096
    rb = check_cell(b, contract=ctr)
    c("batch=4096 -> 拒", not rb.ok and any("batch" in m for m in rb.mismatches))

    miss = dict(cell); miss.pop("gen")
    rm = check_cell(miss, contract=ctr)
    c("缺 gen -> 拒（不把缺席當值）",
      not rm.ok and any("gen" in m for m in rm.mismatches))

    rs = dict(cell); rs["warm_skip"] = str(_ws)
    c("warm_skip 以字串給 == 權威值 -> ok", check_cell(rs, contract=ctr).ok)

    # ── 退役（2026-10-03：所有具名格退役，名字留墓碑）─────────────────────────────────────
    # 這三條是這次改動的判準：退役名**不可選**，但**讀得到**（稽核與 quote_gate 的 pp-less 要用）。
    _ctr2 = dict(ctr, retired_cells={"delivery": {"prompt": 0, "retired_at": "2026-10-03",
                                                  "why": "測試用"}})
    c("退役名不在可選清單裡", "delivery" not in cell_names(_ctr2))
    try:
        resolve_cell(_ctr2, "delivery")
        _rr = None
    except ContractError as e:
        _rr = str(e)
    c("指名退役格 -> 拒、且訊息明說『已退役』（不是『從來沒有這一格』）",
      _rr is not None and "已退役" in _rr)
    c("墓碑唯讀讀得到（prompt 供 pp-less 拆欄）",
      (cell_spec(_ctr2, "delivery") or {}).get("prompt") == 0)
    c("(default) 仍走預設 block", cell_spec(_ctr2, "(default)") == _ctr2["cell"])
    c("從來不存在的格名 still 拒", cell_spec(_ctr2, "no-such-cell") is None)

    try:
        check_cell(dict(cell), contract={"schema": "x"})
        raised = False
    except ContractError:
        raised = True
    c("壞 block（缺 cell）-> ContractError fail-closed", raised)

    # ── named cells ─────────────────────────────────────────────────────────────────────────
    c("預設路徑會把自己標成 (default)", check_cell(dict(cell), contract=ctr).cell == "(default)")

    named = ctr.get("cells") or {}
    if named:
        for nm, ncell in named.items():
            rn = check_cell(dict(ncell), contract=ctr, cell_name=nm)
            c(f"named cell {nm!r} 可用且自我標名", rn.ok and rn.cell == nm)
            # A named cell must also reject the OTHER cell's numbers -- otherwise the second
            # cell is just a second way to say yes.
            other = dict(ncell)
            other["prompt"] = cell["prompt"]
            if other["prompt"] != ncell.get("prompt"):
                c(f"named cell {nm!r} 會拒絵預設 cell 的 prompt",
                  not check_cell(other, contract=ctr, cell_name=nm).ok)
    else:
        c("（本卡尚未宣告 named cell；選擇器仍可測）", True)

    # ── rep-split cells ─────────────────────────────────────────────────────────────────────
    # A synthetic card, so these cases exercise the RULES rather than today's test card.
    _base = {"ngl": 99, "prompt": 0, "gen": 128, "reps": 3, "warm_skip": 64, "ctx_size": 4096}

    def _synth2(cell_over=None, rs_over=None, cell_name="twin", base_cell=None):
        base_cell = dict(base_cell or _base)
        rs = {"of": "plain", "launches": base_cell.get("reps", 3), "cool_to": "NOMINAL",
              "cool_max_s": 420}
        rs.update(rs_over or {})
        twin = dict(base_cell)
        twin["reps"] = 1
        twin.update(cell_over or {})
        twin["rep_split"] = rs
        return {"schema": "t", "cell": dict(base_cell),
                "cells": {"plain": base_cell, cell_name: twin}}

    def _err_of(ctr2):
        try:
            check_cell(dict(ctr2["cell"]), contract=ctr2)
            return ""
        except ContractError as e:
            return str(e)

    ok_c = _synth2()
    c("rep-split 孿生（launches==base reps、只差 reps/rep_split）-> 宣告通過",
      validate_rep_split(ok_c) == [])
    pl = rep_split_plan(ok_c, "twin")
    c("plan: launches=3、每啟動 1 rep、cool_to=NOMINAL",
      bool(pl) and pl["launches"] == 3 and pl["reps_per_launch"] == 1 and pl["cool_to"] == "NOMINAL")
    _tw = {k: v for k, v in ok_c["cells"]["twin"].items() if k != "rep_split"}
    c("孿生的維度以其名通過校驗", check_cell(dict(_tw), contract=ok_c, cell_name="twin").ok)
    c("同一組維度用預設 cell 名 -> 拒（孿生不是全域放寬）",
      not check_cell(dict(_tw), contract=ok_c).ok)
    c("非 rep-split cell 的 plan 是 None", rep_split_plan(ok_c, "plain") is None)

    _bad = [
        ("launches 與 base reps 不符", _synth2(rs_over={"launches": 1}), "launches"),
        ("孿生的 reps=2", _synth2(cell_over={"reps": 2}), "reps 必須是 1"),
        ("孿生改了別的維度（batch）", _synth2(cell_over={"batch": 512}), "孿生 cell 只准"),
        ("cool_to 不是 NOMINAL", _synth2(rs_over={"cool_to": "MODERATE"}), "cool_to"),
        ("cool_max_s 壞值", _synth2(rs_over={"cool_max_s": "no"}), "cool_max_s"),
        ("rep_split 有不認識的鍵", _synth2(rs_over={"bogus": 1}), "不認識"),
        ("of 指向不存在的 cell", _synth2(rs_over={"of": "nope"}), "不存在"),
        ("rep_split 不是 mapping", _synth2(rs_over=None), None),
    ]
    for label, ctr2, needle in _bad[:-1]:
        msg = _err_of(ctr2)
        c(f"壞宣告：{label} -> ContractError 且指名原因",
          bool(msg) and (needle in msg if needle else True))
    _nm = _synth2()
    _nm["cells"]["twin"]["rep_split"] = "yes"
    c("壞宣告：rep_split 不是 mapping -> ContractError", bool(_err_of(_nm)))

    try:
        check_cell(dict(cell), contract=ctr, cell_name="no-such-cell")
        raised2 = False
    except ContractError:
        raised2 = True
    c("指名不存在的 cell -> ContractError（不退回預設）", raised2)

    try:
        check_cell(dict(cell), contract={"schema": "x", "cell": cell, "cells": []})
        raised3 = False
    except ContractError:
        raised3 = True
    c("cells 不是 mapping -> ContractError", raised3)

    # ── reps 孿生（twin_of）────────────────────────────────────────────────────────────────────
    # Same shape as the rep-split cases: a synthetic card, so these exercise the RULES rather than
    # today's test card; plus one pass over the REAL card (where the rule has to hold too).
    _rbase = {"ngl": 99, "prompt": 0, "gen": 128, "reps": 3, "warm_skip": 64, "ctx_size": 4096}

    def _rtwin(cell_over=None, base_over=None, twin="plain"):
        base = dict(_rbase)
        base.update(base_over or {})
        tw = dict(base)
        tw["reps"] = 7
        tw[TWIN_KEY] = twin
        tw.update(cell_over or {})
        return {"schema": "t", "cell": dict(base), "cells": {"plain": base, "twin": tw}}

    r_ok = _rtwin()
    c("reps 孿生（只差 reps 3→7）-> 宣告通過", validate_reps_twins(r_ok) == [])
    _rt = {k: v for k, v in r_ok["cells"]["twin"].items() if k != TWIN_KEY}
    c("reps 孿生的維度以其名通過校驗", check_cell(dict(_rt), contract=r_ok, cell_name="twin").ok)
    c("同一組維度用基底名 -> 拒（孿生不是全域放寬）",
      not check_cell(dict(_rt), contract=r_ok, cell_name="plain").ok)

    _bad_tw = [
        ("twin_of 指向不存在的基底", _rtwin(twin="nope"), "不存在"),
        ("與基底逐字相同", _rtwin(cell_over={"reps": 3}), "逐字相同"),
        ("同時改了別的維度（batch）", _rtwin(cell_over={"batch": 512}), "只准"),
        ("reps 沒加大（5 ≤ 基底 7）", _rtwin(base_over={"reps": 7}, cell_over={"reps": 5}), "只能加大"),
        ("reps 壞型別", _rtwin(cell_over={"reps": "seven"}), ">=2 的整數"),
        ("twin_of 不是字串", _rtwin(twin=7), TWIN_KEY),
    ]
    for label, ctr2, needle in _bad_tw:
        msg = _err_of(ctr2)
        c(f"壞宣告：{label} -> ContractError 且指名原因", bool(msg) and needle in msg)
    _def_tw = _rtwin()
    _def_tw["cell"][TWIN_KEY] = "plain"
    c("預設 cell 宣告 twin_of -> 拒（孿生是命名 cell 的事）",
      bool(_err_of(_def_tw)) and "不能宣告" in _err_of(_def_tw))

    _rt_real = validate_reps_twins(ctr)
    _rt_names = [nm for nm, nc in (ctr.get("cells") or {}).items() if nc.get(TWIN_KEY)]
    c("測試卡上所有 reps 孿生（%d 格：%s）驗證通過"
      % (len(_rt_names), "／".join(_rt_names) or "無"), _rt_real == [])

    for n, ok in chk:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"selftest {sum(ok for _, ok in chk)}/{len(chk)}")
    return 0 if all(ok for _, ok in chk) else 1


# Which bench-side flags a declared cell demands. `harness bench` (and llama-bench) defaults are
# the DEFAULT cell's dimensions (batch/ubatch 5632, prompt 2048, ctx 0, warm-skip 64), so naming a
# cell without passing its own numbers is refused fail-closed -- measured 2026-09-30: the two
# L20-10 arms burned their box window on exactly that ("batch: 實際 5632 ≠ 權威 256/512").
# Printing the flags is the cheap fix: a refusal is safe but it is not free.
_BENCH_FLAG_KEYS = (("batch", "--batch"), ("prompt", "--prompt"), ("gen", "--gen"),
                    ("depths", "--depths"), ("reps", "--reps"),
                    ("warm_skip", "--warm-skip"), ("ctx_size", "--ctx-size"))


def bench_flags(cell: dict) -> tuple[list[str], list[str]]:
    """(flags, warnings) for `harness bench --cell <name>`."""
    flags, warn = [], []
    for key, flag in _BENCH_FLAG_KEYS:
        if key in cell and cell[key] is not None:
            flags += [flag, str(cell[key])]
    if cell.get("ubatch") is not None and cell.get("ubatch") != cell.get("batch"):
        warn.append(f"cell 宣告 ubatch={cell['ubatch']} ≠ batch={cell['batch']}，但 `harness bench` 的 "
                    f"--batch 同時設 -b 與 -ub ⇒ 這個 cell 目前無法從命令列表逹（先改 cell 或加旗標）。")
    return flags, warn


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cell", default=None,
                    help="列出這個 §2.5 cell 權威 block 所要求的 harness-bench 旗標")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    contract = load_contract()
    if args.cell:
        name, cell = resolve_cell(contract, args.cell)
        flags, warn = bench_flags(cell)
        print(f"# cell {name}（{contract['schema']}）——權威 block 要求的旗標：")
        print("python3 scripts/check/harness.py bench --charter <card> --arm prod-new "
              f"--cell {name} " + " ".join(flags) + " --json <out.json>")
        for w in warn:
            print("!! " + w, file=__import__("sys").stderr)
        return 1 if warn else 0
    print(contract["schema"])
    return 0


if __name__ == "__main__":
    sys = __import__("sys")
    sys.exit(main(sys.argv[1:]))
