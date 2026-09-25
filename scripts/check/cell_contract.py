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


@dataclass
class Report:
    ok: bool
    matches: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    declared: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = []
        if self.mismatches:
            lines.append("⛔ cell 口徑與測試卡權威 block 不一致 — 拒跑（fail-closed）：")
            lines += [f"   - {x}" for x in self.mismatches]
            lines.append("   請走生產入口（harness bench / commit_bench）的預設，或顯式對齊測試卡 §2.5。")
        if self.declared:
            lines.append("（可運行調整、需記錄在產物、不阻塞）：")
            lines += [f"   · {x}" for x in self.declared]
        if not self.mismatches:
            lines.append(f"cell 口徑校驗通過（嚴格維度 {len(self.matches)} 項一致）。")
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
               contract: dict | None = None) -> Report:
    contract = contract or load_contract()
    if "cell" not in contract:
        raise ContractError("block 缺 cell")
    cell = contract["cell"]
    adjustable = set(contract.get("runtime_adjustable", []))
    matches, mismatches, declared = [], [], []

    for key, expected in cell.items():
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
                  mismatches=mismatches, declared=declared)


# ───────────────────── selftest ─────────────────────

def selftest() -> int:
    chk: list[tuple[str, bool]] = []

    def c(n: str, ok: bool) -> None:
        chk.append((n, bool(ok)))

    ctr = load_contract()
    cell = ctr["cell"]

    r = check_cell(dict(cell), contract=ctr)
    c("完全一致 -> ok 且無 mismatch", r.ok and not r.mismatches)

    run3 = dict(cell); run3["warm_skip"] = 0
    r3 = check_cell(run3, contract=ctr)
    c("warm_skip=0（run3）-> 拒、mismatch 含 warm_skip",
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

    rs = dict(cell); rs["warm_skip"] = "64"
    c("'64' 字串 == 64 -> ok", check_cell(rs, contract=ctr).ok)

    try:
        check_cell(dict(cell), contract={"schema": "x"})
        raised = False
    except ContractError:
        raised = True
    c("壞 block（缺 cell）-> ContractError fail-closed", raised)

    for n, ok in chk:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"selftest {sum(ok for _, ok in chk)}/{len(chk)}")
    return 0 if all(ok for _, ok in chk) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    print(load_contract()["schema"])
    return 0


if __name__ == "__main__":
    sys = __import__("sys")
    sys.exit(main(sys.argv[1:]))
