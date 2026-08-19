import json
from pathlib import Path
from typing import Any, Dict


def _load(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verify_both(*, edge_report: str, cloud_report: str, output_path: str) -> Dict[str, Any]:
    e = _load(edge_report)
    c = _load(cloud_report)
    e_ok = bool(e.get("ok"))
    c_ok = bool(c.get("ok"))
    ok = bool(e_ok and c_ok)
    rep = {
        "ok": ok,
        "milestone": "m6",
        "verify": {
            "edge_report": str(edge_report),
            "cloud_report": str(cloud_report),
            "edge_ok": e_ok,
            "cloud_ok": c_ok,
        },
        "gate_result": {
            "m6": {
                "status": "PASS" if ok else "FAIL",
                "require_both_gate": {
                    "status": "PASS" if ok else "FAIL",
                    "edge_ok": e_ok,
                    "cloud_ok": c_ok,
                },
            }
        },
    }
    Path(output_path).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep

