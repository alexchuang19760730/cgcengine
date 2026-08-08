import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _exists(p: Path) -> Tuple[bool, str]:
    return (p.exists(), str(p))


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_model_dir(model_dir: Path) -> Dict:
    required = [
        "config.json",
    ]
    optional_any = [
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
    ]

    results: Dict[str, object] = {"model_dir": str(model_dir), "ok": True, "checks": {}}
    checks: Dict[str, object] = {}

    for name in required:
        ok, path = _exists(model_dir / name)
        checks[name] = {"exists": ok, "path": path}
        if not ok:
            results["ok"] = False

    present_any = []
    for name in optional_any:
        ok, path = _exists(model_dir / name)
        if ok:
            present_any.append({"name": name, "path": path})
    checks["weights_presence"] = {"present_any": present_any, "required_any_of": optional_any}
    if not present_any:
        results["ok"] = False

    if (model_dir / "model.safetensors.index.json").exists():
        index = _read_json(model_dir / "model.safetensors.index.json")
        weight_map = index.get("weight_map", {})
        files = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
        missing = []
        for f in files:
            if not (model_dir / f).exists():
                missing.append(str(f))
        checks["safetensors_index"] = {
            "files_listed": len(files),
            "missing_files": missing,
        }
        if missing:
            results["ok"] = False

    results["checks"] = checks
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    report = verify_model_dir(model_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

