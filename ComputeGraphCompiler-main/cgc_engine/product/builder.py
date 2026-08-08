import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from .templates import resolve_template
from ..ort_state import ORTStateStore


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return ORTStateStore.sha256_file(str(path))


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for i in range(8):
        try:
            req = Request(str(url), headers={"User-Agent": "cgc-engine"})
            with urlopen(req, timeout=120) as r:
                data = r.read()
            tmp = dst.with_suffix(dst.suffix + ".partial")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return
        except Exception as e:
            last_err = e
            time.sleep(float(min(10, 1 + i)))
    raise last_err if last_err is not None else RuntimeError("download_failed")


def build_bundle(
    *,
    output_dir: str,
    template: Optional[str] = None,
    ort_model_path: str = "",
    ort_model_url: str = "",
) -> Dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl = resolve_template(template)
    bundle_dir = out_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = bundle_dir / "bundle_config.json"
    cfg_path.write_text(json.dumps(tpl.data, ensure_ascii=False, indent=2), encoding="utf-8")

    model_info = dict(tpl.data.get("model", {}) or {})
    model_local = str(ort_model_path or model_info.get("path", "") or "").strip()
    model_url = str(ort_model_url or model_info.get("url", "") or "").strip()
    model_name = str(model_info.get("filename", "") or "model.onnx").strip()

    downloaded = False
    model_path = ""
    if model_local:
        mp = Path(model_local).expanduser().resolve()
        if not mp.exists():
            raise FileNotFoundError(f"ort_model_not_found: {mp}")
        models_dir = bundle_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dst = models_dir / mp.name
        if dst.resolve() != mp:
            dst.write_bytes(mp.read_bytes())
        model_path = str(dst)
    elif model_url:
        dst = bundle_dir / "models" / model_name
        if not dst.exists():
            cache_dir = (Path.home() / ".cache" / "cgc_engine" / "m6_models").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = (cache_dir / model_name).resolve()
            if cache_path.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(cache_path.read_bytes())
            else:
                _download(model_url, dst)
                cache_path.write_bytes(dst.read_bytes())
                downloaded = True
        model_path = str(dst)
    else:
        raise ValueError("missing_ort_model: set template.model.url or pass --ort-model/--ort-model-url")

    model_sha256 = _sha256_file(Path(model_path))

    manifest = {
        "kind": "cgc_m6_bundle",
        "created_at": int(time.time()),
        "template": {"name": tpl.name, "path": tpl.path, "sha256": _sha256_bytes(cfg_path.read_bytes())},
        "bundle": {
            "bundle_dir": str(bundle_dir),
            "config_path": str(cfg_path),
            "model_path": str(model_path),
            "model_sha256": str(model_sha256),
            "downloaded": bool(downloaded),
        },
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = {
        "status": "PASS",
        "bundle_dir": str(bundle_dir),
        "config_path": str(cfg_path),
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "model_sha256": str(model_sha256),
    }

    report = {
        "ok": True,
        "milestone": "m6",
        "steps": {"build_bundle": {"status": "PASS", "gate": gate, "manifest": manifest}},
        "gate_result": {"m6": {"status": "PASS", "build_bundle_gate": gate}},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
