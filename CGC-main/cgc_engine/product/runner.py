import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_numpy(arr: Any) -> str:
    b = arr.tobytes(order="C")
    return _sha256_bytes(b)


def _hash_outputs(outputs: Any) -> str:
    h = hashlib.sha256()
    for o in outputs:
        h.update(o.tobytes(order="C"))
    return h.hexdigest()


def _make_feed(sess: Any) -> Dict[str, Any]:
    import numpy as np

    feed: Dict[str, Any] = {}
    for inp in sess.get_inputs():
        shape = []
        for d in (inp.shape or []):
            if isinstance(d, int) and d > 0:
                shape.append(int(d))
            else:
                shape.append(1)
        dtype = inp.type
        if "float16" in dtype:
            dt = np.float16
        elif "float" in dtype:
            dt = np.float32
        elif "int64" in dtype:
            dt = np.int64
        elif "int32" in dtype:
            dt = np.int32
        else:
            dt = np.float32
        feed[inp.name] = np.zeros(tuple(shape), dtype=dt)
    return feed


def run_bundle(*, output_dir: str) -> Dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    bundle_dir = out_dir / "bundle"
    cfg_path = bundle_dir / "bundle_config.json"
    manifest_path = bundle_dir / "bundle_manifest.json"

    if not cfg_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("missing_bundle: build first")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = str(manifest.get("bundle", {}).get("model_path", "") or "").strip()
    if not model_path:
        raise ValueError("missing_model_path_in_manifest")

    ep = str(cfg.get("ep", "") or "CPUExecutionProvider").strip()
    db_path = bundle_dir / "state.sqlite3"

    from ..ort_state import ORTStateStore

    store = ORTStateStore(str(db_path))
    model_sha256 = ORTStateStore.sha256_file(model_path)

    import onnxruntime as ort

    providers = [ep]
    sess = ort.InferenceSession(model_path, providers=providers)
    feed = _make_feed(sess)
    input_hash = hashlib.sha256(json.dumps({k: _hash_numpy(v) for k, v in feed.items()}, sort_keys=True).encode("utf-8")).hexdigest()

    cached = store.get_cached(model_sha256=model_sha256, ep=ep, input_hash=input_hash)
    if cached is not None:
        first = {"cache_hit": True, "output_hash": str(cached["output_hash"])}
    else:
        outputs = sess.run(None, feed)
        out_hash = _hash_outputs(outputs)
        store.put(model_sha256=model_sha256, ep=ep, input_hash=input_hash, output_hash=out_hash, outputs={"output_hash": out_hash})
        first = {"cache_hit": False, "output_hash": str(out_hash)}

    cached2 = store.get_cached(model_sha256=model_sha256, ep=ep, input_hash=input_hash)
    if cached2 is None:
        second = {"cache_hit": False, "output_hash": ""}
    else:
        second = {"cache_hit": True, "output_hash": str(cached2["output_hash"])}

    ok = (first["output_hash"] != "") and bool(second["cache_hit"]) and (second["output_hash"] == first["output_hash"])

    gate = {
        "status": "PASS" if ok else "FAIL",
        "bundle_dir": str(bundle_dir),
        "model_path": str(model_path),
        "model_sha256": str(model_sha256),
        "ep": str(ep),
        "state_db_path": str(db_path),
        "input_hash": str(input_hash),
        "first": first,
        "second": second,
        "config_sha256": _sha256_bytes(cfg_path.read_bytes()),
    }

    report = {
        "ok": bool(ok),
        "milestone": "m6",
        "steps": {"run_bundle": {"status": "PASS" if ok else "FAIL", "gate": gate}},
        "gate_result": {"m6": {"status": "PASS" if ok else "FAIL", "run_bundle_gate": gate}},
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

