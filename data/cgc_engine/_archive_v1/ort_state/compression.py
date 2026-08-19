import hashlib
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compress_bytes(raw: bytes, *, algo: str) -> bytes:
    if algo == "zlib9":
        return zlib.compress(raw, level=9)
    raise ValueError(f"unsupported_algo:{algo}")


def _decompress_bytes(blob: bytes, *, algo: str) -> bytes:
    if algo == "zlib9":
        return zlib.decompress(blob)
    raise ValueError(f"unsupported_algo:{algo}")


def chunk_store_paths(store_dir: str) -> Dict[str, str]:
    base = Path(store_dir).resolve()
    chunks = base / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    return {"store_dir": str(base), "chunks_dir": str(chunks)}


def store_totals(store_dir: str) -> Dict[str, int]:
    p = chunk_store_paths(store_dir)
    chunks_dir = Path(p["chunks_dir"])
    total = 0
    count = 0
    for fp in chunks_dir.glob("*.bin"):
        try:
            st = fp.stat()
            total += int(st.st_size)
            count += 1
        except Exception:
            continue
    return {"unique_chunks": int(count), "total_bytes": int(total)}


class StateCompressor:
    def __init__(self, compression_level: int = 9):
        self.algo = f"zlib{compression_level}"

    def compress_file_to_store(
        self,
        *,
        input_path: str,
        store_dir: str,
        algo: str = "zlib9",
    ) -> Dict[str, Any]:
        p = Path(input_path).expanduser().resolve()
        raw = p.read_bytes()
        raw_bytes = int(len(raw))
        raw_hash = sha256_bytes(raw)
        blob = _compress_bytes(raw, algo=algo)
        compressed_bytes = int(len(blob))
        ratio = float(compressed_bytes) / float(max(1, raw_bytes))

        paths = chunk_store_paths(store_dir)
        chunks_dir = Path(paths["chunks_dir"])
        chunk_path = chunks_dir / f"{raw_hash}.{algo}.bin"
        existed = bool(chunk_path.exists())
        if not existed:
            chunk_path.write_bytes(blob)

        return {
            "status": "PASS",
            "algo": str(algo),
            "chunk_hash": str(raw_hash),
            "raw_sha256": str(raw_hash),
            "raw_bytes": int(raw_bytes),
            "compressed_bytes": int(compressed_bytes),
            "ratio": float(ratio),
            "chunk_path": str(chunk_path),
            "dedup_hit": bool(existed),
        }

    def restore_file_from_store(
        self,
        *,
        chunk_hash: str,
        store_dir: str,
        algo: str,
        output_path: str,
        expected_raw_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        paths = chunk_store_paths(store_dir)
        chunks_dir = Path(paths["chunks_dir"])
        chunk_path = chunks_dir / f"{str(chunk_hash)}.{str(algo)}.bin"
        if not chunk_path.exists():
            return {"status": "FAIL", "reason": "missing_chunk", "chunk_path": str(chunk_path)}

        blob = chunk_path.read_bytes()
        raw = _decompress_bytes(blob, algo=algo)
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(raw)
        out_sha = sha256_bytes(raw)
        ok = bool(expected_raw_sha256 is None or str(expected_raw_sha256) == str(out_sha))

        return {
            "status": "PASS" if ok else "FAIL",
            "output_path": str(out_p),
            "output_sha256": str(out_sha),
            "expected_sha256": str(expected_raw_sha256) if expected_raw_sha256 is not None else None,
        }


def replay_decompress_cost(
    *,
    chunk_hash: str,
    store_dir: str,
    algo: str,
    loops: int,
    deadline_ms: float,
) -> Tuple[Dict[str, Any], bytes]:
    paths = chunk_store_paths(store_dir)
    chunk_path = Path(paths["chunks_dir"]) / f"{str(chunk_hash)}.{str(algo)}.bin"
    blob = chunk_path.read_bytes()

    import time

    lat_ms = []
    miss = 0
    last_raw = b""
    for _ in range(int(loops)):
        t0 = time.time()
        raw = _decompress_bytes(blob, algo=algo)
        _ = sha256_bytes(raw)
        dt = float((time.time() - t0) * 1000.0)
        lat_ms.append(dt)
        if dt > float(deadline_ms):
            miss += 1
        last_raw = raw

    lat_ms.sort()

    def _pct(p: float) -> float:
        if not lat_ms:
            return 0.0
        if p <= 0:
            return float(lat_ms[0])
        if p >= 100:
            return float(lat_ms[-1])
        k = (len(lat_ms) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(lat_ms) - 1)
        if f == c:
            return float(lat_ms[f])
        d0 = lat_ms[f] * (c - k)
        d1 = lat_ms[c] * (k - f)
        return float(d0 + d1)

    miss_rate = float(miss) / float(max(1, int(loops)))
    report = {
        "status": "PASS",
        "loops": int(loops),
        "deadline_ms": float(deadline_ms),
        "miss_rate": float(miss_rate),
        "latency_ms": {
            "p50": _pct(50),
            "p90": _pct(90),
            "p99": _pct(99),
            "p999": _pct(99.9),
            "max": float(lat_ms[-1]) if lat_ms else 0.0,
        },
    }
    return report, last_raw

