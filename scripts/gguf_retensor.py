#!/usr/bin/env python3
"""gguf_retensor -- surgical GGUF tensor type conversion, with dense repacking.

WHAT IT IS FOR
--------------
Edge0/GGUF work repeatedly needs "change the storage type of *these* tensors and
touch nothing else" -- e.g. matching a donor model's per-tensor type set for an
MTP head, or forcing a tensor to F32 because a framework pass silently rounds it.
Neither llama-quantize (whole-model) nor gguf-py can do that: gguf-py's quantizer
cannot encode IQ4_XS / Q2_K / Q3_K at all.

WHY IT IS NOT JUST "PATCH THE type FIELD"
-----------------------------------------
ggml REQUIRES the data section to be densely packed in tensor-info order:

    ggml/src/gguf.cpp:770
    if (ti.offset != ctx->size) -> "tensor 'X' has offset A, expected B"
    ctx->size += GGML_PAD(ggml_nbytes(t), alignment)   // per tensor, in order

So any change in a tensor's size invalidates every following offset. This tool
repacks the suffix: from the FIRST affected tensor to EOF, re-emitting every
tensor densely, rewriting the type/offset fields, and truncating. Nothing before
the first affected tensor moves.

NUMBERS COME FROM ggml, NOT FROM A HARDCODED TABLE
--------------------------------------------------
Type ids, block sizes and byte sizes are read from `qctl types`, which enumerates
ggml's own trait table. Conversion is one `qctl cast` call per tensor, so the
bytes come from the same kernels that build production models. Python here does
layout only: offsets, header fields, backup, restore.

SAFETY
------
* Every mutating command is a dry run unless --apply is given.
* --apply takes a backup first (manifest.json + the before-image of the region),
  so `restore` puts the file back byte-for-byte without rebuilding 20 GB.
* Before writing, the parsed header is reconciled against gguf-py and checked for
  dense packing. A repack computed from a mis-parsed header would corrupt the
  file, so this is the gate that makes patching by file position safe.
* After writing, the file is re-parsed and re-verified.

EXAMPLES
--------
  scripts/gguf_retensor.py types
  scripts/gguf_retensor.py list   --gguf x.gguf --match 'ffn_(gate|up)_exps'
  scripts/gguf_retensor.py verify --gguf x.gguf

  # shrink the MTP head's expert stacks to the donor's types (frees ~1.3 GiB).
  # Multiple --type values pair with --match by position:
  scripts/gguf_retensor.py set-type --gguf x.gguf \\
      --match 'blk\\.40\\.ffn_(gate|up)_exps\\.weight' --type Q2_K \\
      --match 'blk\\.40\\.ffn_down_exps\\.weight'      --type Q3_K --apply

  # undo it, byte-for-byte
  scripts/gguf_retensor.py restore --backup x.gguf.retensor-backup --apply

  # prove the round trip
  scripts/gguf_retensor.py digest --gguf x.gguf --range 19535614816:
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

VERSION = 1
TOOL = "gguf_retensor"

# GGUF metadata value types (fixed-width ones, plus string/array handled apart).
_KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_STRING = 8
_KV_ARRAY = 9
_ALIGN_FALLBACK = 32


# --------------------------------------------------------------------------- #
# header parsing
# --------------------------------------------------------------------------- #
class _Reader:
    """Forward-only buffered reader.

    The kv section is not small: tokenizer arrays (248320 tokens + 247587 merges)
    make these headers ~10 MB, so a fixed 1 MiB window is not enough. This grows
    on demand instead of guessing.
    """

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.buf = self.f.read(1 << 20)
        self.pos = 0

    def close(self):
        self.f.close()

    def need(self, n: int):
        while len(self.buf) < n:
            chunk = self.f.read(max(1 << 20, n - len(self.buf)))
            if not chunk:
                raise ValueError("unexpected EOF while parsing the header")
            self.buf += chunk

    def u32(self) -> int:
        self.need(self.pos + 4)
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        self.need(self.pos + 8)
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def raw(self, n: int) -> bytes:
        self.need(self.pos + n)
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def skip(self, n: int):
        self.need(self.pos + n)
        self.pos += n


def _skip_value(rd: _Reader, depth: int = 0, vt: int | None = None):
    if vt is None:
        vt = rd.u32()
    if vt in _KV_FIXED:
        rd.skip(_KV_FIXED[vt])
    elif vt == _KV_STRING:
        rd.string()
    elif vt == _KV_ARRAY:
        if depth > 4:
            raise ValueError("nested array too deep to be a real GGUF")
        et = rd.u32()
        count = struct.unpack("<Q", rd.raw(8))[0]
        if et == _KV_ARRAY:
            for _ in range(count):
                _skip_value(rd, depth + 1, _KV_ARRAY)
        elif et == _KV_STRING:
            for _ in range(count):
                rd.string()
        elif et in _KV_FIXED:
            rd.skip(_KV_FIXED[et] * count)
        else:
            raise ValueError(f"unknown array element type {et}")
    else:
        raise ValueError(f"unknown value type {vt}")


def parse_header(path: Path):
    """Walk the GGUF header. Returns (tensors, info_end, n_kv, n_tensors).

    tensors[name] = {'ne', 'type', 'stored_off', 'index', 'type_pos', 'off_pos'}
    where type_pos / off_pos are absolute byte positions of the two fields we
    rewrite. Those positions are exactly what header_self_check() validates.
    """
    rd = _Reader(path)
    try:
        magic = rd.raw(4)
        if magic != b"GGUF":
            raise ValueError(f"bad magic {magic!r} (expected b'GGUF')")
        version = rd.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = rd.u64()
        n_kv = rd.u64()

        for _ in range(n_kv):
            rd.string()
            _skip_value(rd)

        tensors = {}
        for idx in range(n_tensors):
            name = rd.string()
            ne = [rd.u64() for _ in range(rd.u32())]
            type_pos = rd.pos
            tid = rd.u32()
            off_pos = rd.pos
            stored = rd.u64()
            tensors[name] = {"ne": ne, "type": tid, "stored_off": stored,
                             "index": idx, "type_pos": type_pos, "off_pos": off_pos}
        return tensors, rd.pos, n_kv, n_tensors
    finally:
        rd.close()


def read_alignment(path: Path, default: int = _ALIGN_FALLBACK) -> int:
    try:
        import gguf  # type: ignore
        return int(gguf.GGUFReader(str(path)).fields["general.alignment"].contents())
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# ggml type table (from qctl, never hardcoded)
# --------------------------------------------------------------------------- #
class Types:
    def __init__(self, rows):
        self.by_id = {r["id"]: r for r in rows}
        self.by_name = {r["name"].lower(): r for r in rows}

    def resolve(self, spec: str) -> dict:
        s = str(spec).strip()
        if s.lower() in self.by_name:
            return self.by_name[s.lower()]
        try:
            return self.by_id[int(s)]
        except (ValueError, KeyError):
            raise SystemExit(f"unknown ggml type {spec!r}; try `{TOOL}.py types`")

    def row_bytes(self, tid: int, n_per_row: int) -> int:
        if tid not in self.by_id:
            raise SystemExit(f"ggml type id {tid} is not in this build")
        t = self.by_id[tid]
        blck, size = int(t["blck_size"]), int(t["type_size"])
        if blck > 1 and n_per_row % blck:
            raise SystemExit(f"type {self.name(tid)}: ne[0]={n_per_row} is not a "
                             f"multiple of block size {blck}")
        return (n_per_row // blck) * size

    def name(self, tid: int) -> str:
        return self.by_id.get(tid, {}).get("name", str(tid))

    # qctl's to_f32/from_f32 special-case exactly these three; everything else must
    # have a trait. Note ggml reports F32 with to_float=... false (a memcpy) and
    # i8/i16/i32/i64 with is_quantized=false, so "not quantized" is NOT a usable
    # test for "we can convert it".
    _FLOAT = ("F32", "F16", "BF16")

    def _is_float(self, tid: int) -> bool:
        return self.by_id.get(tid, {}).get("name", "").upper() in self._FLOAT

    def can_decode(self, tid: int) -> bool:
        return tid in self.by_id and (self._is_float(tid) or
                                      bool(self.by_id[tid]["to_float"]))

    def can_encode(self, tid: int) -> bool:
        return tid in self.by_id and (self._is_float(tid) or
                                      bool(self.by_id[tid]["from_float"]))


# --------------------------------------------------------------------------- #
# qctl
# --------------------------------------------------------------------------- #
class Qctl:
    """Builds (once) and drives scripts/gguf_retensor_qctl.c."""

    def __init__(self, repo: Path, override: str | None = None):
        env = os.environ.get("GGUF_RETENSOR_QCTL")
        self.path = Path(override or env) if (override or env) else \
            repo / ".build" / "gguf_retensor" / "qctl"
        self.repo = repo
        self.libdir = repo / "src" / "llama.cpp" / "build" / "bin"
        self.types: Types | None = None

    def ensure(self):
        if self.path.exists():
            return
        src = self.repo / "scripts" / "gguf_retensor_qctl.c"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        if not list(self.libdir.glob("libggml-base.*")):
            raise SystemExit(f"ggml libraries not found in {self.libdir}\n"
                             f"  build llama.cpp first: "
                             f"cmake --build src/llama.cpp/build -j 8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["cc", "-O2", "-o", str(self.path), str(src),
               f"-I{self.repo / 'src' / 'llama.cpp' / 'ggml' / 'include'}",
               f"-L{self.libdir}", "-lggml-base", f"-Wl,-rpath,{self.libdir}"]
        print(f"[build] {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True)

    def _env(self):
        env = dict(os.environ)
        env.setdefault("DYLD_LIBRARY_PATH", str(self.libdir))
        env.setdefault("LD_LIBRARY_PATH", str(self.libdir))
        return env

    def load_types(self) -> Types:
        self.ensure()
        out = subprocess.run([str(self.path), "types"], capture_output=True,
                             text=True, env=self._env(), check=True).stdout
        rows = []
        for line in out.splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 7:
                rows.append({"id": int(f[0]), "name": f[1], "blck_size": int(f[2]),
                             "type_size": int(f[3]), "to_float": f[4] == "1",
                             "from_float": f[5] == "1", "quantized": f[6] == "1"})
        self.types = Types(rows)
        return self.types

    def cast(self, dst_tid: int, blob: bytes, src_tid: int, nrows: int,
             n_per_row: int, scratch: Path) -> bytes:
        src, dst = scratch / "in.bin", scratch / "out.bin"
        src.write_bytes(blob)
        p = subprocess.run([str(self.path), "cast", str(dst_tid), str(src),
                            str(src_tid), str(dst), str(nrows), str(n_per_row)],
                           capture_output=True, text=True, env=self._env())
        if p.returncode != 0:
            raise SystemExit(f"qctl cast {src_tid}->{dst_tid} failed: "
                             f"{(p.stderr or p.stdout).strip()}")
        return dst.read_bytes()


# --------------------------------------------------------------------------- #
# derived fields + header self-check
# --------------------------------------------------------------------------- #
def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def decorate(tensors, types: Types, info_end: int, alignment: int):
    """Add derived fields, including each tensor's ABSOLUTE file offset.

    Stored offsets are relative to the data section, which per the GGUF spec starts
    at the next `alignment` boundary after the tensor infos. Verified against
    gguf-py on both real artifacts (align_up(10990915,32) == 10990944 == the
    reader's offset for blk.0). header_self_check() re-checks this on every run, so
    the structural derivation is never assumed.
    """
    data_start = align_up(info_end, alignment)
    for name, h in tensors.items():
        nrows = 1
        for d in h["ne"][1:]:
            nrows *= int(d)
        h["name"] = name
        h["nrows"] = nrows
        h["n_per_row"] = int(h["ne"][0]) if h["ne"] else 0
        h["row_bytes"] = types.row_bytes(h["type"], h["n_per_row"])
        h["bytes"] = h["row_bytes"] * nrows
        h["abs_offset"] = h["stored_off"] + data_start
    return tensors, data_start


def header_self_check(path: Path, tensors, info_end: int, alignment: int,
                      expect_data_start: int | None = None) -> dict:
    """Reconcile our header walk against gguf-py, and check ggml's density rule.

    Both halves matter. The reader cross-check proves that data_start, type_pos and
    off_pos are what we think they are (patching by position is only safe if so).
    The density check proves that rewriting a suffix cannot clobber bytes belonging
    to a tensor outside that suffix.
    """
    problems: list[str] = []
    structural: list[str] = []
    size = path.stat().st_size

    starts = {h["abs_offset"] - h["stored_off"] for h in tensors.values()}
    data_start = min(starts) if starts else 0
    if len(starts) > 1:
        structural.append(f"ambiguous data-section start: {sorted(starts)[:5]}")
    if expect_data_start is not None and data_start != expect_data_start:
        problems.append(f"data-section start {data_start} != align_up(info_end)"
                        f" {expect_data_start}")
    if alignment and data_start % alignment:
        structural.append(f"data-section start {data_start} is not {alignment}-aligned")
    for name, h in tensors.items():
        if not 1 <= len(h["ne"]) <= 4:
            structural.append(f"{name}: {len(h['ne'])} dims")
        if h["abs_offset"] + h["bytes"] > size:
            problems.append(f"{name}: ends at {h['abs_offset'] + h['bytes']:,} "
                            f"past EOF {size:,}")
    # A desynced walk would mis-read ne/type/offset for everything after it, so
    # non-monotonic stored offsets mean our field positions are wrong, not that the
    # file is exotic. (Unknown type ids / block-size mismatches already blew up in
    # decorate(), so they cannot reach here.)
    prev = -1
    for h in sorted(tensors.values(), key=lambda x: x["index"]):
        if h["stored_off"] < prev:
            structural.append(f"{h['name']}: stored offset {h['stored_off']:,} < "
                              f"previous {prev:,} (header walk desynced)")
            break
        prev = h["stored_off"]
    problems.extend(structural)

    # ggml's rule: offsets accumulate in tensor-info order, each padded.
    expect, first_gap = None, None
    for h in sorted(tensors.values(), key=lambda x: x["index"]):
        if expect is None:
            expect = h["stored_off"]
        if h["stored_off"] != expect and first_gap is None:
            first_gap = {"name": h["name"], "found": h["stored_off"],
                         "expected": expect}
        expect = h["stored_off"] + (h["bytes"] + alignment - 1) // alignment * alignment

    reader_ok, reader_err, reader_checked = False, None, 0
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "src" / "llama.cpp" / "gguf-py"))
        import gguf  # type: ignore
        rmap = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
        for name, h in tensors.items():
            t = rmap.get(name)
            if t is None:
                problems.append(f"{name}: in our walk, absent from gguf-py")
                continue
            exp_abs = h["stored_off"] + data_start
            if int(t.data_offset) != exp_abs:
                problems.append(f"{name}: gguf-py offset {t.data_offset} != parsed {exp_abs}")
            if int(t.tensor_type) != h["type"]:
                problems.append(f"{name}: gguf-py type {t.tensor_type} != parsed {h['type']}")
            reader_checked += 1
        reader_ok = True
    except Exception as e:
        reader_err = f"{type(e).__name__}: {e}"

    return {"file_size": size, "n_tensors": len(tensors), "alignment": alignment,
            "info_end": info_end, "data_start": data_start,
            "dense": first_gap is None, "first_gap": first_gap,
            "reader_ok": reader_ok, "reader_error": reader_err,
            "reader_checked": reader_checked, "problems": problems,
            "structural_ok": not structural, "structural": structural}


# --------------------------------------------------------------------------- #
# selection / planning
# --------------------------------------------------------------------------- #
def _patterns(args):
    return [re.compile(p) for p in (args.match or [])], \
           [re.compile(p) for p in (args.exclude or [])]


def _match_index(name, inc, exc):
    """Index of the first --match pattern that hits, or None if excluded/unmatched."""
    if exc and any(p.search(name) for p in exc):
        return None
    for i, p in enumerate(inc):
        if p.search(name):
            return i
    return None


def _select(tensors, args):
    inc, exc = _patterns(args)
    if not inc:
        raise SystemExit("nothing selected -- pass --match")
    sel = {}
    for name, h in tensors.items():
        i = _match_index(name, inc, exc)
        if i is not None:
            sel[name] = (h, i)
    return sel


def _target_types(args, types: Types, sel):
    """One --type applies to all matches; N --type values pair with N --match."""
    specs = args.type or []
    if not specs:
        raise SystemExit("--type is required")
    if len(specs) == 1:
        tid = types.resolve(specs[0])["id"]
        return {name: tid for name in sel}, types.name(tid)
    if len(specs) != len(args.match or []):
        raise SystemExit("--type must be given once, or exactly as many times as "
                         "--match (they pair by position)")
    tids = [types.resolve(s)["id"] for s in specs]
    out = {name: tids[min(i, len(tids) - 1)] for name, (_, i) in sel.items()}
    return out, " + ".join(types.name(t) for t in tids)


def _plan(tensors, types: Types, changes, alignment: int):
    """Dense layout for the suffix beginning at the first changed tensor."""
    first = min(tensors[n]["index"] for n in changes)
    suffix = [h for h in sorted(tensors.values(), key=lambda x: x["index"])
              if h["index"] >= first]
    region_start = suffix[0]["abs_offset"]
    for h in suffix:
        new_tid = changes.get(h["name"], h["type"])
        h["new_type"] = new_tid
        h["new_bytes"] = types.row_bytes(new_tid, h["n_per_row"]) * h["nrows"]
        h["new_padded"] = (h["new_bytes"] + alignment - 1) // alignment * alignment
    off = region_start
    for h in suffix:
        h["new_offset"] = off
        off += h["new_padded"]
    return suffix, region_start, off


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def _apply_plan(path, q, types, suffix, region_start, new_end, backup_dir, apply,
                force, command, data_start):
    size = path.stat().st_size
    old_region, new_region = size - region_start, new_end - region_start
    growth = max(0, new_region - old_region)
    changed = [h for h in suffix if h["new_type"] != h["type"]]

    print(f"\nregion          : {region_start:,} .. {size:,}  "
          f"({old_region / 2**20:.1f} MiB)")
    print(f"after repack    : {region_start:,} .. {new_end:,}  "
          f"({new_region / 2**20:.1f} MiB)")
    print(f"file size       : {size:,} -> {new_end:,}  "
          f"({'grows' if new_end > size else 'shrinks'} "
          f"{abs(new_end - size) / 2**20:.1f} MiB)")
    print(f"tensors changed : {len(changed)} of {len(suffix)} in the region")

    free = shutil.disk_usage(path.parent).free
    need = new_region + growth
    print(f"disk            : need ~{need / 2**20:.1f} MiB "
          f"(temp tail {new_region / 2**20:.1f} + growth {growth / 2**20:.1f}), "
          f"free {free / 2**20:.1f} MiB")
    if need > free and not force:
        raise SystemExit("not enough free disk -- free space, point --backup at "
                         "another volume, or pass --force")

    if not apply:
        print("\nDRY RUN. Re-run with --apply.")
        for h in changed:
            print(f"  {h['name']:<44} {types.name(h['type']):>7} -> "
                  f"{types.name(h['new_type']):<7} {h['bytes']:>12,} -> "
                  f"{h['new_bytes']:>12,}")
        return 0

    # ---------------- backup (before-image + header fields) ----------------
    backup_dir.mkdir(parents=True, exist_ok=True)
    region_bin = backup_dir / "region.bin"
    with open(path, "rb") as f, open(region_bin, "wb") as g:
        f.seek(region_start)
        shutil.copyfileobj(f, g, 1 << 22)

    patches = []
    with open(path, "rb") as f:
        for h in suffix:
            if h["new_type"] != h["type"]:
                f.seek(h["type_pos"])
                patches.append({"pos": h["type_pos"], "old_hex": f.read(4).hex(),
                                "what": "type", "name": h["name"]})
            f.seek(h["off_pos"])
            patches.append({"pos": h["off_pos"], "old_hex": f.read(8).hex(),
                            "what": "offset", "name": h["name"]})

    manifest = {
        "tool": TOOL, "version": VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:], "command": command, "gguf": str(path),
        "orig_size": size, "new_size": new_end,
        "region_start": region_start, "region_bytes": old_region,
        "header_patches": patches,
        "entries": [{"name": h["name"], "old_type": h["type"], "new_type": h["new_type"],
                     "old_stored_off": h["stored_off"],
                     "new_stored_off": h["new_offset"] - (h["abs_offset"] - h["stored_off"]),
                     "old_bytes": h["bytes"], "new_bytes": h["new_bytes"]}
                    for h in suffix],
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nbackup          : {backup_dir}  (manifest.json + region.bin "
          f"{old_region / 2**20:.1f} MiB)")

    # ---------------- build the new tail ----------------
    tmp = Path(str(path) + ".retensor.tmp")
    scratch = Path(str(path) + ".retensor.scratch")
    scratch.mkdir(exist_ok=True)
    # data_start comes from the caller's header walk, NOT from region_start minus the
    # first suffix tensor's stored offset: `repack` deliberately starts the region at
    # the DENSE position, which is not that tensor's current stored offset.
    try:
        print("\nbuilding new tail:")
        with open(tmp, "wb") as out:
            for h in suffix:
                with open(path, "rb") as f:
                    f.seek(h["abs_offset"])
                    blob = f.read(h["bytes"])
                if h["new_type"] == h["type"]:
                    data = blob
                else:
                    data = q.cast(h["new_type"], blob, h["type"], h["nrows"],
                                  h["n_per_row"], scratch)
                if len(data) != h["new_bytes"]:
                    raise SystemExit(f"{h['name']}: got {len(data)} bytes, expected "
                                     f"{h['new_bytes']}")
                out.write(data)
                out.write(b"\x00" * (h["new_padded"] - h["new_bytes"]))
                print(f"  {h['name']:<44} {types.name(h['type']):>7} -> "
                      f"{types.name(h['new_type']):<7} @ {h['new_offset']:,}")
        got = tmp.stat().st_size
        if got != new_region:
            raise SystemExit(f"tail temp is {got} B, expected {new_region} B")

        with open(tmp, "rb") as w, open(path, "r+b") as g:
            g.seek(region_start)
            shutil.copyfileobj(w, g, 1 << 22)
            g.truncate(new_end)
            for h in suffix:
                g.seek(h["type_pos"])
                g.write(struct.pack("<I", h["new_type"]))
                g.seek(h["off_pos"])
                g.write(struct.pack("<Q", h["new_offset"] - data_start))
    finally:
        tmp.unlink(missing_ok=True)
        for f in scratch.glob("*"):
            f.unlink(missing_ok=True)
        scratch.rmdir()

    print(f"\napplied. file size = {path.stat().st_size:,}")
    return 0


def _post_verify(path, q, note=""):
    tensors, info_end, n_kv, n_tensors = parse_header(path)
    alignment = read_alignment(path)
    tensors, dstart = decorate(tensors, q.types, info_end, alignment)
    r = header_self_check(path, tensors, info_end, alignment, dstart)
    ok = r["dense"] and not r["problems"] and r["reader_ok"]
    print(f"\npost-check{(' (' + note + ')') if note else ''}: "
          f"dense={'OK' if r['dense'] else 'BROKEN'} "
          f"reader={'OK' if r['reader_ok'] else 'SKIPPED'} "
          f"problems={len(r['problems'])} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        for p in (r["problems"] or [r["first_gap"]])[:10]:
            print("  -", p)
    return ok


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _load(args, repo):
    path = Path(args.gguf).resolve()
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    q = Qctl(repo, args.qctl)
    types = q.load_types()
    tensors, info_end, n_kv, n_tensors = parse_header(path)
    alignment = read_alignment(path)
    tensors, data_start = decorate(tensors, types, info_end, alignment)
    return path, q, types, tensors, info_end, n_kv, n_tensors, alignment, data_start


def cmd_types(args, repo):
    t = Qctl(repo, args.qctl).load_types()
    print(f"{'id':>3}  {'name':<12} {'block':>6} {'bytes':>6}  "
          f"{'to_float':>8} {'from_float':>10}")
    for r in sorted(t.by_id.values(), key=lambda r: r["id"]):
        print(f"{r['id']:>3}  {r['name']:<12} {r['blck_size']:>6} {r['type_size']:>6}  "
              f"{str(r['to_float']):>8} {str(r['from_float']):>10}")
    return 0


def cmd_list(args, repo):
    path, q, types, tensors, info_end, n_kv, n_t, alignment, _ = _load(args, repo)
    if args.match:
        sel = {n: h for n, (h, _) in _select(tensors, args).items()}
    else:
        sel = tensors
    key = {"name": lambda h: h["name"],
           "offset": lambda h: h["abs_offset"],
           "size": lambda h: -h["bytes"]}[args.sort]
    rows = sorted(sel.values(), key=key)
    print(f"{path.name}: {len(tensors)} tensors, alignment {alignment}")
    print(f"{'offset':>16} {'type':<8} {'bytes':>14}  name")
    hist, total = {}, 0
    for h in rows:
        hist[types.name(h["type"])] = hist.get(types.name(h["type"]), 0) + 1
        total += h["bytes"]
        print(f"{h['abs_offset']:>16,} {types.name(h['type']):<8} {h['bytes']:>14,}  "
              f"{h['name']}")
    print(f"\n{len(rows)} selected, {total:,} B ({total / 2**20:.1f} MiB)")
    print("types:", ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
    if args.json:
        print(json.dumps({"file": str(path), "selected": len(rows), "bytes": total,
                          "types": hist}, indent=2))
    return 0


def cmd_verify(args, repo):
    path, q, types, tensors, info_end, n_kv, n_tensors, alignment, dstart = _load(args, repo)
    r = header_self_check(path, tensors, info_end, alignment, dstart)
    ok = r["dense"] and not r["problems"] and r["reader_ok"]
    print(f"file            : {path}  ({r['file_size']:,} B)")
    print(f"tensors / kv    : {r['n_tensors']} / {n_kv}")
    print(f"info ends at    : {r['info_end']:,}")
    print(f"data starts at  : {r['data_start']:,}")
    print(f"alignment       : {r['alignment']}")
    print(f"dense packing   : {'OK' if r['dense'] else 'BROKEN'}  "
          f"{'' if r['dense'] else r['first_gap']}")
    print(f"gguf-py agrees  : " + (f"OK ({r['reader_checked']} tensors reconciled)"
                                   if r["reader_ok"] else f"SKIPPED ({r['reader_error']})"))
    for p in r["problems"][:20]:
        print("  -", p)
    print("VERDICT:", "OK" if ok else "PROBLEMS FOUND")
    if args.json:
        print(json.dumps(r, indent=2))
    return 0 if ok else 1


def cmd_set_type(args, repo):
    path, q, types, tensors, info_end, n_kv, n_tensors, alignment, dstart = _load(args, repo)
    r = header_self_check(path, tensors, info_end, alignment, dstart)
    if not r["reader_ok"] and not args.force:
        raise SystemExit(f"header cross-check unavailable ({r['reader_error']}); "
                         f"refusing to repack from an unverified parse -- pass "
                         f"--force if you accept that risk")
    if not r["dense"] and not args.force:
        raise SystemExit(f"file is not densely packed ({r['first_gap']}); repacking a "
                         f"suffix would be unsafe -- run `repack` first, or --force")
    if r["problems"] and not args.force:
        raise SystemExit("header problems: " + "; ".join(r["problems"][:5]))

    sel = _select(tensors, args)
    if not sel:
        raise SystemExit("nothing matched -- check --match / --exclude")
    targets, label = _target_types(args, types, sel)

    for name, tid in targets.items():
        h = tensors[name]
        if not types.can_decode(h["type"]):
            raise SystemExit(f"{name}: source {types.name(h['type'])} has no decoder "
                             f"in this build -- cannot convert")
        if not types.can_encode(tid):
            raise SystemExit(f"{name}: target {types.name(tid)} has no encoder")

    print(f"target: {len(targets)} tensors -> {label}")
    total = 0
    for name, tid in sorted(targets.items()):
        h = tensors[name]
        nb = types.row_bytes(tid, h["n_per_row"]) * h["nrows"]
        total += nb - h["bytes"]
        print(f"  {name:<48} {types.name(h['type']):>7} -> {types.name(tid):<7} "
              f"{h['bytes']:>13,} -> {nb:>13,}")
    print(f"  total {total / 2**20:+.1f} MiB")

    suffix, region_start, new_end = _plan(tensors, types, targets, alignment)
    rc = _apply_plan(path, q, types, suffix, region_start, new_end,
                     Path(args.backup or (str(path) + ".retensor-backup")),
                     args.apply, args.force, "set-type", dstart)
    if rc == 0 and args.apply:
        _post_verify(path, q)
    return rc


def cmd_repack(args, repo):
    path, q, types, tensors, info_end, n_kv, n_tensors, alignment, dstart = _load(args, repo)
    r = header_self_check(path, tensors, info_end, alignment, dstart)
    if r["dense"]:
        print("already densely packed -- nothing to do")
        return 0
    # repack is a REPAIR operation, so it must not depend on gguf-py being able to read
    # the file -- a broken data layout is exactly what makes the reader throw. What it
    # does depend on is our own walk being sound, which the structural invariants test.
    if not r["structural_ok"] and not args.force:
        raise SystemExit("our own header walk looks desynced: " +
                         "; ".join(r["structural"][:5]) + " -- pass --force to override")
    if not r["reader_ok"]:
        print(f"note: gguf-py cannot read this file ({r['reader_error']})\n"
              f"      proceeding on structural invariants only (repair path)")
    gap = r["first_gap"]
    print(f"first gap: {gap}")
    first = tensors[gap["name"]]["index"]
    suffix = [h for h in sorted(tensors.values(), key=lambda x: x["index"])
              if h["index"] >= first]
    # Close the gap: start the region at the DENSE position, not at the offset the file
    # currently claims. gap['expected'] is the padded end of the preceding tensor -- in
    # STORED (data-section-relative) coordinates, like every stored_off, so it must be
    # rebased by data_start before it can be used as a file position.
    region_start = gap["expected"] + dstart
    off = region_start
    for h in suffix:
        h["new_type"] = h["type"]
        h["new_bytes"] = h["bytes"]
        h["new_padded"] = (h["bytes"] + alignment - 1) // alignment * alignment
        h["new_offset"] = off
        off += h["new_padded"]
    rc = _apply_plan(path, q, types, suffix, region_start, off,
                     Path(args.backup or (str(path) + ".retensor-backup")),
                     args.apply, args.force, "repack", dstart)
    if rc == 0 and args.apply:
        _post_verify(path, q)
    return rc


def cmd_restore(args, repo):
    bdir = Path(args.backup).resolve()
    mp = bdir / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"no manifest at {mp}")
    man = json.loads(mp.read_text())
    path, region_bin = Path(man["gguf"]), bdir / "region.bin"
    if not path.exists() or not region_bin.exists():
        raise SystemExit(f"missing {path} or {region_bin}")

    cur = path.stat().st_size
    print(f"manifest   : {man['tool']} v{man['version']}  {man['created']}")
    print(f"command    : {man.get('command')}  {man.get('argv')}")
    print(f"file       : {path}")
    print(f"size       : {cur:,} -> {man['orig_size']:,}")
    print(f"region     : {man['region_start']:,}  ({man['region_bytes']:,} B)")
    print(f"tensors    : {len(man['entries'])}  header patches: "
          f"{len(man['header_patches'])}")
    if cur != man["new_size"] and not args.force:
        raise SystemExit(f"file size {cur:,} != manifest's expected "
                         f"{man['new_size']:,}; the file changed since the backup "
                         f"-- pass --force to restore anyway")
    if not args.apply:
        print("\nDRY RUN. Re-run with --apply.")
        return 0

    with open(path, "r+b") as g:
        for p in man["header_patches"]:
            g.seek(p["pos"])
            g.write(bytes.fromhex(p["old_hex"]))
        g.seek(man["region_start"])
        with open(region_bin, "rb") as f:
            shutil.copyfileobj(f, g, 1 << 22)
        g.truncate(man["orig_size"])
    print(f"\nrestored. file size = {path.stat().st_size:,}")
    return 0


def cmd_digest(args, repo):
    path = Path(args.gguf).resolve()
    size = path.stat().st_size
    start, end = 0, size
    if args.range:
        a, _, b = args.range.partition(":")
        start = int(a) if a.strip() else 0
        end = int(b) if b.strip() else size
    h = hashlib.sha256()
    left = end - start
    with open(path, "rb") as f:
        f.seek(start)
        while left > 0:
            chunk = f.read(min(1 << 22, left))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    print(f"{h.hexdigest()}  {path.name}[{start}:{end}]")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog=f"{TOOL}.py",
                                description="Surgical GGUF tensor type conversion.")
    p.add_argument("--repo", help="repo root (default: this script's parent dir)")
    p.add_argument("--qctl", help="path to the qctl binary (default: auto-build)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--gguf", required=True, help="path to the .gguf file")
        sp.add_argument("--match", action="append",
                        help="regex on the tensor name (repeatable)")
        sp.add_argument("--exclude", action="append",
                        help="regex to drop from the selection (repeatable)")
        sp.add_argument("--json", action="store_true", help="also emit JSON")

    def mutating(sp):
        common(sp)
        sp.add_argument("--backup", help="backup dir (default: <gguf>.retensor-backup)")
        sp.add_argument("--apply", action="store_true", help="actually write")
        sp.add_argument("--force", action="store_true",
                        help="proceed despite a failed check / low disk")

    # NOTE: never set_defaults() for --repo / --qctl here. A subparser's defaults are
    # applied AFTER the parent namespace is built, so they would silently clobber the
    # top-level option the user passed.
    sub.add_parser("types", help="list the ggml types this build knows") \
       .set_defaults(func=cmd_types)

    s = sub.add_parser("list", help="list tensors (offset/type/bytes)")
    common(s)
    s.add_argument("--sort", choices=("name", "offset", "size"), default="offset")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("verify", help="header self-check: parse, density, bounds")
    common(s)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("set-type", help="change the storage type of matched tensors")
    mutating(s)
    s.add_argument("--type", action="append",
                   help="target ggml type (name or id); repeat to pair with --match")
    s.set_defaults(func=cmd_set_type)

    s = sub.add_parser("repack", help="repack densely without changing any type")
    mutating(s)
    s.set_defaults(func=cmd_repack)

    s = sub.add_parser("restore", help="undo a set-type/repack from its backup")
    s.add_argument("--backup", required=True)
    s.add_argument("--apply", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_restore)

    s = sub.add_parser("digest", help="sha256 of the file, or of a byte range")
    s.add_argument("--gguf", required=True)
    s.add_argument("--range", help="START:END (default: whole file)")
    s.set_defaults(func=cmd_digest)

    return p


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    repo = Path(args.repo).resolve() if getattr(args, "repo", None) else \
        Path(__file__).resolve().parent.parent
    try:
        return args.func(args, repo)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
