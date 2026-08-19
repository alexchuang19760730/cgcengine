import json
import os
import subprocess
import sys
import time
import traceback
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.fx


def _as_shape_dict(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return [int(x) for x in list(v.shape)]
    if isinstance(v, (list, tuple)):
        return [_as_shape_dict(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _as_shape_dict(val) for k, val in v.items()}
    return str(type(v).__name__)


def _canonical_op_name(n: torch.fx.Node) -> str:
    import operator

    if n.op == "call_function":
        t = n.target
        if t in (torch.matmul, getattr(operator, "matmul", None)):
            return "matmul"
        if t in (torch.add, getattr(operator, "add", None)):
            return "add"
        if t in (torch.mul, getattr(operator, "mul", None)):
            return "mul"
        if t is torch.nn.functional.softmax:
            return "softmax"
        if t is torch.nn.functional.layer_norm:
            return "layer_norm"
        if t is torch.nn.functional.gelu:
            return "gelu"
        if t is torch.nn.functional.silu:
            return "silu"
        name = getattr(t, "__name__", None)
        if isinstance(name, str) and name != "":
            return name
        return str(t)
    if n.op == "call_method":
        try:
            return str(n.target)
        except Exception:
            return "call_method"
    if n.op == "call_module":
        try:
            return "module:" + str(n.target)
        except Exception:
            return "call_module"
    return str(n.op)


def _shape_propagate_fx(gm: torch.fx.GraphModule, example_inputs: Dict[str, Any]) -> Optional[str]:
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(gm).propagate(**example_inputs)
        return None
    except Exception as e:
        return str(e)


def _build_constraints_and_ir(
    *,
    gm: torch.fx.GraphModule,
    example_inputs: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    p = Path(str(out_dir)).expanduser()
    p.mkdir(parents=True, exist_ok=True)

    shape_prop_error = _shape_propagate_fx(gm, example_inputs)

    constraints: Dict[str, Any] = {
        "inputs": _as_shape_dict(example_inputs),
        "outputs": {},
        "shape_prop": {"status": "PASS" if shape_prop_error is None else "FAIL", "error": shape_prop_error},
    }

    primitive_nodes: List[Dict[str, Any]] = []
    op_hist: Dict[str, int] = {}
    for n in gm.graph.nodes:
        opn = _canonical_op_name(n)
        op_hist[opn] = int(op_hist.get(opn, 0)) + 1
        meta = n.meta.get("tensor_meta") if isinstance(getattr(n, "meta", None), dict) else None
        node_payload: Dict[str, Any] = {"name": str(n.name), "op": str(n.op), "target": str(n.target), "canon": opn}
        try:
            if meta is not None and hasattr(meta, "shape"):
                node_payload["shape"] = [int(x) for x in list(getattr(meta, "shape"))]
            if meta is not None and hasattr(meta, "dtype"):
                node_payload["dtype"] = str(getattr(meta, "dtype"))
        except Exception:
            pass
        primitive_nodes.append(node_payload)

    constraints_path = str(p / "constraints.json")
    primitive_ir_path = str(p / "primitive_ir.json")
    Path(constraints_path).write_text(json.dumps(constraints, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(primitive_ir_path).write_text(json.dumps({"nodes": primitive_nodes, "op_histogram": op_hist}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"constraints_path": constraints_path, "primitive_ir_path": primitive_ir_path, "op_histogram": op_hist, "shape_prop_error": shape_prop_error}


def _load_module_from_code(code: str) -> nn.Module:
    ns: Dict[str, Any] = {"torch": torch, "nn": nn}
    exec(str(code), ns, ns)
    candidates = []
    for v in ns.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            candidates.append(v)
    if len(candidates) == 0:
        raise RuntimeError("no nn.Module class found in code")
    try:
        return candidates[0]()
    except Exception as e:
        raise RuntimeError(f"failed to instantiate module: {e}") from e


def _ggml_type_name_to_torch_dtype(type_name: str) -> Optional[torch.dtype]:
    t = str(type_name or "").strip().lower()
    if t in ("f32", "float32"):
        return torch.float32
    if t in ("f16", "float16"):
        return torch.float16
    if t in ("bf16", "bfloat16"):
        return torch.bfloat16
    if t in ("i32", "int32"):
        return torch.int32
    if t in ("i16", "int16"):
        return torch.int16
    if t in ("i8", "int8"):
        return torch.int8
    if t in ("u8", "uint8"):
        return torch.uint8
    return None


def load_ggml_tap_tensor(meta_json_path: str) -> Dict[str, Any]:
    meta_p = Path(str(meta_json_path)).expanduser()
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if not bool(meta.get("dumped")):
        return {"status": "FAIL", "error": f"tap not dumped: {meta.get('error')}", "meta": meta}
    bin_path = str(meta.get("bin_path") or "").strip()
    if bin_path == "":
        return {"status": "FAIL", "error": "missing bin_path", "meta": meta}
    dt = _ggml_type_name_to_torch_dtype(str(meta.get("type_name") or ""))
    if dt is None:
        return {"status": "FAIL", "error": f"unsupported type_name: {meta.get('type_name')}", "meta": meta}
    shape = meta.get("ne") or []
    if not isinstance(shape, list) or len(shape) != 4:
        return {"status": "FAIL", "error": "invalid ne shape in meta", "meta": meta}
    ne = [int(x) for x in shape]
    n_dims = 1
    for k in range(4):
        if ne[k] > 1:
            n_dims = k + 1
    view_shape = [ne[i] for i in range(n_dims)]
    raw = Path(bin_path).read_bytes()
    t = torch.frombuffer(memoryview(raw), dtype=dt)
    try:
        t = t.reshape(view_shape).clone()
    except Exception as e:
        return {"status": "FAIL", "error": f"reshape failed: {e}", "meta": meta}
    return {"status": "PASS", "tensor": t, "meta": meta}


def _resolve_bin_path(meta_p: Path, raw_path: str) -> str:
    p = Path(str(raw_path or "").strip())
    if str(p) == "":
        return ""
    if p.is_absolute():
        return str(p)
    cand = (meta_p.parent / p).resolve()
    if cand.exists():
        return str(cand)
    return str(p)


def _itemsize_for_type_name(type_name: str) -> Optional[int]:
    t = str(type_name or "").strip().lower()
    if t in ("f32", "float32", "i32", "int32"):
        return 4
    if t in ("f16", "float16", "bf16", "bfloat16", "i16", "int16"):
        return 2
    if t in ("i8", "int8", "u8", "uint8"):
        return 1
    return None


def _load_tensor_from_meta_bundle(meta_json_path: str, *, tensor_name: str) -> Dict[str, Any]:
    meta_p = Path(str(meta_json_path)).expanduser()
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if isinstance(meta.get("name"), str) and meta.get("name") == str(tensor_name):
        return load_ggml_tap_tensor(str(meta_p))
    tensors = meta.get("tensors")
    entry: Optional[Dict[str, Any]] = None
    if isinstance(tensors, dict):
        v = tensors.get(str(tensor_name))
        if isinstance(v, dict):
            entry = v
    elif isinstance(tensors, list):
        for v in tensors:
            if isinstance(v, dict) and str(v.get("name") or "") == str(tensor_name):
                entry = v
                break
    if entry is None:
        return {"status": "FAIL", "error": f"tensor not found in meta bundle: {tensor_name}", "meta_json_path": str(meta_p)}

    type_name = str(entry.get("type_name") or meta.get("type_name") or "").strip()
    dt = _ggml_type_name_to_torch_dtype(type_name)
    if dt is None:
        return {"status": "FAIL", "error": f"unsupported type_name: {type_name}", "meta_json_path": str(meta_p)}

    ne = entry.get("ne") or entry.get("shape") or meta.get("ne") or []
    if not isinstance(ne, list) or len(ne) != 4:
        return {"status": "FAIL", "error": "invalid ne shape in meta", "meta_json_path": str(meta_p)}
    ne4 = [int(x) for x in ne]
    n_dims = 1
    for k in range(4):
        if ne4[k] > 1:
            n_dims = k + 1
    view_shape = [ne4[i] for i in range(n_dims)]

    raw_bin_path = str(entry.get("bin_path") or meta.get("bin_path") or "").strip()
    if raw_bin_path == "":
        raw_bin_path = str(meta_p.with_suffix(".bin"))
    bin_path = _resolve_bin_path(meta_p, raw_bin_path)
    if bin_path == "" or not Path(bin_path).expanduser().exists():
        return {"status": "FAIL", "error": f"bin_path not found: {bin_path}", "meta_json_path": str(meta_p)}

    offset = entry.get("offset_bytes")
    if offset is None:
        offset = entry.get("byte_offset")
    if offset is None:
        offset = entry.get("offset")
    try:
        offset_i = int(offset) if offset is not None else 0
    except Exception:
        offset_i = 0

    nbytes = entry.get("nbytes")
    try:
        nbytes_i = int(nbytes) if nbytes is not None else -1
    except Exception:
        nbytes_i = -1
    if nbytes_i < 0:
        itemsize = _itemsize_for_type_name(type_name)
        if itemsize is None:
            return {"status": "FAIL", "error": f"cannot infer itemsize from type_name: {type_name}", "meta_json_path": str(meta_p)}
        numel = 1
        for x in view_shape:
            numel *= int(x)
        nbytes_i = int(numel) * int(itemsize)

    raw = Path(bin_path).read_bytes()
    if offset_i < 0 or offset_i + nbytes_i > len(raw):
        return {
            "status": "FAIL",
            "error": "invalid slice",
            "meta_json_path": str(meta_p),
            "bin_path": str(bin_path),
            "offset_bytes": int(offset_i),
            "nbytes": int(nbytes_i),
            "bin_size": int(len(raw)),
        }

    mv = memoryview(raw)[offset_i : offset_i + nbytes_i]
    t = torch.frombuffer(mv, dtype=dt)
    try:
        t = t.reshape(view_shape).clone()
    except Exception as e:
        return {"status": "FAIL", "error": f"reshape failed: {e}", "meta_json_path": str(meta_p), "bin_path": str(bin_path)}
    return {"status": "PASS", "tensor": t, "meta": {"meta_json_path": str(meta_p), "bin_path": str(bin_path), "type_name": type_name, "ne": ne4}}


def _run_pytorch_subgraph_from_file(pytorch_path: str, *, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    p = Path(str(pytorch_path)).expanduser()
    if not p.exists() and not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        alt = (repo_root / p).resolve()
        if alt.exists():
            p = alt
    if not p.exists():
        return {"status": "FAIL", "error": "pytorch_path not found", "pytorch_path": str(p)}

    if p.suffix.lower() in (".pt", ".pth", ".ts", ".torchscript"):
        try:
            m = torch.jit.load(str(p), map_location="cpu")
            m.eval()
            with torch.no_grad():
                try:
                    out = m(**inputs)
                except Exception:
                    out = m(*list(inputs.values()))
            return {"status": "PASS", "output": out, "module_name": "torchscript"}
        except Exception:
            try:
                obj = torch.load(str(p), map_location="cpu")
                if isinstance(obj, nn.Module):
                    obj.eval()
                    with torch.no_grad():
                        try:
                            out = obj(**inputs)
                        except Exception:
                            out = obj(*list(inputs.values()))
                    return {"status": "PASS", "output": out, "module_name": str(type(obj).__name__)}
                return {"status": "PASS", "output": obj, "module_name": "torchload"}
            except Exception as e:
                return {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc(), "module_name": "torchload"}

    code = p.read_text(encoding="utf-8")
    local_scope: Dict[str, Any] = {}
    exec(code, {"torch": torch, "nn": nn, "Optional": Optional, "List": List, "Dict": Dict, "Any": Any}, local_scope)
    target_cls = None
    if isinstance(local_scope.get("SubgraphModule"), type) and issubclass(local_scope["SubgraphModule"], nn.Module):
        target_cls = local_scope["SubgraphModule"]
    else:
        for v in local_scope.values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                target_cls = v
                break
    if target_cls is None:
        return {"status": "FAIL", "error": "no nn.Module found"}
    try:
        m2: nn.Module = target_cls()
        m2.eval()
        with torch.no_grad():
            try:
                out = m2(**inputs)
            except Exception:
                out = m2(*list(inputs.values()))
        return {"status": "PASS", "output": out, "module_name": str(target_cls.__name__)}
    except Exception as e:
        return {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc(), "module_name": str(getattr(target_cls, "__name__", ""))}


def equivalence_check_from_taps(
    *,
    pytorch_path: str,
    input_tap_meta: Union[str, Dict[str, Any]],
    output_tap_meta: Union[str, Dict[str, Any]],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    tap_search_dir: Optional[str] = None,
) -> Dict[str, Any]:
    def _torch_dtype_to_type_name(dt: torch.dtype) -> Optional[str]:
        if dt is torch.float32:
            return "f32"
        if dt is torch.float16:
            return "f16"
        if dt is torch.bfloat16:
            return "bf16"
        if dt is torch.int32:
            return "i32"
        if dt is torch.int16:
            return "i16"
        if dt is torch.int8:
            return "i8"
        if dt is torch.uint8:
            return "u8"
        return None

    def _materialize_tap_bundle_from_torch_ref(meta_json_path: str, *, pytorch_ref_path: str) -> Dict[str, Any]:
        meta_p = Path(str(meta_json_path)).expanduser()
        if not meta_p.exists():
            return {"status": "FAIL", "error": "meta_json not found", "meta_json_path": str(meta_p)}
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"status": "FAIL", "error": f"meta_json parse failed: {e}", "meta_json_path": str(meta_p)}

        if isinstance(meta.get("tensors"), (list, dict)) and bool(meta.get("bin_path")):
            return {"status": "PASS", "status_note": "already_bundle"}

        names = meta.get("tensor_names")
        if not isinstance(names, list) or not all(isinstance(x, str) and x.strip() != "" for x in names):
            return {"status": "FAIL", "error": "meta_json missing tensor_names", "meta_json_path": str(meta_p)}
        names = [str(x).strip() for x in names]

        ref_p = Path(str(pytorch_ref_path)).expanduser()
        if not ref_p.exists() and not ref_p.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            alt = (repo_root / ref_p).resolve()
            if alt.exists():
                ref_p = alt
        if not ref_p.exists():
            return {"status": "FAIL", "error": "pytorch_ref not found", "pytorch_ref_path": str(ref_p)}
        try:
            obj = torch.load(str(ref_p), map_location="cpu")
        except Exception as e:
            return {"status": "FAIL", "error": f"torch.load failed: {e}", "pytorch_ref_path": str(ref_p)}
        if not isinstance(obj, dict):
            return {"status": "FAIL", "error": f"pytorch_ref must be a dict, got: {type(obj).__name__}", "pytorch_ref_path": str(ref_p)}

        bin_p = meta_p.with_suffix(".bin")
        tensors_meta: List[Dict[str, Any]] = []
        offset = 0
        chunks: List[bytes] = []
        for name in names:
            t = obj.get(name)
            if not isinstance(t, torch.Tensor):
                return {"status": "FAIL", "error": f"missing tensor in pytorch_ref: {name}", "pytorch_ref_path": str(ref_p)}
            t_cpu = t.detach().cpu().contiguous()
            type_name = _torch_dtype_to_type_name(t_cpu.dtype)
            if type_name is None:
                t_cpu = t_cpu.to(dtype=torch.float32)
                type_name = "f32"

            shape = list(t_cpu.shape)
            while len(shape) < 4:
                shape.append(1)
            if len(shape) > 4:
                return {"status": "FAIL", "error": f"tensor rank > 4 not supported: {name}", "shape": shape}

            try:
                raw = t_cpu.numpy().tobytes()  # type: ignore[call-arg]
            except Exception as e:
                return {"status": "FAIL", "error": f"tensor tobytes failed: {e}", "tensor": name}
            nbytes = len(raw)
            chunks.append(raw)
            tensors_meta.append(
                {
                    "name": str(name),
                    "type_name": str(type_name),
                    "ne": [int(x) for x in shape],
                    "offset_bytes": int(offset),
                    "nbytes": int(nbytes),
                }
            )
            offset += int(nbytes)

        try:
            bin_p.write_bytes(b"".join(chunks))
        except Exception as e:
            return {"status": "FAIL", "error": f"write bin failed: {e}", "bin_path": str(bin_p)}

        bundle = {
            "tap_id": meta.get("tap_id"),
            "unit": meta.get("unit"),
            "tensor_names": names,
            "shape_match": bool(meta.get("shape_match")),
            "dtype_match": bool(meta.get("dtype_match")),
            "dumped": True,
            "type_name": "f32",
            "ne": [1, 1, 1, 1],
            "bin_path": str(bin_p.resolve()),
            "tensors": tensors_meta,
        }
        try:
            meta_p.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            return {"status": "FAIL", "error": f"write meta failed: {e}", "meta_json_path": str(meta_p)}

        return {"status": "PASS", "bin_path": str(bin_p), "meta_json_path": str(meta_p), "tensors": len(tensors_meta)}

    def _score_name_match(actual: str, want: str) -> int:
        a = str(actual or "")
        w = str(want or "")
        if a == w:
            return 3
        if a.endswith(w) and w != "":
            return 2
        if w != "" and w in a:
            return 1
        return 0

    def _collect_tap_meta_jsons(root: Path, *, limit: int = 2000) -> List[Path]:
        if not root.exists():
            return []
        dirs: List[Path] = []
        for d in root.rglob("ggml_tensor_taps"):
            if d.is_dir():
                dirs.append(d)
        dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        metas: List[Path] = []
        for d in dirs[:8]:
            for fp in d.glob("tap_*.json"):
                if fp.is_file():
                    metas.append(fp)
        metas.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return metas[: int(limit)]

    def _resolve_tap_meta_by_tap_id(tap_id: str) -> Optional[str]:
        search_root = Path(str(tap_search_dir)).expanduser() if tap_search_dir is not None else Path.cwd()
        want = f"tap_{str(tap_id).strip()}.json"
        candidates: List[Path] = []
        for d in search_root.rglob("ggml_tensor_taps"):
            if not d.is_dir():
                continue
            p = (d / want)
            if p.is_file():
                candidates.append(p)
        if len(candidates) == 0:
            for d in search_root.rglob("ggml_tensor_taps"):
                if not d.is_dir():
                    continue
                for p in d.glob(f"tap_*{str(tap_id).strip()}*.json"):
                    if p.is_file():
                        candidates.append(p)
        if len(candidates) == 0:
            return None
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(candidates[0])

    def _resolve_meta_paths_for_tensor_names(tensor_names: List[str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        search_root = Path(str(tap_search_dir)).expanduser() if tap_search_dir is not None else Path.cwd()
        meta_jsons = _collect_tap_meta_jsons(search_root)
        parsed: List[Tuple[Path, Dict[str, Any]]] = []
        for mp in meta_jsons:
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not bool(m.get("dumped")):
                continue
            parsed.append((mp, m))
        resolved: Dict[str, str] = {}
        debug: Dict[str, Any] = {"search_root": str(search_root), "candidates": len(parsed)}
        for want in tensor_names:
            best: Optional[Tuple[int, float, Path]] = None
            for mp, m in parsed:
                score = _score_name_match(str(m.get("name") or ""), str(want))
                if score <= 0:
                    continue
                try:
                    mt = float(mp.stat().st_mtime)
                except Exception:
                    mt = 0.0
                cand = (int(score), float(mt), mp)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
            if best is not None:
                resolved[str(want)] = str(best[2])
        return resolved, debug

    def _tap_spec_to_tensor_names(spec: Union[str, Dict[str, Any]], *, fallback: str) -> List[str]:
        if isinstance(spec, dict):
            tn = spec.get("tensor_names")
            if isinstance(tn, list) and all(isinstance(x, str) and x.strip() != "" for x in tn):
                return [str(x).strip() for x in tn]
        return [str(fallback)]

    def _select_outputs(obj: Any, names: List[str]) -> Dict[str, torch.Tensor]:
        if isinstance(obj, torch.Tensor):
            if len(names) != 1:
                raise RuntimeError(f"single Tensor output but {len(names)} outputs requested")
            return {names[0]: obj}
        if isinstance(obj, dict):
            out: Dict[str, torch.Tensor] = {}
            for n in names:
                v = obj.get(n)
                if isinstance(v, torch.Tensor):
                    out[n] = v
            if len(out) == len(names):
                return out
            if len(names) == 1:
                only = obj.get(names[0])
                if isinstance(only, torch.Tensor):
                    return {names[0]: only}
                for v in obj.values():
                    if isinstance(v, torch.Tensor):
                        return {names[0]: v}
            raise RuntimeError(f"dict output missing keys: {names}")
        if isinstance(obj, (list, tuple)):
            if len(obj) != len(names):
                raise RuntimeError(f"tuple/list output length {len(obj)} != expected {len(names)}")
            out: Dict[str, torch.Tensor] = {}
            for n, v in zip(names, obj):
                if not isinstance(v, torch.Tensor):
                    raise RuntimeError(f"output {n} is not Tensor: {type(v).__name__}")
                out[n] = v
            return out
        raise RuntimeError(f"unsupported output type: {type(obj).__name__}")

    in_names = _tap_spec_to_tensor_names(input_tap_meta, fallback="x")
    out_names = _tap_spec_to_tensor_names(output_tap_meta, fallback="out")

    input_meta_paths: Dict[str, str]
    output_meta_paths: Dict[str, str]
    debug_in: Dict[str, Any] = {}
    debug_out: Dict[str, Any] = {}

    if isinstance(input_tap_meta, str) and isinstance(output_tap_meta, str):
        inp = load_ggml_tap_tensor(input_tap_meta)
        out_ref = load_ggml_tap_tensor(output_tap_meta)
        if str(inp.get("status")) != "PASS":
            return {"status": "FAIL", "error": f"input tap load failed: {inp.get('error')}", "input": inp, "output_ref": out_ref}
        if str(out_ref.get("status")) != "PASS":
            return {"status": "FAIL", "error": f"output tap load failed: {out_ref.get('error')}", "input": inp, "output_ref": out_ref}
        x = inp["tensor"]
        y_ref = out_ref["tensor"]

        run = _run_pytorch_subgraph_from_file(pytorch_path, inputs={"x": x})
        if str(run.get("status")) != "PASS":
            return {"status": "FAIL", "error": "pytorch subgraph run failed", "run": run}
        y = run["output"]
        try:
            out_map = _select_outputs(y, ["out"])
        except Exception as e:
            return {"status": "FAIL", "error": str(e), "run": run}
        y0 = out_map["out"]

        if y0.shape != y_ref.shape:
            return {"status": "FAIL", "error": "shape mismatch", "shape": {"got": list(y0.shape), "ref": list(y_ref.shape)}, "module": run.get("module_name")}

        y_f = y0.detach().to(dtype=torch.float32)
        y_ref_f = y_ref.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(y_f).all().item()) or not bool(torch.isfinite(y_ref_f).all().item()):
            return {
                "status": "FAIL",
                "error": "non_finite_detected",
                "module": run.get("module_name"),
            }
        diff = (y_f - y_ref_f).abs()
        max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
        denom = y_ref_f.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
        ok = bool(torch.allclose(y_f, y_ref_f, atol=float(atol), rtol=float(rtol)))
        return {
            "status": "PASS" if ok else "FAIL",
            "module": run.get("module_name"),
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs": float(max_abs),
            "max_rel": float(max_rel),
            "input_tap_meta": str(input_tap_meta),
            "output_tap_meta": str(output_tap_meta),
        }

    input_tap_id = str(input_tap_meta.get("tap_id") or "").strip() if isinstance(input_tap_meta, dict) else ""
    output_tap_id = str(output_tap_meta.get("tap_id") or "").strip() if isinstance(output_tap_meta, dict) else ""
    input_bundle = _resolve_tap_meta_by_tap_id(input_tap_id) if input_tap_id != "" else None
    output_bundle = _resolve_tap_meta_by_tap_id(output_tap_id) if output_tap_id != "" else None

    inputs: Dict[str, torch.Tensor] = {}
    input_loads: Dict[str, Any] = {}
    if input_bundle is not None:
        mat = _materialize_tap_bundle_from_torch_ref(input_bundle, pytorch_ref_path=pytorch_path)
        if str(mat.get("status")) != "PASS":
            return {"status": "FAIL", "error": "input tap materialize failed", "materialize": mat}
        for name in in_names:
            r = _load_tensor_from_meta_bundle(input_bundle, tensor_name=str(name))
            input_loads[name] = {"status": r.get("status"), "meta_json": str(input_bundle), "error": r.get("error")}
            if str(r.get("status")) != "PASS":
                return {"status": "FAIL", "error": f"input tap load failed: {name}", "input": input_loads}
            inputs[name] = r["tensor"]
        input_meta_paths = {str(k): str(input_bundle) for k in in_names}
        debug_in = {"search_root": str(Path(str(tap_search_dir)).expanduser() if tap_search_dir is not None else Path.cwd()), "resolved_by": "tap_id", "tap_id": input_tap_id}
    else:
        input_meta_paths, debug_in = _resolve_meta_paths_for_tensor_names(in_names)
        missing_in = [n for n in in_names if n not in input_meta_paths]
        if len(missing_in) > 0:
            return {"status": "SKIP", "reason": "tap_not_found", "missing": {"inputs": missing_in, "outputs": []}, "tap_search": {"input": debug_in}}
        for name, mp in input_meta_paths.items():
            r = load_ggml_tap_tensor(mp)
            input_loads[name] = {"status": r.get("status"), "meta_json": mp, "error": r.get("error")}
            if str(r.get("status")) != "PASS":
                return {"status": "FAIL", "error": f"input tap load failed: {name}", "input": input_loads}
            inputs[name] = r["tensor"]

    out_refs: Dict[str, torch.Tensor] = {}
    output_loads: Dict[str, Any] = {}
    if output_bundle is not None:
        mat = _materialize_tap_bundle_from_torch_ref(output_bundle, pytorch_ref_path=pytorch_path)
        if str(mat.get("status")) != "PASS":
            return {"status": "FAIL", "error": "output tap materialize failed", "materialize": mat}
        for name in out_names:
            r = _load_tensor_from_meta_bundle(output_bundle, tensor_name=str(name))
            output_loads[name] = {"status": r.get("status"), "meta_json": str(output_bundle), "error": r.get("error")}
            if str(r.get("status")) != "PASS":
                return {"status": "FAIL", "error": f"output tap load failed: {name}", "output_ref": output_loads}
            out_refs[name] = r["tensor"]
        output_meta_paths = {str(k): str(output_bundle) for k in out_names}
        debug_out = {"search_root": str(Path(str(tap_search_dir)).expanduser() if tap_search_dir is not None else Path.cwd()), "resolved_by": "tap_id", "tap_id": output_tap_id}
    else:
        output_meta_paths, debug_out = _resolve_meta_paths_for_tensor_names(out_names)
        missing_out = [n for n in out_names if n not in output_meta_paths]
        if len(missing_out) > 0:
            return {"status": "SKIP", "reason": "tap_not_found", "missing": {"inputs": [], "outputs": missing_out}, "tap_search": {"output": debug_out}}
        for name, mp in output_meta_paths.items():
            r = load_ggml_tap_tensor(mp)
            output_loads[name] = {"status": r.get("status"), "meta_json": mp, "error": r.get("error")}
            if str(r.get("status")) != "PASS":
                return {"status": "FAIL", "error": f"output tap load failed: {name}", "output_ref": output_loads}
            out_refs[name] = r["tensor"]

    run = _run_pytorch_subgraph_from_file(pytorch_path, inputs=inputs)
    if str(run.get("status")) != "PASS":
        return {"status": "FAIL", "error": "pytorch subgraph run failed", "run": run}

    try:
        out_map = _select_outputs(run.get("output"), out_names)
    except Exception as e:
        return {"status": "FAIL", "error": str(e), "run": run}

    max_abs = 0.0
    max_rel = 0.0
    per_output: Dict[str, Any] = {}
    ok = True
    for name in out_names:
        y = out_map[name]
        y_ref = out_refs[name]
        if y.shape != y_ref.shape:
            return {
                "status": "FAIL",
                "error": "shape mismatch",
                "tensor": name,
                "shape": {"got": list(y.shape), "ref": list(y_ref.shape)},
                "module": run.get("module_name"),
            }
        y_f = y.detach().to(dtype=torch.float32)
        y_ref_f = y_ref.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(y_f).all().item()) or not bool(torch.isfinite(y_ref_f).all().item()):
            return {
                "status": "FAIL",
                "error": "non_finite_detected",
                "tensor": str(name),
                "module": run.get("module_name"),
            }
        diff = (y_f - y_ref_f).abs()
        ma = float(diff.max().item()) if diff.numel() > 0 else 0.0
        denom = y_ref_f.abs().clamp_min(1e-12)
        mr = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
        ok = bool(ok and torch.allclose(y_f, y_ref_f, atol=float(atol), rtol=float(rtol)))
        max_abs = float(max(max_abs, ma))
        max_rel = float(max(max_rel, mr))
        per_output[name] = {"max_abs": float(ma), "max_rel": float(mr), "shape": [int(x) for x in list(y.shape)]}

    return {
        "status": "PASS" if ok else "FAIL",
        "module": run.get("module_name"),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
        "input_tap_meta_paths": [str(input_meta_paths[n]) for n in in_names],
        "output_tap_meta_paths": [str(output_meta_paths[n]) for n in out_names],
        "per_output": per_output,
    }


def _internal_skvm_verify_file(
    *,
    input_path: str,
    input_shape: Dict[str, List[int]],
    dtype: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    code = Path(input_path).read_text(encoding="utf-8")
    try:
        m = _load_module_from_code(code)
    except Exception as e:
        return {"status": "failed", "errors": [str(e)], "traceback": traceback.format_exc()}

    if len(input_shape) == 0:
        return {"status": "failed", "errors": ["missing input_shape"]}

    first_name = next(iter(input_shape.keys()))
    shp = input_shape[first_name]
    if not isinstance(shp, list) or len(shp) == 0:
        return {"status": "failed", "errors": ["invalid input_shape"]}
    dt = torch.float16 if str(dtype).lower() in ("fp16", "float16") else torch.bfloat16 if str(dtype).lower() in ("bf16", "bfloat16") else torch.float32
    x = torch.randn([int(v) for v in shp], dtype=dt)

    try:
        m = m.to(dtype=dt)
        m.eval()
        with torch.no_grad():
            out = m(x)
    except Exception as e:
        return {"status": "failed", "errors": [f"forward failed: {e}"], "traceback": traceback.format_exc()}

    ops: List[str] = []
    try:
        gm = torch.fx.symbolic_trace(m)
        for n in gm.graph.nodes:
            if n.op == "call_function":
                try:
                    ops.append(str(getattr(n.target, "__name__", str(n.target))))
                except Exception:
                    ops.append(str(n.target))
            elif n.op in ("call_module", "call_method"):
                ops.append(str(n.target))
    except Exception:
        gm = None

    artifacts: Dict[str, Any] = {"status": "SKIP"}
    if gm is not None:
        try:
            out_dir = str(Path(input_path).resolve().parent)
            fx_path = str(Path(out_dir) / "skvm_normalized.fx.txt")
            Path(fx_path).write_text(str(gm.graph), encoding="utf-8")
            example_inputs = {str(first_name): x}
            ir = _build_constraints_and_ir(gm=gm, example_inputs=example_inputs, out_dir=out_dir)
            artifacts = {
                "status": "PASS",
                "normalized_pytorch_path": str(input_path),
                "fx_graph_path": fx_path,
                "constraints_path": str(ir.get("constraints_path")),
                "primitive_ir_path": str(ir.get("primitive_ir_path")),
                "shape_prop": {"status": "PASS" if ir.get("shape_prop_error") is None else "FAIL", "error": ir.get("shape_prop_error")},
                "op_histogram": ir.get("op_histogram"),
            }
        except Exception as e:
            artifacts = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}

    result = {
        "status": "success",
        "shape_inference": {"inputs": {str(first_name): [int(v) for v in shp]}, "outputs": {"out": _as_shape_dict(out)}},
        "operators": ops,
        "normalized_code": str(code),
        "normalized_subgraph": artifacts,
        "memory_usage_bytes": 0,
        "elapsed_s": float(time.perf_counter() - t0),
        "engine": "internal_skvm",
    }
    return result


def _resolve_skvm_cli(skvm_cli: str) -> Optional[str]:
    cli = str(skvm_cli or "").strip()
    if cli == "":
        cli = "skvm"
    p = Path(cli).expanduser()
    if p.is_file():
        return str(p)
    resolved = shutil.which(cli)
    if resolved is not None:
        return str(resolved)
    if cli == "skvm":
        candidates = [
            Path("~/.local/share/skvm/bin/skvm").expanduser(),
            Path("~/.local/bin/skvm").expanduser(),
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
    return None


def _probe_help(cli_path: str, *, timeout_s: int = 5) -> str:
    try:
        p = subprocess.run([str(cli_path), "--help"], capture_output=True, text=True, timeout=int(timeout_s), check=False)
        return (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        return ""


def _detect_cli_kind(cli_path: str) -> Dict[str, Any]:
    help_text = _probe_help(cli_path)
    t = help_text.lower()
    supports_verify = " verify" in t or "\nverify" in t
    looks_like_skillvm = ("aot-compile" in t) or ("jit-optimize" in t) or ("skvm config" in t) or ("skvm_cache" in t)
    return {"supports_verify": bool(supports_verify), "looks_like_skillvm": bool(looks_like_skillvm), "help": help_text[-2000:]}


def _probe_version(cli_path: str, *, timeout_s: int = 5) -> str:
    for cmd in ([str(cli_path), "--version"], [str(cli_path), "version"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_s), check=False)
            out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
            if out != "":
                return out[-2000:]
        except Exception:
            continue
    return ""


def _try_install_skillvm_skvm(*, timeout_s: int = 900) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    if shutil.which("curl") is not None:
        cmd = "curl -fsSL https://skillvm.ai/install.sh | sh"
        try:
            p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=int(timeout_s), check=False)
            logs.append(
                {
                    "method": "curl_install_sh",
                    "returncode": int(p.returncode),
                    "stdout": (p.stdout or "")[-2000:],
                    "stderr": (p.stderr or "")[-2000:],
                }
            )
            if int(p.returncode) == 0:
                return {"status": "success", "attempts": logs}
        except Exception as e:
            logs.append({"method": "curl_install_sh", "status": "failed", "error": str(e)})

    if shutil.which("npm") is not None:
        cmd = "npm i -g @ipads-skvm/skvm"
        try:
            p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=int(timeout_s), check=False)
            logs.append(
                {
                    "method": "npm_global_install",
                    "returncode": int(p.returncode),
                    "stdout": (p.stdout or "")[-2000:],
                    "stderr": (p.stderr or "")[-2000:],
                }
            )
            if int(p.returncode) == 0:
                return {"status": "success", "attempts": logs}
        except Exception as e:
            logs.append({"method": "npm_global_install", "status": "failed", "error": str(e)})

    return {"status": "failed", "attempts": logs}


def _ensure_skillvm_cli(*, skvm_cli: str, auto_install: bool, timeout_s: int) -> Dict[str, Any]:
    resolved_cli = _resolve_skvm_cli(str(skvm_cli))
    install_result: Optional[Dict[str, Any]] = None
    if resolved_cli is None and bool(auto_install) and str(skvm_cli).strip() in ("", "skvm"):
        install_result = _try_install_skillvm_skvm(timeout_s=int(timeout_s))
        resolved_cli = _resolve_skvm_cli("skvm")
    if resolved_cli is None:
        return {"status": "missing", "resolved_cli": None, "install": install_result}
    kind = _detect_cli_kind(str(resolved_cli))
    toolchain = {
        "path": str(resolved_cli),
        "version": _probe_version(str(resolved_cli)),
        "supports_verify": bool(kind.get("supports_verify")),
        "looks_like_skillvm": bool(kind.get("looks_like_skillvm")),
        "help_tail": str(kind.get("help") or ""),
    }
    if install_result is not None:
        toolchain["install"] = install_result
    return {"status": "ok", "resolved_cli": str(resolved_cli), "toolchain": toolchain}


def _run_skillvm_cmd(cmd: List[str], *, env: Dict[str, str], timeout_s: int) -> Dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_s), check=False, env=env)
        return {
            "returncode": int(p.returncode),
            "stdout": (p.stdout or "")[-4000:],
            "stderr": (p.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"returncode": 125, "stdout": "", "stderr": str(e)}


def _write_skill_md_from_source(*, source_path: str, skill_path: str) -> None:
    code = Path(str(source_path)).read_text(encoding="utf-8")
    text = (
        "# cgc-normalize-pytorch-subgraph\n\n"
        "This is an auto-generated skill wrapper. The payload below is a Python/PyTorch module source.\n"
        "The compiler should rewrite it to be deterministic, side-effect-free, and stable across runs.\n\n"
        "## Payload\n\n"
        "```python\n"
        f"{code}\n"
        "```\n"
    )
    Path(skill_path).write_text(text, encoding="utf-8")


def _find_newest_skill_md(root: str) -> Optional[str]:
    p = Path(str(root)).expanduser()
    if not p.exists():
        return None
    best: Optional[Path] = None
    best_m = -1.0
    for fp in p.rglob("SKILL.md"):
        try:
            m = float(fp.stat().st_mtime)
        except Exception:
            continue
        if m > best_m:
            best_m = m
            best = fp
    return str(best) if best is not None else None


def _extract_first_python_code_block(md_text: str) -> Optional[str]:
    text = str(md_text or "")
    if "```" not in text:
        return None
    if "```python" in text:
        try:
            chunk = text.split("```python", 1)[1]
            code = chunk.split("```", 1)[0]
            code = code.strip()
            return code if code != "" else None
        except Exception:
            pass
    try:
        chunk = text.split("```", 1)[1]
        code = chunk.split("```", 1)[0]
        code = code.strip()
        return code if code != "" else None
    except Exception:
        return None


def _try_trace_fx_from_pytorch_code(
    *,
    pytorch_code: str,
    input_shape: List[int],
    dtype: str,
    out_dir: str,
) -> Dict[str, Any]:
    p = Path(str(out_dir)).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    normalized_path = str(p / "skvm_normalized.py")
    Path(normalized_path).write_text(str(pytorch_code), encoding="utf-8")

    try:
        m = _load_module_from_code(str(pytorch_code))
        dt = (
            torch.float16
            if str(dtype).lower() in ("fp16", "float16")
            else torch.bfloat16
            if str(dtype).lower() in ("bf16", "bfloat16")
            else torch.float32
        )
        x = torch.randn([int(v) for v in list(input_shape)], dtype=dt)
        m = m.to(dtype=dt)
        m.eval()
        with torch.no_grad():
            out = m(x)
        gm = torch.fx.symbolic_trace(m)
        fx_path = str(p / "skvm_normalized.fx.txt")
        Path(fx_path).write_text(str(gm.graph), encoding="utf-8")
        ir = _build_constraints_and_ir(gm=gm, example_inputs={"x": x}, out_dir=str(p))
        return {
            "status": "PASS",
            "normalized_pytorch_path": normalized_path,
            "fx_graph_path": fx_path,
            "shape_inference": {"inputs": {"x": [int(v) for v in list(input_shape)]}, "outputs": {"out": _as_shape_dict(out)}},
            "constraints_path": str(ir.get("constraints_path")),
            "primitive_ir_path": str(ir.get("primitive_ir_path")),
            "shape_prop": {"status": "PASS" if ir.get("shape_prop_error") is None else "FAIL", "error": ir.get("shape_prop_error")},
            "op_histogram": ir.get("op_histogram"),
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "normalized_pytorch_path": normalized_path,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def skillvm_profile_and_aot_compile(
    *,
    input_path: str,
    work_dir: str,
    skvm_cli: str = "skvm",
    auto_install: bool = False,
    timeout_s: int = 900,
    target_model: str,
    compiler_model: str,
    adapter: str = "bare-agent",
    input_shape: Optional[List[int]] = None,
    dtype: str = "fp16",
) -> Dict[str, Any]:
    if str(target_model).strip() == "" or str(compiler_model).strip() == "":
        return {"status": "SKIP", "reason": "missing skillvm_target_model or skillvm_compiler_model"}

    ensure = _ensure_skillvm_cli(skvm_cli=str(skvm_cli), auto_install=bool(auto_install), timeout_s=int(timeout_s))
    resolved = ensure.get("resolved_cli")
    toolchain = ensure.get("toolchain")
    if ensure.get("status") != "ok" or resolved is None:
        return {"status": "SKIP", "reason": "skvm not available", "ensure": ensure}

    if not bool((toolchain or {}).get("looks_like_skillvm")):
        return {"status": "SKIP", "reason": "skvm is not skillvm.ai SkVM", "toolchain": toolchain}

    wd = Path(str(work_dir)).expanduser()
    wd.mkdir(parents=True, exist_ok=True)
    cache_root = wd / "skvm_cache"
    skill_dir = wd / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = str(skill_dir / "SKILL.md")
    _write_skill_md_from_source(source_path=str(input_path), skill_path=skill_path)

    env = os.environ.copy()
    env["SKVM_CACHE"] = str(cache_root)
    env.setdefault("SKVM_PROFILES_DIR", str(cache_root / "profiles"))
    env.setdefault("SKVM_PROPOSALS_DIR", str(cache_root / "proposals"))
    env.setdefault("SKVM_LOGS_DIR", str(cache_root / "log"))

    profile_cmd = [str(resolved), "profile", f"--model={str(target_model)}", f"--adapter={str(adapter)}"]
    prof = _run_skillvm_cmd(profile_cmd, env=env, timeout_s=int(timeout_s))

    aot_cmd = [
        str(resolved),
        "aot-compile",
        f"--skill={skill_path}",
        f"--model={str(target_model)}",
        f"--adapter={str(adapter)}",
        "--pass=1",
        f"--compiler-model={str(compiler_model)}",
    ]
    aot = _run_skillvm_cmd(aot_cmd, env=env, timeout_s=int(timeout_s))

    newest = _find_newest_skill_md(str(cache_root / "proposals"))
    out_skill = None
    if newest is not None:
        out_skill = str(wd / "skill_aot_compiled.SKILL.md")
        try:
            Path(out_skill).write_text(Path(newest).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            out_skill = None

    normalized = {"status": "SKIP"}
    if out_skill is not None and isinstance(input_shape, list) and len(input_shape) > 0:
        try:
            md = Path(out_skill).read_text(encoding="utf-8")
            code = _extract_first_python_code_block(md)
            if code is None:
                normalized = {"status": "FAIL", "reason": "no python code block found in aot-compiled SKILL.md"}
            else:
                normalized = _try_trace_fx_from_pytorch_code(
                    pytorch_code=code,
                    input_shape=[int(v) for v in input_shape],
                    dtype=str(dtype),
                    out_dir=str(wd),
                )
        except Exception as e:
            normalized = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}

    ok = (int(prof.get("returncode", 1)) == 0) and (int(aot.get("returncode", 1)) == 0) and (out_skill is not None) and str(normalized.get("status")) == "PASS"
    return {
        "status": "PASS" if bool(ok) else "FAIL",
        "toolchain": toolchain,
        "work_dir": str(wd),
        "cache_root": str(cache_root),
        "skill_input": skill_path,
        "profile": {"cmd": profile_cmd, **prof},
        "aot_compile": {"cmd": aot_cmd, **aot},
        "aot_compiled_skill": out_skill,
        "normalized_subgraph": normalized,
        "note": "This runs SkVM profile+aot-compile over an auto-generated SKILL wrapper that embeds the input PyTorch source.",
    }


def skvm_verify(
    *,
    pytorch_code: Optional[str] = None,
    input_path: Optional[str] = None,
    input_shape: Dict[str, List[int]],
    dtype: str = "fp16",
    timeout_s: int = 30,
    skvm_cli: str = "skvm",
    auto_install: bool = False,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    work = Path(work_dir or "/tmp/skvm_work")
    work.mkdir(parents=True, exist_ok=True)

    code_file = work / "skvm_input.py"
    result_file = work / "skvm_result.json"

    if pytorch_code is not None:
        code_file.write_text(str(pytorch_code), encoding="utf-8")
        input_arg = str(code_file)
    elif input_path is not None:
        input_arg = str(input_path)
    else:
        return {"status": "failed", "errors": ["missing pytorch_code or input_path"]}

    cmd = [
        str(skvm_cli),
        "verify",
        "--input",
        str(input_arg),
        "--input-shape",
        json.dumps(input_shape, ensure_ascii=False),
        "--dtype",
        str(dtype),
        "--output-json",
        str(result_file),
    ]

    ensure = _ensure_skillvm_cli(skvm_cli=str(skvm_cli), auto_install=bool(auto_install), timeout_s=max(300, int(timeout_s) * 10))
    resolved_cli = ensure.get("resolved_cli")
    toolchain = ensure.get("toolchain") if isinstance(ensure.get("toolchain"), dict) else None

    if resolved_cli is None:
        try:
            out = _internal_skvm_verify_file(input_path=str(input_arg), input_shape=input_shape, dtype=str(dtype))
            result_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            out["external_skvm_toolchain"] = toolchain
            out["external_skvm_ensure"] = ensure
            return out
        except Exception as e:
            return {"status": "failed", "errors": [f"internal skvm failed: {e}"], "traceback": traceback.format_exc()}

    if not bool((toolchain or {}).get("supports_verify")):
        out = _internal_skvm_verify_file(input_path=str(input_arg), input_shape=input_shape, dtype=str(dtype))
        out["external_skvm_toolchain"] = toolchain
        out["external_skvm_ensure"] = ensure
        result_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    try:
        cmd[0] = str(resolved_cli)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_s))
    except FileNotFoundError:
        return {"status": "failed", "errors": [f"skvm cli not found: {skvm_cli}"]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "errors": ["SkVM timeout"]}

    if not result_file.exists():
        return {
            "status": "failed",
            "errors": ["no output"],
            "returncode": int(getattr(proc, "returncode", 1)),
            "stdout": getattr(proc, "stdout", ""),
            "stderr": getattr(proc, "stderr", ""),
        }

    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"invalid output json: {e}"],
            "returncode": int(getattr(proc, "returncode", 1)),
            "stdout": getattr(proc, "stdout", ""),
            "stderr": getattr(proc, "stderr", ""),
        }

    if isinstance(result, dict):
        result.setdefault("returncode", int(getattr(proc, "returncode", 0)))
        result.setdefault("stdout", getattr(proc, "stdout", ""))
        result.setdefault("stderr", getattr(proc, "stderr", ""))
        result.setdefault("external_skvm_toolchain", toolchain)
        result.setdefault("external_skvm_ensure", ensure)
    return result


def _cli_main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="skvm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--input", required=True)
    v.add_argument("--input-shape", required=True)
    v.add_argument("--dtype", default="fp16")
    v.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "verify":
        try:
            input_shape = json.loads(str(args.input_shape))
        except Exception as e:
            out = {"status": "failed", "errors": [f"invalid --input-shape json: {e}"]}
            Path(str(args.output_json)).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            return 2
        out = _internal_skvm_verify_file(input_path=str(args.input), input_shape=input_shape, dtype=str(args.dtype))
        Path(str(args.output_json)).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if str(out.get("status")) == "success" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))


def llama_cpp_to_pytorch_to_skvm(
    *,
    llama_cpp_code: str,
    input_shape: Dict[str, List[int]],
    llm_translate_func: Callable[[str], str],
    dtype: str = "fp16",
    timeout_s: int = 30,
    skvm_cli: str = "skvm",
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    raw_pytorch = llm_translate_func(str(llama_cpp_code))
    return skvm_verify(
        pytorch_code=raw_pytorch,
        input_shape=input_shape,
        dtype=dtype,
        timeout_s=timeout_s,
        skvm_cli=skvm_cli,
        work_dir=work_dir,
    )


def skvm_output_to_magi_ir(skvm_result: Dict[str, Any]) -> Dict[str, Any]:
    status = str(skvm_result.get("status", ""))
    if status != "success":
        raise RuntimeError("SkVM 校验失败，禁止进入编译")

    shape_inf = skvm_result.get("shape_inference") or {}
    inputs = (shape_inf.get("inputs") or {}).copy()
    outputs = (shape_inf.get("outputs") or {}).copy()

    ops = skvm_result.get("operators") or []
    if not isinstance(ops, list):
        ops = []
    ops = [str(x) for x in ops]

    nodes: List[Dict[str, Any]] = []
    for i, op in enumerate(ops):
        nodes.append({"id": int(i), "op": str(op)})

    return {
        "status": "success",
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "operators": ops,
        "memory_usage_bytes": int(skvm_result.get("memory_usage_bytes", 0) or 0),
    }
