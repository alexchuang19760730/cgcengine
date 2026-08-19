from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "build_bundle": (".builder", "build_bundle"),
    "run_bundle": (".runner", "run_bundle"),
    "verify_both": (".verify", "verify_both"),
    "run_m1_gate": (".m1_m6_pipeline_gates", "run_m1_gate"),
    "run_m2_gate": (".m1_m6_pipeline_gates", "run_m2_gate"),
    "run_m3_gate": (".m1_m6_pipeline_gates", "run_m3_gate"),
    "run_m4_gate_internal": (".m1_m6_pipeline_gates", "run_m4_gate_internal"),
    "run_m5_gate": (".m1_m6_pipeline_gates", "run_m5_gate"),
    "run_m6_gate": (".m1_m6_pipeline_gates", "run_m6_gate"),
    "run_m7_gate": (".m7_gate", "run_m7_gate"),
    "run_m72_gate": (".m72_gate", "run_m72_gate"),
    "run_m73_gate": (".m73_gate", "run_m73_gate"),
    "run_m74_gate": (".m74_dflash_kda_gate", "run_m74_gate"),
    "run_upkg21_gate": (".upkg21_gate", "run_upkg21_gate"),
    "run_upkg21_rerun_gate": (".upkg21_gate", "run_upkg21_rerun_gate"),
    "run_m75_trueorthokda_active_runtime": (
        ".m75_trueorthokda_active_runtime",
        "run_m75_trueorthokda_active_runtime",
    ),
    "run_m75_trueorthokda_active_gate": (
        ".m75_trueorthokda_active_runtime",
        "run_m75_trueorthokda_active_gate",
    ),
    "run_m75_gate": (
        ".m75_trueorthokda_active_runtime",
        "run_m75_trueorthokda_active_gate",
    ),
    "run_m76_gate": (".m76_gate", "run_m76_gate"),
    "run_m77_gate": (".m77_gate", "run_m77_gate"),
    "run_upkg37_gate": (".m77_gate", "run_upkg37_gate"),
    "run_m78_gate": (".m78_gate", "run_m78_gate"),
    "run_upkg38_gate": (".m78_gate", "run_upkg38_gate"),
    "run_upkg39_gate": (".upkg39_gate", "run_upkg39_gate"),
    "build_upkg39_fusionroute_manifest": (
        ".upkg39_fusionroute_manifest_builder",
        "build_upkg39_fusionroute_manifest",
    ),
    "run_m79_gate": (".m79_gate", "run_m79_gate"),
    "run_upkg40_gate": (".m79_gate", "run_upkg40_gate"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
