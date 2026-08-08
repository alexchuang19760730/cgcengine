"""colossalai_runtime_candidate_verifier.py — Gate 2.0 ColossalAI runtime candidate verifier"""
from __future__ import annotations

import sys
from pathlib import Path

from .base import BaseVerifier, VerificationStatus


workspace_root = Path(__file__).resolve().parents[3]
for candidate in (workspace_root, workspace_root / "ComputeGraphCompiler-main"):
    raw = str(candidate)
    if raw not in sys.path:
        sys.path.insert(0, raw)


class ColossalAIRuntimeCandidateVerifier(BaseVerifier):
    capability = "colossalai_distributed_runtime_candidate"

    def verify(self):
        start = self._start()
        try:
            pipeline_source = (workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "pipeline.py").read_text(encoding="utf-8")
            m76_source = (workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "product" / "m76_gate.py").read_text(encoding="utf-8")

            requested_runtime = "colossalai" if '"requested_distributed_runtime"' in m76_source and '"colossalai"' in m76_source else ""
            distributed_backend = "colossalai" if 'distributed_runtime_backend' in m76_source and '"colossalai"' in m76_source else ""
            probe = {
                "backend": "colossalai" if '"backend": "colossalai"' in m76_source else "",
                "plugin": "HybridParallelPlugin" if "HybridParallelPlugin" in m76_source else "",
                "source_contract": True,
            }
            has_booster_import = "from colossalai.booster import Booster" in pipeline_source and "from colossalai.booster import Booster" in m76_source
            has_plugin_import = "HybridParallelPlugin" in pipeline_source and "HybridParallelPlugin" in m76_source
            has_topology_compute = "compute_parallel_topology" in pipeline_source
            has_dist_init = "init_distributed_for_training" in pipeline_source

            self._add_metric("requested_distributed_runtime", requested_runtime)
            self._add_metric("distributed_runtime_backend", distributed_backend)
            self._add_metric("probe", probe)
            self._add_metric("has_booster_import", has_booster_import)
            self._add_metric("has_plugin_import", has_plugin_import)
            self._add_metric("has_topology_compute", has_topology_compute)
            self._add_metric("has_dist_init", has_dist_init)

            self._add_evidence(
                "✓ colossalai contract:"
                f" requested={requested_runtime} backend={distributed_backend}"
                f" probe_backend={probe.get('backend')} probe_plugin={probe.get('plugin')}"
            )
            self._add_evidence(
                "✓ colossalai wrapper surface:"
                f" booster={has_booster_import} plugin={has_plugin_import}"
                f" topology={has_topology_compute} dist_init={has_dist_init}"
            )

            if requested_runtime != "colossalai":
                return self._finish(start, VerificationStatus.FAIL, f"requested_distributed_runtime={requested_runtime}")
            if distributed_backend != "colossalai":
                return self._finish(start, VerificationStatus.FAIL, f"distributed_runtime_backend={distributed_backend}")
            if str(probe.get("backend") or "") != "colossalai":
                return self._finish(start, VerificationStatus.FAIL, f"probe_backend={probe.get('backend')}")
            if str(probe.get("plugin") or "") != "HybridParallelPlugin":
                return self._finish(start, VerificationStatus.FAIL, f"probe_plugin={probe.get('plugin')}")
            if not all([has_booster_import, has_plugin_import, has_topology_compute, has_dist_init]):
                return self._finish(start, VerificationStatus.FAIL, "colossalai wrapper surface incomplete")

            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.FAIL, str(exc))
