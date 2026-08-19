#!/usr/bin/env python3

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

SWE_VERIFIED_PASS_THRESHOLD = 0.99
SWE_VERIFIED_TOTAL = 500

class SWEOfficialEvaluator:
    def __init__(self, gate_dir: Optional[Path] = None):
        if gate_dir is None:
            self.gate_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / \
                "docs/technical_whitepapers/CGC_Gate_6.0_fusionroute_complete"
        else:
            self.gate_dir = gate_dir
        
        self.eval_cache = {}
    
    def load_formal_summary(self) -> Optional[Dict[str, Any]]:
        summary_path = self.gate_dir / "swe_verified_formal_summary.json"
        if not summary_path.exists():
            return None
        
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_runbatch_result(self, runbatch_id: str = "latest") -> Optional[Dict[str, Any]]:
        if runbatch_id == "latest":
            runbatch_files = list(self.gate_dir.glob("runbatch_*.json"))
            if not runbatch_files:
                return None
            runbatch_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            runbatch_path = runbatch_files[0]
        else:
            runbatch_path = self.gate_dir / f"runbatch_{runbatch_id}.json"
        
        if not runbatch_path.exists():
            return None
        
        with open(runbatch_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_official_eval(self) -> Optional[Dict[str, Any]]:
        eval_path = self.gate_dir / "swe_verified_official_eval.json"
        if not eval_path.exists():
            return None
        
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def validate_evidence_chain(self) -> Tuple[bool, str, Dict[str, Any]]:
        evidence = {
            "validations": [],
            "metrics": {},
            "claims": [],
            "warnings": [],
            "errors": []
        }
        
        formal_summary = self.load_formal_summary()
        if not formal_summary:
            evidence["errors"].append("MISSING_EVIDENCE: swe_verified_formal_summary.json not found")
            return False, "Missing formal summary", evidence
        
        evidence["validations"].append("FORMAL_SUMMARY_LOADED")
        
        official_eval = formal_summary.get("official_evaluation", {})
        runbatch = formal_summary.get("runbatch_summary", {})
        evidence_refs = formal_summary.get("evidence_references", [])
        
        swe_status = formal_summary.get("swe_verified_status", "UNKNOWN")
        formal_readiness = formal_summary.get("formal_readiness", "UNKNOWN")
        
        total_instances = official_eval.get("total_instances", 0)
        completed_instances = official_eval.get("completed_instances", 0)
        resolution_rate = official_eval.get("resolution_rate", 0.0)
        success_rate = official_eval.get("success_rate", 0.0)
        quality_drop = official_eval.get("quality_drop", 0.0)
        speedup_ratio = official_eval.get("speedup_ratio", 0.0)
        
        run_count = runbatch.get("run_count", 0)
        completed_runs = runbatch.get("completed_runs", 0)
        official_eval_status = runbatch.get("official_eval_status", "UNKNOWN")
        claimable = runbatch.get("claimable", False)
        
        evidence["metrics"] = {
            "total_instances": total_instances,
            "completed_instances": completed_instances,
            "resolution_rate": resolution_rate,
            "success_rate": success_rate,
            "quality_drop": quality_drop,
            "speedup_ratio": speedup_ratio,
            "run_count": run_count,
            "completed_runs": completed_runs,
            "swe_status": swe_status,
            "formal_readiness": formal_readiness,
            "official_eval_status": official_eval_status,
            "claimable": claimable
        }
        
        if total_instances != SWE_VERIFIED_TOTAL:
            evidence["warnings"].append(f"INSTANCE_MISMATCH: Expected {SWE_VERIFIED_TOTAL} instances, got {total_instances}")
        
        if completed_instances == total_instances:
            evidence["validations"].append("ALL_INSTANCES_COMPLETED")
        else:
            evidence["errors"].append(f"INCOMPLETE_RUN: {completed_instances}/{total_instances} completed")
        
        if resolution_rate >= SWE_VERIFIED_PASS_THRESHOLD:
            evidence["validations"].append(f"PASS_RATE_MET: {resolution_rate*100:.1f}% >= {SWE_VERIFIED_PASS_THRESHOLD*100}%")
        else:
            evidence["errors"].append(f"PASS_RATE_FAILED: {resolution_rate*100:.1f}% < {SWE_VERIFIED_PASS_THRESHOLD*100}%")
        
        if swe_status == "VERIFIED":
            evidence["validations"].append("SWE_STATUS_VERIFIED")
        else:
            evidence["errors"].append(f"SWE_STATUS_NOT_VERIFIED: {swe_status}")
        
        if formal_readiness == "PRODUCTION_READY":
            evidence["validations"].append("FORMAL_READINESS_PRODUCTION")
        else:
            evidence["errors"].append(f"FORMAL_READINESS_NOT_PRODUCTION: {formal_readiness}")
        
        if official_eval_status == "PASSED":
            evidence["validations"].append("OFFICIAL_EVAL_PASSED")
        else:
            evidence["errors"].append(f"OFFICIAL_EVAL_NOT_PASSED: {official_eval_status}")
        
        if claimable:
            evidence["validations"].append("CLAIMABLE_TRUE")
        else:
            evidence["errors"].append("CLAIMABLE_FALSE")
        
        evidence["claims"].append({
            "id": "swe_verified_500",
            "name": "SWE Verified 500",
            "claimed": claimable,
            "pass_rate": resolution_rate,
            "evidence_count": len(evidence_refs)
        })
        
        if quality_drop > 0.01:
            evidence["warnings"].append(f"QUALITY_DROP_HIGH: {quality_drop*100:.2f}%")
        
        if speedup_ratio < 0.3:
            evidence["warnings"].append(f"SPEEDUP_LOW: {speedup_ratio*100:.1f}%")
        
        is_valid = len(evidence["errors"]) == 0
        
        if is_valid:
            message = "SWE Verified 500 - All official evaluation criteria met"
        else:
            message = f"SWE Verified 500 - Validation failed with {len(evidence['errors'])} errors"
        
        return is_valid, message, evidence
    
    def generate_claim_report(self) -> str:
        valid, message, evidence = self.validate_evidence_chain()
        
        report = [
            "=" * 80,
            "      SWE Verified 500 - Official Evaluation Claim Report",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Claim Status: {'APPROVED' if valid else 'REJECTED'}",
            "=" * 80,
            "",
        ]
        
        metrics = evidence["metrics"]
        report.extend([
            "📊 OFFICIAL EVALUATION METRICS",
            "-" * 40,
            f"  Total Instances:          {metrics['total_instances']}",
            f"  Completed Instances:      {metrics['completed_instances']}",
            f"  Resolution Rate:          {metrics['resolution_rate']*100:.2f}%",
            f"  Success Rate:             {metrics['success_rate']*100:.2f}%",
            f"  Quality Drop:             {metrics['quality_drop']*100:.3f}%",
            f"  Speedup Ratio:            {metrics['speedup_ratio']*100:.1f}%",
            f"  Run Count:                {metrics['run_count']}",
            f"  Completed Runs:           {metrics['completed_runs']}",
            "",
        ])
        
        report.extend([
            "✅ VALIDATIONS PASSED",
            "-" * 40,
        ])
        for v in evidence["validations"]:
            report.append(f"  ✓ {v}")
        
        if evidence["warnings"]:
            report.extend([
                "",
                "⚠️ WARNINGS",
                "-" * 40,
            ])
            for w in evidence["warnings"]:
                report.append(f"  ⚠ {w}")
        
        if evidence["errors"]:
            report.extend([
                "",
                "❌ ERRORS",
                "-" * 40,
            ])
            for e in evidence["errors"]:
                report.append(f"  ✗ {e}")
        
        report.extend([
            "",
            "📝 CLAIM DETAILS",
            "-" * 40,
        ])
        for claim in evidence["claims"]:
            report.append(f"  Claim ID:    {claim['id']}")
            report.append(f"  Name:        {claim['name']}")
            report.append(f"  Claimable:   {claim['claimed']}")
            report.append(f"  Pass Rate:   {claim['pass_rate']*100:.2f}%")
            report.append(f"  Evidence:    {claim['evidence_count']} references")
        
        report.extend([
            "",
            "=" * 80,
            f"Final Verdict: {'✅ SWE Verified 500 CLAIM APPROVED' if valid else '❌ SWE Verified 500 CLAIM REJECTED'}",
            "=" * 80,
        ])
        
        return "\n".join(report)
    
    def execute(self) -> Tuple[bool, str]:
        valid, message, evidence = self.validate_evidence_chain()
        
        report_file = Path(__file__).parent / f"swe_verified_500_official_claim_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "claim_id": "swe_verified_500",
            "valid": valid,
            "message": message,
            "evidence": evidence,
            "thresholds": {
                "pass_threshold": SWE_VERIFIED_PASS_THRESHOLD,
                "total_tasks": SWE_VERIFIED_TOTAL
            },
            "hash": hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return valid, str(report_file)

def main():
    evaluator = SWEOfficialEvaluator()
    valid, report_path = evaluator.execute()
    
    print(evaluator.generate_claim_report())
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    sys.exit(0 if valid else 1)

if __name__ == "__main__":
    main()