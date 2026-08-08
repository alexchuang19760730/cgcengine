"""sitecustomize.py — auto-imported by Python at startup."""
import os

if os.environ.get("ORTHO_KDA_ENABLED", "0") == "1":
    try:
        if os.environ.get("DEBUG_PATCH", "0") == "1":
            import debug_patch_loader
        else:
            import orthokda_patch_loader
    except Exception as e:
        print(f"[sitecustomize] patch load failed: {e}", flush=True)
