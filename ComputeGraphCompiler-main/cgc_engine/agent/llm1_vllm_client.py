import json
import os
import urllib.request
from typing import Any, Dict, List, Optional


def vllm_chat_completions(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: int = 120,
    api_key: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = str(base_url).rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "temperature": 0.0,
    }
    if extra_body:
        payload.update(extra_body)

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    key = api_key if api_key is not None else os.environ.get("LLM1_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": str(e)}

    try:
        out = json.loads(raw)
    except Exception as e:
        return {"ok": False, "error": f"invalid json: {e}", "raw": raw}

    out["ok"] = True
    return out


def extract_chat_content(resp: Dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    return str(content or "")

