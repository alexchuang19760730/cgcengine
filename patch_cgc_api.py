import re

with open('cgc_api_server_remote.py', 'r') as f:
    content = f.read()

new_network_call = """        async def _do_network_call():
            import httpx
            
            # 1. Prepare HTTP payload
            if isinstance(payload, dict):
                req_json = payload
                if "model" not in req_json:
                    req_json["model"] = "deepseek-v4-flash:latest"
            elif isinstance(payload, list):
                req_json = {"messages": payload, "model": "deepseek-v4-flash:latest"}
            else:
                req_json = {"prompt": payload, "model": "deepseek-v4-flash:latest"}
                
            url = f"http://{CLOUD_HOST}:{CLOUD_PORT}/v1/chat/completions"
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=req_json)
                response.raise_for_status()
                resp_data = response.json()
                
            cloud_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not cloud_text and "choices" in resp_data and "text" in resp_data["choices"][0]:
                cloud_text = resp_data["choices"][0]["text"]
                
            self._last_cloud_meta = {
                "mode": "http",
                "payload_size": len(response.content),
                "num_chunks": 1,
                "chunk_size": len(response.content),
                "text_len": len(cloud_text),
                "text_preview": _safe_debug_value(cloud_text, limit=160),
                "text_is_ok": cloud_text.strip() == "OK",
            }
            return cloud_text
"""

pattern = re.compile(r'        async def _do_network_call\(\):.*?return cloud_text\n', re.DOTALL)
content = pattern.sub(new_network_call, content)

with open('cgc_api_server_remote_patched.py', 'w') as f:
    f.write(content)
