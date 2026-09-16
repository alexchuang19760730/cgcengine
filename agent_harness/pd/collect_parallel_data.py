import os, sys
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""鎺￠泦 MoT-h 瑷撶反骞宠灏? {text, h_gemma4, h_qwen36}.

鍚屼竴鏂囨湰鍒嗗垾鐢?Gemma4 鍜?Qwen3.6 鍋?prefill,
鎺￠泦鍚勮嚜鏈堡 hidden state, 绲勬垚骞宠灏?

闇€姹?
  - Mac A (Gemma4): TurboFieldfare + /v1/cgc/emit
  - Mac B (Qwen3.6): TurboFieldfare + /v1/cgc/emit (闇€椤嶅瀵︾従)
  - 鍏╁彴姗熷櫒鍦ㄥ悓涓€缍插煙

鐢ㄦ硶:
  py collect_parallel_data.py \
    --emit-host-a 192.168.1.10 --emit-port-a 8080 \
    --emit-host-b 192.168.1.20 --emit-port-b 8081 \
    --input corpus.jsonl \
    --output parallel_pairs/ \
    --num-samples 1000

杓稿嚭鏍煎紡 (姣忚涓€鍊?JSON):
  {"text": "...", "h_gemma4_b64": "...", "h_qwen36_b64": "...",
   "seq_len": 128, "gemma4_hidden_dim": 2816, "qwen36_hidden_dim": 2048}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# 鍕曟厠鍔犲叆 cgc-engine/pd 璺緫
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from protocol import EmitRequest, EmitResponse, encode_hidden_state, decode_hidden_state  # noqa: E402
from turbofieldfare_adapter import TurboFieldfareClient  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 鏁告摎鎺￠泦鍣?
# ---------------------------------------------------------------------------
class ParallelDataCollector:
    """鎺￠泦 Gemma4 鈫?Qwen3.6 hidden state 骞宠灏?"""

    def __init__(
        self,
        client_a: TurboFieldfareClient,  # Mac A (Gemma4)
        client_b: TurboFieldfareClient,  # Mac B (Qwen3.6)
    ):
        self.client_a = client_a
        self.client_b = client_b

    async def collect_one(self, text: str) -> dict | None:
        """鎺￠泦鍠鏂囨湰鐨勫钩琛屽皪.

        Args:
            text: 杓稿叆鏂囨湰

        Returns:
            {"text": ..., "h_gemma4_b64": ..., "h_qwen36_b64": ..., ...}
            鎴?None (鎺￠泦澶辨晽)
        """
        request_id = f"collect-{int(time.time() * 1000) % 1000000}"

        try:
            # 涓﹁ emit (Gemma4 + Qwen3.6)
            emit_a, emit_b = await asyncio.gather(
                self.client_a.emit(EmitRequest(prompt=text, request_id=f"{request_id}-a")),
                self.client_b.emit(EmitRequest(prompt=text, request_id=f"{request_id}-b")),
            )

            # 椹楄瓑 seq_len 涓€鑷?(涓嶅悓 tokenizer 鍙兘涓嶅悓)
            seq_a = emit_a.packet.seq_len
            seq_b = emit_b.packet.seq_len
            if seq_a != seq_b:
                logger.warning(
                    "seq_len mismatch: gemma4=%d qwen36=%d (涓嶅悓 tokenizer), "
                    "灏嶉綂鍒拌純鐭暦搴?, seq_a, seq_b,
                )
                # 灏嶉綂鍒拌純鐭暦搴?(MVP: 鎴柗)
                min_len = min(seq_a, seq_b)
                h_a = emit_a.packet.to_tensor()[:min_len]
                h_b = emit_b.packet.to_tensor()[:min_len]
            else:
                h_a = emit_a.packet.to_tensor()
                h_b = emit_b.packet.to_tensor()

            return {
                "text": text,
                "h_gemma4_b64": encode_hidden_state(h_a),
                "h_qwen36_b64": encode_hidden_state(h_b),
                "seq_len": h_a.shape[0],
                "gemma4_hidden_dim": h_a.shape[1],
                "qwen36_hidden_dim": h_b.shape[1],
                "gemma4_prefill_ms": emit_a.prefill_latency_ms,
                "qwen36_prefill_ms": emit_b.prefill_latency_ms,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error("collect failed for request %s: %s", request_id, e)
            return None

    async def collect_batch(
        self,
        texts: list[str],
        output_path: str,
        concurrency: int = 4,
    ) -> int:
        """鎵归噺鎺￠泦, 瀵叆 JSONL 鏂囦欢.

        Args:
            texts: 鏂囨湰鍒楄〃
            output_path: 杓稿嚭 JSONL 璺緫
            concurrency: 涓︾櫦鏁?

        Returns:
            鎴愬姛鎺￠泦鐨勬暩閲?
        """
        semaphore = asyncio.Semaphore(concurrency)
        success_count = 0
        total = len(texts)

        async def collect_with_sem(idx: int, text: str):
            nonlocal success_count
            async with semaphore:
                result = await self.collect_one(text)
                if result:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success_count += 1
                    if success_count % 10 == 0:
                        logger.info("閫插害: %d/%d (%.1f%%)", success_count, total, 100 * success_count / total)
                return result

        tasks = [collect_with_sem(i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)

        logger.info("鎺￠泦瀹屾垚: %d/%d 鎴愬姛", success_count, total)
        return success_count


# ---------------------------------------------------------------------------
# 瑾炴枡璁€鍙?
# ---------------------------------------------------------------------------
def load_corpus(input_path: str, max_samples: int = -1) -> list[str]:
    """璁€鍙栬獮鏂欐枃浠?

    鏀寔鏍煎紡:
      - .jsonl: 姣忚涓€鍊?JSON, 鍙?"text" 瀛楁
      - .txt: 姣忚涓€姊濇枃鏈?
      - .json: list of {"text": ...}
    """
    texts = []
    ext = Path(input_path).suffix

    if ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("prompt", ""))
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    texts.append(line)  # 鐣剁磾鏂囨湰铏曠悊
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(item.get("text", item.get("prompt", "")))
    else:  # .txt
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if max_samples > 0:
        texts = texts[:max_samples]

    logger.info("杓夊叆瑾炴枡: %d 姊?from %s", len(texts), input_path)
    return texts


# ---------------------------------------------------------------------------
# 涓诲嚱鏁?
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="鎺￠泦 MoT-h 瑷撶反骞宠灏?)
    parser.add_argument("--emit-host-a", default=os.getenv("TF_EMIT_HOST", "127.0.0.1"),
                        help="Mac A (Gemma4) 鍦板潃")
    parser.add_argument("--emit-port-a", type=int, default=int(os.getenv("TF_EMIT_PORT", "8080")))
    parser.add_argument("--emit-host-b", default=os.getenv("TF_EMIT_HOST_B", "127.0.0.1"),
                        help="Mac B (Qwen3.6) 鍦板潃")
    parser.add_argument("--emit-port-b", type=int, default=int(os.getenv("TF_EMIT_PORT_B", "8081")))
    parser.add_argument("--input", required=True, help="瑾炴枡鏂囦欢璺緫")
    parser.add_argument("--output", required=True, help="杓稿嚭 JSONL 璺緫")
    parser.add_argument("--num-samples", type=int, default=-1, help="鎺￠泦鏁搁噺 (-1 = 鍏ㄩ儴)")
    parser.add_argument("--concurrency", type=int, default=4, help="涓︾櫦鏁?)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 杓夊叆瑾炴枡
    texts = load_corpus(args.input, args.num_samples)
    if not texts:
        logger.error("鐒¤獮鏂欏彲鎺￠泦")
        return

    # 鍓靛缓 clients
    client_a = TurboFieldfareClient(f"http://{args.emit_host_a}:{args.emit_port_a}")
    client_b = TurboFieldfareClient(f"http://{args.emit_host_b}:{args.emit_port_b}")

    # 鍋ュ悍妾㈡煡
    a_ok = await client_a.health()
    b_ok = await client_b.health()
    logger.info("Mac A (Gemma4): %s", "ok" if a_ok else "FAIL")
    logger.info("Mac B (Qwen3.6): %s", "ok" if b_ok else "FAIL")
    if not (a_ok and b_ok):
        logger.error("TurboFieldfare 鏈嶅嫏涓嶅彲鐢? 璜嬫鏌?Mac 绔?/v1/cgc/emit 鏄惁宸插鐝?)
        await client_a.close()
        await client_b.close()
        return

    # 鎺￠泦
    collector = ParallelDataCollector(client_a, client_b)
    count = await collector.collect_batch(texts, args.output, args.concurrency)

    logger.info("瀹屾垚: 鎺￠泦 %d 姊濆钩琛屽皪 鈫?%s", count, args.output)

    await client_a.close()
    await client_b.close()


if __name__ == "__main__":
    asyncio.run(main())
