#!/usr/bin/env python3
"""採集 MoT-h 訓練平行對: {text, h_gemma4, h_qwen36}.

同一文本分別用 Gemma4 和 Qwen3.6 做 prefill,
採集各自末層 hidden state, 組成平行對.

需求:
  - Mac A (Gemma4): TurboFieldfare + /v1/cgc/emit
  - Mac B (Qwen3.6): TurboFieldfare + /v1/cgc/emit (需額外實現)
  - 兩台機器在同一網域

用法:
  py collect_parallel_data.py \
    --emit-host-a 192.168.1.10 --emit-port-a 8080 \
    --emit-host-b 192.168.1.20 --emit-port-b 8081 \
    --input corpus.jsonl \
    --output parallel_pairs/ \
    --num-samples 1000

輸出格式 (每行一個 JSON):
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

# 動態加入 cgc-engine/pd 路徑
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from protocol import EmitRequest, EmitResponse, encode_hidden_state, decode_hidden_state  # noqa: E402
from turbofieldfare_adapter import TurboFieldfareClient  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 數據採集器
# ---------------------------------------------------------------------------
class ParallelDataCollector:
    """採集 Gemma4 ↔ Qwen3.6 hidden state 平行對."""

    def __init__(
        self,
        client_a: TurboFieldfareClient,  # Mac A (Gemma4)
        client_b: TurboFieldfareClient,  # Mac B (Qwen3.6)
    ):
        self.client_a = client_a
        self.client_b = client_b

    async def collect_one(self, text: str) -> dict | None:
        """採集單條文本的平行對.

        Args:
            text: 輸入文本

        Returns:
            {"text": ..., "h_gemma4_b64": ..., "h_qwen36_b64": ..., ...}
            或 None (採集失敗)
        """
        request_id = f"collect-{int(time.time() * 1000) % 1000000}"

        try:
            # 並行 emit (Gemma4 + Qwen3.6)
            emit_a, emit_b = await asyncio.gather(
                self.client_a.emit(EmitRequest(prompt=text, request_id=f"{request_id}-a")),
                self.client_b.emit(EmitRequest(prompt=text, request_id=f"{request_id}-b")),
            )

            # 驗證 seq_len 一致 (不同 tokenizer 可能不同)
            seq_a = emit_a.packet.seq_len
            seq_b = emit_b.packet.seq_len
            if seq_a != seq_b:
                logger.warning(
                    "seq_len mismatch: gemma4=%d qwen36=%d (不同 tokenizer), "
                    "對齊到較短長度", seq_a, seq_b,
                )
                # 對齊到較短長度 (MVP: 截斷)
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
        """批量採集, 寫入 JSONL 文件.

        Args:
            texts: 文本列表
            output_path: 輸出 JSONL 路徑
            concurrency: 並發數

        Returns:
            成功採集的數量
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
                        logger.info("進度: %d/%d (%.1f%%)", success_count, total, 100 * success_count / total)
                return result

        tasks = [collect_with_sem(i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)

        logger.info("採集完成: %d/%d 成功", success_count, total)
        return success_count


# ---------------------------------------------------------------------------
# 語料讀取
# ---------------------------------------------------------------------------
def load_corpus(input_path: str, max_samples: int = -1) -> list[str]:
    """讀取語料文件.

    支持格式:
      - .jsonl: 每行一個 JSON, 取 "text" 字段
      - .txt: 每行一條文本
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
                    texts.append(line)  # 當純文本處理
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

    logger.info("載入語料: %d 條 from %s", len(texts), input_path)
    return texts


# ---------------------------------------------------------------------------
# 主函數
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="採集 MoT-h 訓練平行對")
    parser.add_argument("--emit-host-a", default=os.getenv("TF_EMIT_HOST", "127.0.0.1"),
                        help="Mac A (Gemma4) 地址")
    parser.add_argument("--emit-port-a", type=int, default=int(os.getenv("TF_EMIT_PORT", "8080")))
    parser.add_argument("--emit-host-b", default=os.getenv("TF_EMIT_HOST_B", "127.0.0.1"),
                        help="Mac B (Qwen3.6) 地址")
    parser.add_argument("--emit-port-b", type=int, default=int(os.getenv("TF_EMIT_PORT_B", "8081")))
    parser.add_argument("--input", required=True, help="語料文件路徑")
    parser.add_argument("--output", required=True, help="輸出 JSONL 路徑")
    parser.add_argument("--num-samples", type=int, default=-1, help="採集數量 (-1 = 全部)")
    parser.add_argument("--concurrency", type=int, default=4, help="並發數")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 載入語料
    texts = load_corpus(args.input, args.num_samples)
    if not texts:
        logger.error("無語料可採集")
        return

    # 創建 clients
    client_a = TurboFieldfareClient(f"http://{args.emit_host_a}:{args.emit_port_a}")
    client_b = TurboFieldfareClient(f"http://{args.emit_host_b}:{args.emit_port_b}")

    # 健康檢查
    a_ok = await client_a.health()
    b_ok = await client_b.health()
    logger.info("Mac A (Gemma4): %s", "ok" if a_ok else "FAIL")
    logger.info("Mac B (Qwen3.6): %s", "ok" if b_ok else "FAIL")
    if not (a_ok and b_ok):
        logger.error("TurboFieldfare 服務不可用, 請檢查 Mac 端 /v1/cgc/emit 是否已實現")
        await client_a.close()
        await client_b.close()
        return

    # 採集
    collector = ParallelDataCollector(client_a, client_b)
    count = await collector.collect_batch(texts, args.output, args.concurrency)

    logger.info("完成: 採集 %d 條平行對 → %s", count, args.output)

    await client_a.close()
    await client_b.close()


if __name__ == "__main__":
    asyncio.run(main())
