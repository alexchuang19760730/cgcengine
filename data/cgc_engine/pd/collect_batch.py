#!/usr/bin/env python3
"""批量採集 MoT-h 訓練對: 資料夾 → 兩台 Mac 並行 emit → train.pt.

用法:
  py collect_batch.py \\
    --gemma4-url http://192.168.101.X:8080 \\
    --qwen36-url http://192.168.101.Y:8080 \\
    --input corpus/ \\
    --output train.pt \\
    --concurrency 4

輸入資料夾結構:
  corpus/
    ├── 001.txt
    ├── 002.json
    ├── 003.py
    └── ...

支持的副檔名: .txt .json .py .js .ts .md .jsonl
- .txt/.py/.js/.ts/.md: 整個檔案當作一條文本
- .json: {"text": "..."} 或 [{"text": "..."}, ...]
- .jsonl: 每行一個 {"text": "..."}

輸出 (train.pt):
  {
    "pairs": [
      {"h_src": Tensor[seq, 2816], "h_tgt": Tensor[seq, 2048], "text": str, "seq_len": int},
      ...
    ],
    "meta": {
      "src_dim": 2816, "tgt_dim": 2048,
      "num_pairs": N, "total_tokens": T,
      "gemma4_url": ..., "qwen36_url": ...,
      "collected_at": "2026-08-12T..."
    }
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import torch

# 動態加入路徑
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from protocol import decode_hidden_state  # noqa: E402

logger = logging.getLogger(__name__)

# 支持的副檔名
SUPPORTED_EXTS = {".txt", ".json", ".py", ".js", ".ts", ".md", ".jsonl", ".swift", ".java", ".cpp"}


# ---------------------------------------------------------------------------
# 語料讀取
# ---------------------------------------------------------------------------
def load_file_text(path: Path) -> list[str]:
    """從單個檔案讀取文本, 返回一條或多條文本.

    - .jsonl: 每行 {"text": "..."} → 多條
    - .json:  list or dict → 一條或多條
    - 其他: 整個檔案當作一條
    """
    ext = path.suffix.lower()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        logger.warning("讀取失敗 %s: %s", path, e)
        return []

    if not content.strip():
        return []

    if ext == ".jsonl":
        texts = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("prompt", "")) if isinstance(obj, dict) else str(obj)
                if text and len(text.strip()) > 20:
                    texts.append(text)
            except json.JSONDecodeError:
                if len(line) > 20:
                    texts.append(line)
        return texts

    if ext == ".json":
        try:
            data = json.loads(content)
            if isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, str) and len(item) > 20:
                        texts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("prompt", ""))
                        if text and len(text) > 20:
                            texts.append(text)
                return texts
            elif isinstance(data, dict):
                text = data.get("text", data.get("prompt", ""))
                return [text] if text and len(text) > 20 else []
        except json.JSONDecodeError:
            # 當純文本處理
            return [content] if len(content) > 20 else []

    # .txt .py .js .ts .md .swift .java .cpp
    return [content] if len(content) > 20 else []


def load_corpus_folder(folder: str, max_files: int = -1) -> list[tuple[str, str]]:
    """從資料夾載入所有語料.

    Returns:
        [(filename, text), ...]
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"資料夾不存在: {folder}")

    pairs = []
    files = sorted([p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

    if max_files > 0:
        files = files[:max_files]

    for fp in files:
        texts = load_file_text(fp)
        for i, text in enumerate(texts):
            name = f"{fp.stem}_{i}" if len(texts) > 1 else fp.stem
            pairs.append((name, text))

    logger.info("載入語料: %d 條 from %s (%d 個檔案)", len(pairs), folder, len(files))
    return pairs


# ---------------------------------------------------------------------------
# 並行採集
# ---------------------------------------------------------------------------
async def emit_one(session: aiohttp.ClientSession, url: str, prompt: str,
                    request_id: str, max_seq_len: int = 4096,
                    timeout: float = 180.0) -> dict:
    """POST /v1/cgc/emit 到一台 Mac."""
    endpoint = url.rstrip("/") + "/v1/cgc/emit"
    payload = {
        "prompt": prompt,
        "request_id": request_id,
        "max_seq_len": max_seq_len,
    }
    async with session.post(endpoint, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json()


async def collect_pair(session: aiohttp.ClientSession,
                        gemma4_url: str, qwen36_url: str,
                        name: str, text: str,
                        max_seq_len: int = 4096) -> dict | None:
    """採集一條文本的 (h_src, h_tgt) 平行對."""
    request_id = f"batch-{name}-{int(time.time() * 1000) % 1000000}"
    try:
        # 並行 emit 兩台 Mac
        resp_a, resp_b = await asyncio.gather(
            emit_one(session, gemma4_url, text, f"{request_id}-g4", max_seq_len),
            emit_one(session, qwen36_url, text, f"{request_id}-q36", max_seq_len),
        )

        if not resp_a.get("success") or not resp_b.get("success"):
            err = resp_a.get("error", "") or resp_b.get("error", "")
            logger.warning("emit 失敗 %s: %s", name, err)
            return None

        # 解碼 hidden state
        h_src = decode_hidden_state(
            resp_a["hidden_state_b64"], resp_a["seq_len"], resp_a["hidden_dim"])
        h_tgt = decode_hidden_state(
            resp_b["hidden_state_b64"], resp_b["seq_len"], resp_b["hidden_dim"])

        # 對齊 seq_len (tokenizer 可能不同)
        min_len = min(h_src.shape[0], h_tgt.shape[0])
        if h_src.shape[0] != h_tgt.shape[0]:
            logger.info("seq_len 對齊 %s: g4=%d q36=%d → %d",
                         name, h_src.shape[0], h_tgt.shape[0], min_len)
            h_src = h_src[:min_len]
            h_tgt = h_tgt[:min_len]

        return {
            "h_src": h_src,
            "h_tgt": h_tgt,
            "text": text[:200],  # 只保留前 200 字元做 metadata
            "seq_len": min_len,
            "gemma4_prefill_ms": resp_a.get("prefill_ms", 0),
            "qwen36_prefill_ms": resp_b.get("prefill_ms", 0),
        }
    except Exception as e:
        logger.error("collect %s 失敗: %s", name, e)
        return None


async def collect_batch(gemma4_url: str, qwen36_url: str,
                         corpus: list[tuple[str, str]],
                         output_path: str,
                         concurrency: int = 4,
                         max_seq_len: int = 4096) -> int:
    """批量採集, 保存到 .pt 文件."""
    semaphore = asyncio.Semaphore(concurrency)
    pairs = []
    total = len(corpus)
    success_count = 0
    total_tokens = 0
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        async def collect_with_sem(idx: int, name: str, text: str):
            nonlocal success_count, total_tokens
            async with semaphore:
                result = await collect_pair(session, gemma4_url, qwen36_url,
                                             name, text, max_seq_len)
                if result:
                    pairs.append(result)
                    success_count += 1
                    total_tokens += result["seq_len"]
                    if success_count % 5 == 0:
                        elapsed = time.time() - t0
                        rate = success_count / elapsed if elapsed > 0 else 0
                        logger.info("進度: %d/%d (%.1f%%) | %.1f 條/秒 | %d tokens",
                                     success_count, total, 100 * success_count / total,
                                     rate, total_tokens)
                return result

        tasks = [collect_with_sem(i, name, text) for i, (name, text) in enumerate(corpus)]
        await asyncio.gather(*tasks)

    # 保存
    meta = {
        "src_dim": pairs[0]["h_src"].shape[1] if pairs else 0,
        "tgt_dim": pairs[0]["h_tgt"].shape[1] if pairs else 0,
        "num_pairs": len(pairs),
        "total_tokens": total_tokens,
        "gemma4_url": gemma4_url,
        "qwen36_url": qwen36_url,
        "collected_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
    }
    torch.save({"pairs": pairs, "meta": meta}, output_path)

    logger.info("=" * 60)
    logger.info("採集完成")
    logger.info("  成功: %d/%d", success_count, total)
    logger.info("  總 tokens: %d", total_tokens)
    logger.info("  耗時: %.1fs", time.time() - t0)
    logger.info("  保存到: %s", output_path)
    logger.info("  src_dim=%d, tgt_dim=%d", meta["src_dim"], meta["tgt_dim"])
    logger.info("=" * 60)
    return success_count


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="批量採集 MoT-h 訓練對")
    parser.add_argument("--gemma4-url", required=True,
                        help="Mac A (Gemma4) URL, e.g. http://192.168.101.X:8080")
    parser.add_argument("--qwen36-url", required=True,
                        help="Mac B (Qwen3.6) URL, e.g. http://192.168.101.Y:8080")
    parser.add_argument("--input", required=True,
                        help="語料資料夾路徑")
    parser.add_argument("--output", default="train.pt",
                        help="輸出 .pt 文件 (default: train.pt)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="並發數 (default: 4, Mac 端建議 1-2)")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="最大序列長度 (default: 4096)")
    parser.add_argument("--max-files", type=int, default=-1,
                        help="最多讀取 N 個檔案 (default: 全部)")
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="單條文本最大字符數 (default: 8000, 避免過長)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 載入語料
    corpus = load_corpus_folder(args.input, args.max_files)
    if not corpus:
        logger.error("沒有讀到任何語料")
        sys.exit(1)

    # 截斷過長文本
    truncated = 0
    cleaned = []
    for name, text in corpus:
        if len(text) > args.max_chars:
            text = text[:args.max_chars]
            truncated += 1
        cleaned.append((name, text))
    corpus = cleaned
    if truncated:
        logger.info("截斷 %d 條過長文本到 %d 字元", truncated, args.max_chars)

    logger.info("開始採集: %d 條文本, 並發=%d", len(corpus), args.concurrency)
    logger.info("  Gemma4: %s", args.gemma4_url)
    logger.info("  Qwen3.6: %s", args.qwen36_url)

    asyncio.run(collect_batch(
        gemma4_url=args.gemma4_url,
        qwen36_url=args.qwen36_url,
        corpus=corpus,
        output_path=args.output,
        concurrency=args.concurrency,
        max_seq_len=args.max_seq_len,
    ))


if __name__ == "__main__":
    main()
