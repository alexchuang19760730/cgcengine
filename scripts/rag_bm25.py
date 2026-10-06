#!/usr/bin/env python3
"""BM25 retrieval over a local corpus, for putting "1M tokens" in reach on a 16 GB box.

WHY THIS EXISTS
---------------
Storing 1M tokens of KV for this model costs 10.5 KiB/token = 10.8 GB (see
docs/DECODE_SPEED_MILESTONES_2026-10-02.md and the KV derivation in
.workbuddy/memory/2026-10-02.md). That does not fit next to the 8 GiB expert
pool. This tool takes the other route: keep the corpus on disk, retrieve only
what the question needs, and feed a few thousand tokens to the running server.

DESIGN CONSTRAINT: no second resident model.
A dedicated embedding model would have to be co-resident with the 35B and its
8 GiB pool, and this box is already swapping. So retrieval here is lexical
(BM25 via sqlite FTS5, stdlib only, zero model, ~0 RAM at query time).

NOT PART OF THE CERTIFIED PIPELINE. This is a prototype on a new axis; it feeds
no board cell and no gate. Do not quote any t/s measured with it without
running scripts/check/caliber_gate.py and scripts/check/quote_gate.py first.

USAGE
-----
  # build an index (run once per corpus)
  python3 scripts/rag_bm25.py index --db /tmp/rag.db --input ./corpus \
      --chunk-tokens 500 --overlap-tokens 50

  # ask (prints the assembled prompt that would go to the server)
  python3 scripts/rag_bm25.py query --db /tmp/rag.db --q "你的問題" --k 8 \
      --budget-tokens 4000

  # ask and actually POST to the OpenAI-compat server on :8080
  python3 scripts/rag_bm25.py query --db /tmp/rag.db --q "你的問題" \
      --server http://127.0.0.1:8080
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request

# Rough bytes->token ratio. 4.0 is right for English prose; Chinese is closer to
# 1.5-2 chars/token. 3.0 is a deliberately middle-of-the-road default; tune with
# --chars-per-token if your corpus is one-sided.
DEFAULT_CPT = 3.0

CJK = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]")


def tok_estimate(text, cpt=DEFAULT_CPT):
    return max(1, int(len(text) / cpt))


def expand_for_index(text):
    """Make CJK searchable under FTS5's default unicode61 tokenizer.

    unicode61 splits on whitespace/punctuation, so a Chinese sentence with no
    spaces becomes ONE token and never matches. Emitting overlapping character
    bigrams ("機器學習" -> "機器 器學 學習") fixes this with no extra dependency.
    ASCII words pass through untouched.
    """
    out = []
    buf = ""
    for ch in text:
        if CJK.match(ch):
            buf += ch
        else:
            if buf:
                out.append(" ".join(buf[i:i + 2] for i in range(len(buf) - 1)) or buf)
                buf = ""
            out.append(ch)
    if buf:
        out.append(" ".join(buf[i:i + 2] for i in range(len(buf) - 1)) or buf)
    return "".join(out)


def split_chunks(text, chunk_tokens, overlap_tokens, cpt):
    """Greedy paragraph/sentence packing to ~chunk_tokens with overlap."""
    pieces = re.split(r"(?<=[。．.!?！？\n])", text)
    pieces = [p for p in pieces if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if cur and tok_estimate(cur + p, cpt) > chunk_tokens:
            chunks.append(cur.strip())
            tail = cur[-max(1, int(overlap_tokens * cpt)):]
            cur = tail + p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def cmd_index(a):
    con = sqlite3.connect(a.db)
    # body = bigram-expanded text (indexed, matched against)
    # text = verbatim original (UNINDEXED, what we hand back to the model)
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                "body, text UNINDEXED, doc UNINDEXED, ord UNINDEXED)")
    paths = []
    if os.path.isdir(a.input):
        for root, _, files in os.walk(a.input):
            for f in sorted(files):
                if f.endswith((".txt", ".md", ".markdown")):
                    paths.append(os.path.join(root, f))
    else:
        paths = [a.input]
    if not paths:
        sys.exit(f"no .txt/.md under {a.input}")

    n_doc = n_chunk = n_tok = 0
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except OSError as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        chunks = split_chunks(text, a.chunk_tokens, a.overlap_tokens, a.cpt)
        rows = [(expand_for_index(c), c, p, i) for i, c in enumerate(chunks)]
        con.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
        n_doc += 1
        n_chunk += len(chunks)
        n_tok += sum(tok_estimate(c, a.cpt) for c in chunks)
        print(f"  {p}: {len(chunks)} chunks")
    con.commit()
    print(f"indexed {n_doc} docs / {n_chunk} chunks / ~{n_tok} tokens -> {a.db}")
    print(f"approx KV if all of it went into context: "
          f"{n_tok * 10.5 / 1024 / 1024:.2f} GiB")


def cmd_query(a):
    con = sqlite3.connect(a.db)
    # FTS5 MATCH is AND by default, and CJK bigrams almost never all co-occur.
    # OR + bm25() ranking gives "best chunk wins" behaviour instead of 0 hits.
    terms = [t for t in expand_for_index(a.q).split() if t]
    if not terms:
        sys.exit("query produced no searchable terms")
    match = " OR ".join('"%s"' % t.replace('"', "") for t in terms)
    try:
        rows = con.execute(
            "SELECT text, doc, ord, bm25(chunks) AS s FROM chunks "
            "WHERE chunks MATCH ? ORDER BY s LIMIT ?",
            (match, a.candidates)).fetchall()
    except sqlite3.OperationalError as e:
        sys.exit(f"query failed ({e}). Did you run `index` first?")

    picked, used = [], 0
    for text, doc, ord_, _ in rows:
        t = tok_estimate(text, a.cpt)
        if used + t > a.budget_tokens:
            continue
        picked.append({"text": text, "doc": doc, "ord": ord_, "tok": t})
        used += t
        if len(picked) >= a.k:
            break

    print(f"retrieved {len(picked)} chunks / ~{used} tokens "
          f"(budget {a.budget_tokens})", file=sys.stderr)
    est_prefill = used / 250.0
    print(f"est. prefill @250 t/s: {est_prefill:.1f} s "
          f"(1M raw would be {1_000_000/250.0:.0f} s)", file=sys.stderr)

    prompt = "Use only the context below to answer.\n\n"
    for i, c in enumerate(picked):
        prompt += f"[{i+1}] ({os.path.basename(c['doc'])}#{c['ord']})\n{c['text']}\n\n"
    prompt += f"Question: {a.q}\nAnswer:"

    if a.server:
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0}
        req = urllib.request.Request(
            a.server.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            print(json.loads(r.read())["choices"][0]["message"]["content"])
    else:
        print(prompt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index")
    p.add_argument("--db", required=True)
    p.add_argument("--input", required=True, help="file or directory of .txt/.md")
    p.add_argument("--chunk-tokens", type=int, default=500)
    p.add_argument("--overlap-tokens", type=int, default=50)
    p.add_argument("--chars-per-token", dest="cpt", type=float, default=DEFAULT_CPT)
    p.set_defaults(fn=cmd_index)

    q = sub.add_parser("query")
    q.add_argument("--db", required=True)
    q.add_argument("--q", required=True)
    q.add_argument("--k", type=int, default=8, help="chunks to keep")
    q.add_argument("--candidates", type=int, default=60, help="BM25 pool to consider")
    q.add_argument("--budget-tokens", type=int, default=4000)
    q.add_argument("--chars-per-token", dest="cpt", type=float, default=DEFAULT_CPT)
    q.add_argument("--server", default=None, help="e.g. http://127.0.0.1:8080")
    q.add_argument("--model", default="gemma-4-26b-a4b-it")
    q.add_argument("--timeout", type=float, default=600)
    q.set_defaults(fn=cmd_query)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
