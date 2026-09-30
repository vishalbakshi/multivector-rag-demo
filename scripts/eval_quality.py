"""Measure how much retrieval quality binary document vectors cost, on BEIR SciFact.

Encodes SciFact with pylate-rs (the WebAssembly build, run by the project's portable Node),
then scores all 300 test queries with exact MaxSim against every document, under several
query x document precision pairings, and reports NDCG@10. Each pairing is scored twice:
with every document token, and without punctuation tokens (PyLate's default for documents).

Quantization is simulated in float32: this measures ranking quality, not speed.

Usage: .venv/bin/python scripts/eval_quality.py
"""

import csv
import io
import json
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

from common import CACHE, ROOT, best_matches, keep_tokens, load_vectors, punctuation_mask, run_encode

SCIFACT = CACHE / "datasets" / "scifact"
EMB = CACHE / "embeddings"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
PUBLISHED_NDCG10 = 74.77  # answerai-colbert-small-v1 model card


def prepare_scifact():
    """Download SciFact and write the {"id", "text"} JSONL files encode.mjs reads."""
    if not (SCIFACT / "corpus.jsonl").exists():
        with urllib.request.urlopen(SCIFACT_URL) as res:
            zipfile.ZipFile(io.BytesIO(res.read())).extractall(SCIFACT.parent)
    qrels = {}
    for row in csv.DictReader(open(SCIFACT / "qrels" / "test.tsv"), delimiter="\t"):
        if int(row["score"]) > 0:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    if not (SCIFACT / "docs.jsonl").exists():
        with open(SCIFACT / "corpus.jsonl") as f, open(SCIFACT / "docs.jsonl", "w") as out:
            for line in f:
                d = json.loads(line)
                out.write(json.dumps({"id": d["_id"], "text": f"{d['title']} {d['text']}".strip()}) + "\n")
    if not (SCIFACT / "test-queries.jsonl").exists():
        with open(SCIFACT / "queries.jsonl") as f, open(SCIFACT / "test-queries.jsonl", "w") as out:
            for line in f:
                q = json.loads(line)
                if q["_id"] in qrels:
                    out.write(json.dumps({"id": q["_id"], "text": q["text"]}) + "\n")
    return qrels


def encoded(jsonl, prefix, is_query):
    """Load token vectors for a JSONL file, encoding it first if needed."""
    if not Path(f"{prefix}.json").exists():
        run_encode(jsonl, prefix, is_query=is_query)
    meta, vecs, offsets = load_vectors(prefix)
    return meta["ids"], vecs, offsets


def as_fp32(x):
    return x


def as_int8(x):
    """Round each vector to int8 with its own scale, then map back to float to score it."""
    scale = np.abs(x).max(axis=1, keepdims=True) / 127
    return (np.clip(np.round(x / scale), -127, 127) * scale).astype(np.float32)


def as_binary(x):
    """Keep only the sign of each number, as +1 or -1."""
    return np.where(x > 0, 1, -1).astype(np.float32)


PAIRINGS = [  # (label, query precision, document precision)
    ("fp32 x fp32", as_fp32, as_fp32),
    ("int8 x int8", as_int8, as_int8),
    ("fp32 x binary", as_fp32, as_binary),
    ("int8 x binary", as_int8, as_binary),
    ("binary x binary", as_binary, as_binary),
]
BYTES_PER_NUMBER = {as_fp32: 4, as_int8: 1, as_binary: 1 / 8}  # raw payload only


def rankings(q_vecs, q_offsets, d_vecs, d_offsets, k=100):
    """Exact MaxSim of every query against every document; returns top-k document indices."""
    top = []
    for i in range(len(q_offsets) - 1):
        scores = best_matches(q_vecs[q_offsets[i] : q_offsets[i + 1]], d_vecs, d_offsets).sum(axis=0)
        best = np.argpartition(-scores, k)[:k]
        top.append(best[np.lexsort((best, -scores[best]))])  # score descending, ties by document order
    return top


def ndcg_at_10(ranked_ids, rels):
    dcg = sum(rels.get(d, 0) / np.log2(r + 2) for r, d in enumerate(ranked_ids[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    return dcg / sum(g / np.log2(r + 2) for r, g in enumerate(ideal))


def main():
    qrels = prepare_scifact()
    doc_ids, d_vecs, d_offsets = encoded(SCIFACT / "docs.jsonl", EMB / "scifact-docs", is_query=False)
    q_ids, q_vecs, q_offsets = encoded(SCIFACT / "test-queries.jsonl", EMB / "scifact-queries", is_query=True)
    texts = [json.loads(line)["text"] for line in open(SCIFACT / "docs.jsonl")]
    keep = punctuation_mask(texts, d_offsets)
    variants = {"all": (d_vecs, d_offsets), "no punctuation": keep_tokens(d_vecs, d_offsets, keep)}
    dim = d_vecs.shape[1]
    print(f"{len(doc_ids):,} documents, {len(d_vecs):,} token vectors of size {dim} "
          f"({(~keep).sum():,} are punctuation); {len(q_ids)} queries\n")

    rows, baseline = [], None
    for label, q_fn, d_fn in PAIRINGS:
        for tokens, (vecs, offsets) in variants.items():
            start = time.time()
            top = rankings(q_fn(q_vecs), q_offsets, d_fn(vecs), offsets)
            ranked = [[doc_ids[j] for j in t] for t in top]
            ndcg = 100 * np.mean([ndcg_at_10(r, qrels[q]) for q, r in zip(q_ids, ranked)])
            recall = 100 * np.mean([len(set(r) & set(qrels[q])) / len(qrels[q]) for q, r in zip(q_ids, ranked)])
            top10 = [set(r[:10]) for r in ranked]
            baseline = baseline or (ndcg, top10)  # fp32 x fp32 with every token
            overlap = 100 * np.mean([len(a & b) / 10 for a, b in zip(top10, baseline[1])])
            size_mb = len(vecs) * dim * BYTES_PER_NUMBER[d_fn] / 1e6
            rows.append((label, tokens, ndcg, recall, overlap, size_mb))
            print(f"{label:<16} {tokens:<15} NDCG@10 {ndcg:6.2f}  Recall@100 {recall:6.2f}  "
                  f"top-10 overlap {overlap:5.1f}%  ({time.time() - start:.0f} s)")

    table = [
        "| Query x document | Document tokens | NDCG@10 | Change | Recall@100 | Same top 10 as fp32 | Document vectors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, (label, tokens, ndcg, recall, overlap, size_mb) in enumerate(rows):
        change = "–" if i == 0 else f"{ndcg - baseline[0]:+.2f}"
        table.append(f"| {label} | {tokens} | {ndcg:.2f} | {change} | {recall:.2f} | {overlap:.1f}% | {size_mb:.1f} MB |")

    model = json.loads(Path(f"{EMB / 'scifact-docs'}.json").read_text())["model"]
    report = f"""# Binary document vectors on SciFact

Run on {date.today().isoformat()} with `scripts/eval_quality.py`.

- Model: `{model['repo']}` at revision `{model['revision'][:7]}`, encoded with pylate-rs (WebAssembly build, in Node)
- Corpus: BEIR SciFact, {len(doc_ids):,} documents, {len(d_vecs):,} token vectors of {dim} numbers
  ({(~keep).sum():,} of them punctuation); {len(q_ids)} test queries
- Scoring: exact MaxSim against every document. Quantization simulated in float32. int8 uses one scale per vector.
- "No punctuation" leaves out document tokens that are a single ASCII punctuation mark, PyLate's default.
  pylate-rs keeps them. Queries are unchanged.
- Published NDCG@10 for this model: {PUBLISHED_NDCG10}

{chr(10).join(table)}

"Change" and "Same top 10 as fp32" compare against fp32 x fp32 with every token (the first row).
"Same top 10" is the average share of that top 10 each row also returns.
Document vector sizes count raw vector payload only.
"""
    out = ROOT / "results" / "scifact.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"\n{chr(10).join(table)}\n\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
