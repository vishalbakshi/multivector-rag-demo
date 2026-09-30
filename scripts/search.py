"""Try searches against the packed blog index, scored the way the page will score them.

Float query vectors against the sign bits in data/index.bin, exact MaxSim over every passage.
Shows the best passage from each of the top posts with the words that matched, and where
full-precision (float32) scoring of the cached vectors would have ranked the same post.

Usage: .venv/bin/python scripts/search.py "first query" ["second query" ...] [--top 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import CACHE, DATA, best_matches, keep_tokens, load_vectors, punctuation_mask, run_encode, tokenizer


def load_index():
    docs = json.loads((DATA / "docs.json").read_text())
    dim = docs["dim"]
    bits = np.fromfile(DATA / "index.bin", dtype=np.uint8).reshape(-1, dim // 8)
    signs = np.unpackbits(bits, axis=1)[:, :dim].astype(np.float32) * 2 - 1  # 1 -> +1, 0 -> -1
    spans = np.fromfile(DATA / "spans.bin", dtype="<u2").reshape(-1, 2)
    return docs, signs, spans, np.array(docs["offsets"])


def encode_queries(queries):
    path = CACHE / "queries" / "search"
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl = path.with_suffix(".jsonl")
    jsonl.write_text("".join(json.dumps({"id": str(i), "text": q}) + "\n" for i, q in enumerate(queries)))
    run_encode(jsonl, path, is_query=True, workers=1)
    _, vecs, offsets = load_vectors(path)
    return [vecs[offsets[i] : offsets[i + 1]] for i in range(len(queries))]


def post_ranking(passage_scores, passage_post, n_posts):
    """Score each post by its best passage; returns (posts best first, best passage of each post)."""
    order = np.argsort(-passage_scores, kind="stable")
    best = np.full(n_posts, -1)
    for p in order:
        if best[passage_post[p]] < 0:
            best[passage_post[p]] = p
    posts = np.argsort(-passage_scores[best], kind="stable")
    return posts, best


def snippet(text, spans, header_lines, width=320):
    """Passage body around the first match, with matched words in «»."""
    header_end = 0  # passages start with the post title, and the section heading if there is one
    for _ in range(header_lines):
        header_end = text.index("\n", header_end) + 1
    body_spans = sorted({(s, e) for s, e in spans if s >= header_end})
    words = []  # expand word pieces to whole words
    for s, e in body_spans:
        while s > 0 and text[s - 1].isalnum():
            s -= 1
        while e < len(text) and text[e].isalnum():
            e += 1
        if not words or s > words[-1][1]:
            words.append((s, e))
    start = max(header_end, (words[0][0] - 80) if words else header_end)
    end = min(len(text), start + width)
    out, at = [], start
    for s, e in words:
        if s >= end:
            break
        if s >= at:
            out += [text[at:s], "«", text[s:e], "»"]
            at = e
    out.append(text[at:end])
    return ("…" if start > header_end else "") + " ".join("".join(out).split()) + ("…" if end < len(text) else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json", help="also write every passage's score and the top 10 posts here, "
                        "for scripts/check_scorer.mjs to compare against")
    args = parser.parse_args()
    reference = []

    docs, signs, spans, offsets = load_index()
    _, floats, float_offsets = load_vectors(CACHE / "embeddings" / "blog")
    if docs.get("skippedTokens"):  # compare against float32 vectors of the same tokens
        keep = punctuation_mask([p["text"] for p in docs["passages"]], float_offsets)
        floats, float_offsets = keep_tokens(floats, float_offsets, keep)
    assert (float_offsets == offsets).all(), "cached float32 vectors do not match data/: rebuild the index"
    passage_post = np.array([p["post"] for p in docs["passages"]])
    n_posts = len(docs["posts"])
    q_tok, q_prefix = tokenizer(is_query=True)

    for query, q_vecs in zip(args.queries, encode_queries(args.queries)):
        enc = q_tok.encode(q_prefix + query)
        words = [k for k, (s, e) in enumerate(enc.offsets) if e > len(q_prefix)]  # the query's own tokens
        scores = best_matches(q_vecs, signs, offsets).sum(0)
        posts, best = post_ranking(scores, passage_post, n_posts)
        reference.append({"query": query, "scores": scores.tolist(), "posts": posts[:10].tolist()})
        float_posts, _ = post_ranking(best_matches(q_vecs, floats, offsets).sum(0), passage_post, n_posts)
        float_rank = {post: r + 1 for r, post in enumerate(float_posts)}

        print(f"\n{'=' * 100}\nQuery: {query}")
        for r, post in enumerate(posts[: args.top], 1):
            p = best[post]
            passage = docs["passages"][p]
            lo, hi = offsets[p], offsets[p + 1]
            winners = (q_vecs[words] @ signs[lo:hi].T).argmax(axis=1) + lo  # best document token per query word
            # Plain ints: arithmetic on uint16 values wraps around below zero.
            matched = [(int(spans[j][0]), int(spans[j][1])) for j in winners if spans[j][1] > 0]
            section = f" › {passage['section']}" if passage["section"] else ""
            print(f"\n{r}. {docs['posts'][post]['title']}{section}   (float32 rank {float_rank[post]})")
            print(f"   {passage['url']}")
            print(f"   {snippet(passage['text'], matched, 2 if passage['section'] else 1)}")
        shared = len(set(posts[: args.top]) & set(float_posts[: args.top]))
        print(f"\nTop {args.top} posts shared with float32 scoring: {shared}/{args.top}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reference))


if __name__ == "__main__":
    main()
