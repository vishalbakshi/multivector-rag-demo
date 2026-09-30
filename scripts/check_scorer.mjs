// Check that scorer.js (what the page runs) scores the blog index like scripts/search.py does.
//
// Encodes each query with pylate-rs, scores every passage with scorer.js, and compares with the
// reference written by `search.py --json`: every passage score, and the order of the top 10 posts.
//
// Usage: node scripts/check_scorer.mjs cache/parity/python.json
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { lookupTables, scorePassages, tokenScores, topPosts } from "../scorer.js";
import { loadColBERT, ROOT } from "./lib/colbert.mjs";

const TOLERANCE = 1e-3; // Python sums in float32, scorer.js mostly in float64

const reference = JSON.parse(await readFile(process.argv[2], "utf8"));
const docs = JSON.parse(await readFile(join(ROOT, "data", "docs.json"), "utf8"));
const bits = new Uint8Array(await readFile(join(ROOT, "data", "index.bin")));
const offsets = Int32Array.from(docs.offsets);
const passagePost = docs.passages.map((p) => p.post);
const model = await loadColBERT();

let failures = 0;
for (const { query, scores: expected, posts: expectedPosts } of reference) {
  let start = performance.now();
  const { embeddings } = model.encode({ sentences: [query] }, true);
  const encodeMs = performance.now() - start;

  start = performance.now();
  const tables = lookupTables(Float32Array.from(embeddings[0].flat()), docs.dim);
  const scores = scorePassages(tables, bits, offsets);
  const scoreMs = performance.now() - start;

  const maxDiff = scores.reduce((m, s, i) => Math.max(m, Math.abs(s - expected[i])), 0);
  const top = topPosts(scores, passagePost, 10);
  const posts = top.map((r) => r.post);
  const sameOrder = posts.length === expectedPosts.length && posts.every((p, i) => p === expectedPosts[i]);
  // Per-token scores must add up to each result's score.
  const partsAddUp = top.every((r) => Math.abs(tokenScores(tables, bits, offsets, r.passage).reduce((a, b) => a + b) - r.score) < 1e-6);
  const ok = maxDiff < TOLERANCE && sameOrder && partsAddUp;
  failures += !ok;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${query.padEnd(52)} max score difference ${maxDiff.toExponential(1)}, ` +
      `top 10 posts ${sameOrder ? "identical" : "differ"}, token scores ${partsAddUp ? "add up" : "DON'T add up"} | ` +
      `encode ${encodeMs.toFixed(0)} ms, score ${scoreMs.toFixed(0)} ms`,
  );
}
console.log(failures ? `\n${failures} of ${reference.length} queries differ` : `\nAll ${reference.length} queries match`);
process.exit(failures ? 1 : 0);
