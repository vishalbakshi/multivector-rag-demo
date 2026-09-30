// Encode a JSONL file of {"id", "text"} records with pylate-rs (WebAssembly) and save the token vectors.
//
// Writes <out>.f32 (every token vector, float32, back to back) and <out>.json
// (ids, where each record's vectors start and end, vector size, model, and a
// SHA-256 of the input file so callers can tell whether the cache is stale).
//
// The WebAssembly build runs on one core, so records are split across worker threads,
// each with its own copy of the model.
//
// Usage: node scripts/encode.mjs <input.jsonl> <out-prefix> [--query] [--workers 6] [--batch-size 32] [--limit N]
import { createHash } from "node:crypto";
import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";
import { loadColBERT, MODEL, PYLATE_RS_VERSION } from "./lib/colbert.mjs";

if (isMainThread) {
  await main();
} else {
  await encodeInWorker(workerData);
}

async function encodeInWorker({ texts, isQuery, batchSize }) {
  const model = await loadColBERT({ batchSize });
  for (let i = 0; i < texts.length; i += batchSize) {
    // One batch per call: pylate-rs pads each batch to its own longest record,
    // and joining batches of different lengths fails.
    const { embeddings } = model.encode({ sentences: texts.slice(i, i + batchSize) }, isQuery);
    // Documents come back zero-padded to the batch's longest one. Real vectors have unit length.
    const kept = embeddings.map((rows) => (isQuery ? rows : rows.filter((row) => row.some((v) => v !== 0))));
    const dim = kept[0][0].length;
    const buf = new Float32Array(kept.reduce((n, rows) => n + rows.length, 0) * dim);
    let at = 0;
    for (const rows of kept) for (const row of rows) buf.set(row, (at++) * dim);
    parentPort.postMessage({ dim, lengths: kept.map((rows) => rows.length), buf }, [buf.buffer]);
  }
}

async function main() {
  const args = process.argv.slice(2);
  const flag = (name, fallback) => {
    const i = args.indexOf(name);
    return i >= 0 ? Number(args.splice(i, 2)[1]) : fallback;
  };
  const isQuery = args.includes("--query");
  const workers = flag("--workers", 6);
  const batchSize = flag("--batch-size", 32);
  const limit = flag("--limit", Infinity);
  const [input, outPrefix] = args.filter((a) => !a.startsWith("--"));
  if (!input || !outPrefix) {
    console.error("Usage: node scripts/encode.mjs <input.jsonl> <out-prefix> [--query] [--workers 6] [--batch-size 32] [--limit N]");
    process.exit(1);
  }

  const raw = await readFile(input);
  const inputSha256 = createHash("sha256").update(raw).digest("hex");
  const records = raw
    .toString("utf8")
    .split("\n")
    .filter(Boolean)
    .slice(0, limit)
    .map((line) => JSON.parse(line));

  // Contiguous slices, so writing worker 0's output, then worker 1's, and so on keeps the input order.
  const n = Math.max(1, Math.min(workers, Math.ceil(records.length / batchSize)));
  const size = Math.ceil(records.length / n);
  const results = Array.from({ length: n }, () => []);
  const start = performance.now();
  let done = 0;
  let dim = 0;

  await Promise.all(
    results.map((chunks, w) => {
      const worker = new Worker(new URL(import.meta.url), {
        workerData: { texts: records.slice(w * size, (w + 1) * size).map((r) => r.text), isQuery, batchSize },
      });
      worker.on("message", (msg) => {
        chunks.push(msg);
        dim = msg.dim;
        done += msg.lengths.length;
        const secs = (performance.now() - start) / 1000;
        const eta = ((records.length - done) * secs) / done;
        console.error(`${done}/${records.length} records, ${(done / secs).toFixed(1)}/s, ~${Math.ceil(eta / 60)} min left`);
      });
      return new Promise((resolve, reject) => {
        worker.on("error", reject);
        worker.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`worker ${w} exited with ${code}`))));
      });
    }),
  );

  await mkdir(dirname(outPrefix), { recursive: true });
  const out = createWriteStream(`${outPrefix}.f32`);
  const offsets = [0];
  for (const chunks of results) {
    for (const { lengths, buf } of chunks) {
      if (!out.write(Buffer.from(buf.buffer))) await once(out, "drain");
      for (const len of lengths) offsets.push(offsets.at(-1) + len);
    }
  }
  out.end();
  await once(out, "finish");
  await writeFile(
    `${outPrefix}.json`,
    JSON.stringify({
      model: MODEL,
      pylateRs: PYLATE_RS_VERSION,
      inputSha256: Number.isFinite(limit) ? null : inputSha256,
      isQuery,
      dim,
      ids: records.map((r) => r.id),
      offsets,
    }),
  );
  console.error(`Wrote ${offsets.at(-1)} vectors of size ${dim} in ${((performance.now() - start) / 1000).toFixed(1)} s`);
}
