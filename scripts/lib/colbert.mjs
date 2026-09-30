// Load a PyLate ColBERT model into pylate-rs's WebAssembly build from Node,
// the same build index.html will use in the browser.
import { readFileSync } from "node:fs";
import { access, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ColBERT, initSync } from "pylate-rs";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

// Pinned in model.json so the Python scripts read the same model and revision.
export const MODEL = JSON.parse(readFileSync(join(ROOT, "model.json"), "utf8"));
export const PYLATE_RS_VERSION = JSON.parse(
  readFileSync(join(ROOT, "node_modules", "pylate-rs", "package.json"), "utf8"),
).version;

// Order matches the ColBERT constructor arguments.
const FILES = [
  "model.safetensors",
  "1_Dense/model.safetensors",
  "tokenizer.json",
  "config.json",
  "config_sentence_transformers.json",
  "1_Dense/config.json",
  "special_tokens_map.json",
];

const exists = (p) => access(p).then(() => true, () => false);

export async function fetchModelFiles({ repo, revision } = MODEL) {
  const dir = join(ROOT, "cache", "models", repo.replace("/", "__"), revision);
  const files = [];
  for (const name of FILES) {
    const path = join(dir, name);
    if (!(await exists(path))) {
      const url = `https://huggingface.co/${repo}/resolve/${revision}/${name}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      await mkdir(dirname(path), { recursive: true });
      // Write then rename, so an interrupted download never leaves a truncated file behind.
      await writeFile(`${path}.part`, Buffer.from(await res.arrayBuffer()));
      await rename(`${path}.part`, path);
    }
    files.push(new Uint8Array(await readFile(path)));
  }
  return files;
}

export async function loadColBERT({ batchSize = 32 } = {}) {
  initSync({ module: await readFile(join(ROOT, "node_modules", "pylate-rs", "pylate_rs_bg.wasm")) });
  return new ColBERT(...(await fetchModelFiles()), batchSize);
}
