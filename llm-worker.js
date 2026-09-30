// Writes an answer from the top search results with Pleias-RAG-350M, streaming its raw output
// (reasoning trace included), in a background thread so the page stays responsive.
//
// Messages in:  { type: "generate", id, query, sources: [text, ...] } | { type: "stop" }
// Messages out: { id, type: "loading" } | { id, type: "progress", loaded, total } (downloading)
//               | { id, type: "reading", sources, tokens } (reading the prompt)
//               | { id, type: "text", text, tokens, rate } (writing; tokens so far, tokens per second)
//               | { id, type: "done", seconds, tokens } | { id, type: "error", message }

// transformers.min.js bundles ONNX Runtime; the .web build expects the page to provide it.
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

const MODEL = "onnx-community/Pleias-RAG-350M-ONNX";
const REVISION = "ce1c1168050c6ff53db493517252dc5d0e71199f";
const CONTEXT = 4096; // the model's max_position_embeddings

const stopper = new InterruptableStoppingCriteria();
let loading = null;
let queue = Promise.resolve();
let latest = null; // id of the only request still wanted

// 4-bit weights with 32-bit math on WebGPU. The q4f16 build is smaller but produced only
// [UNK] on an AMD RDNA1 GPU (16-bit overflow), and the CPU builds are far too slow.
function load(id) {
  loading ??= (async () => {
    if (!(await navigator.gpu?.requestAdapter())) throw new Error("This needs a browser with WebGPU.");
    return Promise.all([
      AutoTokenizer.from_pretrained(MODEL, { revision: REVISION }),
      AutoModelForCausalLM.from_pretrained(MODEL, {
        revision: REVISION,
        device: "webgpu",
        dtype: "q4",
        progress_callback: (p) => {
          if (p.status === "progress" && p.file.endsWith(".onnx")) {
            self.postMessage({ id, type: "progress", loaded: p.loaded, total: p.total });
          }
        },
      }),
    ]);
  })().catch((error) => {
    loading = null; // let the next click try again
    throw error;
  });
  return loading;
}

// The layout Pleias's own library builds (pleias_rag_interface/RAGWithCitations.py).
// The query is framed as a question about the sources: asked bare ("What does the cavern
// represent?"), the model assumed a natural cavern and judged the blog passages irrelevant.
// "According to these blog posts:" was worse: the model took the question to be about blogging.
function prompt(query, sources) {
  let text = `<|query_start|>According to the sources: ${query}<|query_end|>\n`;
  sources.forEach((source, i) => (text += `<|source_start|><|source_id|>${i + 1} ${source}<|source_end|>\n`));
  return text + "<|language_start|>\n";
}

async function generate({ id, query, sources }) {
  try {
    self.postMessage({ id, type: "loading" });
    const [tokenizer, model] = await load(id);
    if (id !== latest) return; // superseded or stopped while the model was loading
    const inputs = tokenizer(prompt(query, sources));
    self.postMessage({ id, type: "reading", sources: sources.length, tokens: inputs.input_ids.dims[1] });
    let tokens = 0;
    let firstAt = 0;
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false, // keep the section markers: the raw trace is the point
      callback_function: (text) => {
        const rate = tokens > 1 ? (tokens - 1) / ((performance.now() - firstAt) / 1000) : 0;
        self.postMessage({ id, type: "text", text, tokens, rate });
      },
      token_callback_function: () => {
        tokens++;
        firstAt ||= performance.now();
      },
    });
    stopper.reset();
    const start = performance.now();
    await model.generate({
      ...inputs,
      max_new_tokens: CONTEXT - inputs.input_ids.dims[1],
      do_sample: false, // greedy, as Pleias's library does by default
      stopping_criteria: stopper,
      streamer,
    });
    self.postMessage({ id, type: "done", seconds: (performance.now() - start) / 1000, tokens });
  } catch (error) {
    self.postMessage({ id, type: "error", message: error.message });
  }
}

self.onmessage = ({ data }) => {
  stopper.interrupt(); // a new request, or a stop, ends whatever is running
  latest = data.type === "generate" ? data.id : null;
  if (data.type === "generate") queue = queue.then(() => generate(data));
};
