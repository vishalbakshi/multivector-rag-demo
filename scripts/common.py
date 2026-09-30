"""Helpers shared by the Python scripts: paths, the pinned model, and calls into encode.mjs."""

import hashlib
import json
import string
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA = ROOT / "data"
NODE = ROOT / ".tools" / "node" / "bin" / "node"
MODEL = json.loads((ROOT / "model.json").read_text())
MODEL_DIR = CACHE / "models" / MODEL["repo"].replace("/", "__") / MODEL["revision"]


def model_file(name):
    """Path to one of the model's files, downloaded into the same cache layout colbert.mjs uses."""
    path = MODEL_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{name}"
        with urllib.request.urlopen(url) as res:
            part.write_bytes(res.read())
        part.rename(path)
    return path


def tokenizer(is_query=False):
    """The model's tokenizer, set up the way pylate-rs tokenizes documents or queries.

    Returns (tokenizer, prefix). pylate-rs prepends the prefix to the text and truncates to
    the model's length limit, counting the special tokens.
    """
    config = json.loads(model_file("config_sentence_transformers.json").read_text())
    kind = "query" if is_query else "document"
    tok = Tokenizer.from_file(str(model_file("tokenizer.json")))
    tok.no_padding()
    tok.enable_truncation(config[f"{kind}_length"])
    return tok, config[f"{kind}_prefix"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_encode(jsonl, out_prefix, is_query=False, workers=6):
    cmd = [str(NODE), str(ROOT / "scripts" / "encode.mjs"), str(jsonl), str(out_prefix), "--workers", str(workers)]
    subprocess.run(cmd + (["--query"] if is_query else []), check=True)


def load_vectors(prefix):
    """Load encode.mjs output: (metadata, float32 vectors, offsets where each record's vectors start)."""
    meta = json.loads(Path(f"{prefix}.json").read_text())
    vecs = np.fromfile(f"{prefix}.f32", dtype=np.float32).reshape(-1, meta["dim"])
    offsets = np.array(meta["offsets"])
    assert len(vecs) == offsets[-1], f"{prefix}.f32 does not match {prefix}.json"
    return meta, vecs, offsets


def punctuation_mask(texts, offsets):
    """True for each document token vector to keep, False for punctuation tokens.

    Matches PyLate's default for documents: leave out tokens that are a single ASCII
    punctuation mark. Special tokens and the prefix are kept.
    """
    tok, prefix = tokenizer()
    skip = {tok.token_to_id(c) for c in string.punctuation} - {None}
    ids = [e.ids for e in tok.encode_batch([prefix + t for t in texts])]
    assert [len(i) for i in ids] == np.diff(offsets).tolist(), "tokenizer disagrees with the cached vectors"
    return ~np.isin(np.concatenate(ids), list(skip))


def keep_tokens(vecs, offsets, keep):
    """Drop the vectors where keep is False; returns (vectors, new offsets)."""
    counts = np.add.reduceat(keep.astype(np.int64), offsets[:-1])
    return vecs[keep], np.concatenate([[0], np.cumsum(counts)])


def best_matches(q_vecs, d_vecs, offsets):
    """For each query token, its best similarity within each document: (query tokens, documents).

    Summing over query tokens gives the MaxSim score.
    """
    return np.maximum.reduceat(q_vecs @ d_vecs.T, offsets[:-1], axis=1)
