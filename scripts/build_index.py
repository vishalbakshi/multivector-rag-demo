"""Build the blog's search index: split posts into passages, encode them, and pack sign bits.

1. Get the blog's source files: a read-only, text-only copy in cache/blog-source.
2. Split each post into passages that fit the model's 300-token limit.
3. Encode the passages with pylate-rs (WebAssembly, via the project's portable Node).
   Full float32 vectors stay in cache/embeddings, so changing precision later never re-encodes.
   The cache is reused only if the passages and model are unchanged.
4. Pack each vector's signs into data/index.bin, each token's position in its passage into
   data/spans.bin, and passage text and links into data/docs.json.

Usage: .venv/bin/python scripts/build_index.py [--blog-dir PATH] [--refresh] [--split-only] [--keep-punctuation]
  --blog-dir          use an existing checkout of the blog instead of the cached copy
  --refresh           re-download the cached copy of the blog
  --split-only        stop after step 2, to inspect cache/blog/passages.jsonl
  --keep-punctuation  keep punctuation token vectors in data/ (by default they are left out, as PyLate does)
"""

import argparse
import html
import json
import math
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from common import CACHE, DATA, MODEL, keep_tokens, load_vectors, punctuation_mask, run_encode, sha256, tokenizer

BLOG_REPO = "https://github.com/vishalbakshi/blog.git"
SITE = "https://vishalbakshi.github.io/blog/"
# Post sources, plus the rendered pages, which hold the real anchor of every heading.
SPARSE = ["/posts/**/*.ipynb", "/posts/**/*.md", "/posts/**/*.qmd", "/docs/posts/**/*.html"]
WORK = CACHE / "blog"
EMB = CACHE / "embeddings" / "blog"

FRONT_MATTER = re.compile(r"\A\s*---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
CODE_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.S | re.M)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
EXPLICIT_ID = re.compile(r"\{[^}\n]*#([\w-]+)[^}\n]*\}")
CLEANERS = [  # (pattern, replacement), applied in order to prose
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),  # images, including ones embedded as data URIs
    (re.compile(r"<img\b[^>]*>", re.I), ""),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links keep their text
    (re.compile(r"<https?://[^>\s]+>"), ""),
    (re.compile(r"</?[A-Za-z][^>\n]*>"), " "),  # other HTML tags keep their text
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"\{\{<.*?>\}\}"), ""),  # Quarto shortcodes
    (re.compile(r"\{\s*(?:[#.][\w-]+|[\w-]+=)[^}\n]*\}"), ""),  # attributes like {.callout} or {width=50%}
    (re.compile(r"^[ \t]*:::.*$", re.M), ""),  # Quarto divs
    (re.compile(r"^[ \t|:-]*-[ \t|:-]*$", re.M), ""),  # table alignment rows like |---| or :-: :-:
    (re.compile(r"^[ \t]*([-*_][ \t]*){3,}$", re.M), ""),  # horizontal rules
    (re.compile(r"^[ \t]*>[ \t]?", re.M), ""),  # blockquote markers
    (re.compile(r"^[ \t]*[-*+][ \t]+", re.M), ""),  # list bullets
    (re.compile(r"\||\*\*|`"), " "),  # table pipes, bold markers, inline-code backticks
]


def blog_source(blog_dir, refresh):
    if blog_dir:
        return Path(blog_dir).expanduser().resolve()
    src = CACHE / "blog-source"
    if refresh and src.exists():
        shutil.rmtree(src)
    def git(*args):
        subprocess.run(["git", *args], check=True)
    if not src.exists():
        git("clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout", BLOG_REPO, str(src))
        git("-C", str(src), "sparse-checkout", "set", "--no-cone", *SPARSE)
        git("-C", str(src), "checkout", "--quiet")
    else:  # picks up any files added to SPARSE since the copy was made
        git("-C", str(src), "sparse-checkout", "set", "--no-cone", *SPARSE)
    return src


def parse_front_matter(text):
    """Returns (metadata, rest of text). Falls back to reading the title line if the YAML is invalid."""
    m = FRONT_MATTER.match(text)
    if not m:
        return None, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        title = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        meta = {"title": title.group(1).strip("'\"") if title else ""}
    return meta, text[m.end():]


def read_post(folder):
    """Returns (source file, metadata, markdown prose) for a post folder, or None if it has no source."""
    sources = sorted(p for p in folder.iterdir() if p.suffix in (".ipynb", ".md", ".qmd"))
    if not sources:
        return None
    src = next((p for p in sources if p.stem == "index"), sources[0])
    meta = None
    if src.suffix == ".ipynb":
        parts = []
        for i, cell in enumerate(json.loads(src.read_text())["cells"]):
            text = "".join(cell["source"])
            if i == 0 and cell["cell_type"] in ("raw", "markdown"):  # front matter lives in the first cell
                meta, text = parse_front_matter(text)
            if cell["cell_type"] == "markdown":  # code cells and their outputs are left out
                parts.append(text)
        markdown = "\n\n".join(parts)
    else:
        meta, markdown = parse_front_matter(src.read_text())
    return src, meta or {}, markdown


def clean(text):
    for pattern, replacement in CLEANERS:
        text = pattern.sub(replacement, text)
    # After tags are gone, so an escaped "&lt;div&gt;" in the prose stays as literal text.
    return html.unescape(text)


def pandoc_id(heading):
    """The anchor Quarto (via pandoc) gives a heading."""
    s = re.sub(r"[^\w\s.-]", "", heading)
    s = re.sub(r"\s+", "-", s.strip()).lower()
    return re.sub(r"^[^a-z]+", "", s) or "section"


def sections(markdown):
    """Split prose at headings: yields (heading, anchor, paragraphs). The text before the first heading has no heading."""
    markdown = CODE_FENCE.sub("\n", COMMENT.sub("", markdown))
    heading, anchor, lines, seen = "", None, [], {}
    out = []
    for line in markdown.split("\n") + ["# <end>"]:
        m = HEADING.match(line)
        if not m:
            lines.append(line)
            continue
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", clean("\n".join(lines)))]
        out.append((heading, anchor, [p for p in paragraphs if p]))
        raw = m.group(2)
        heading = " ".join(clean(EXPLICIT_ID.sub("", raw)).split())
        explicit = EXPLICIT_ID.search(raw)
        anchor = explicit.group(1) if explicit else pandoc_id(heading)
        if not explicit and anchor in seen:  # pandoc numbers repeated headings: #setup, #setup-1, ...
            seen[anchor] += 1
            anchor = f"{anchor}-{seen[anchor]}"
        else:
            seen.setdefault(anchor, 0)
        lines = []
    return [s for s in out if s[2]]


def split_to_fit(paragraph, budget, count):
    """Split a paragraph into pieces of at most `budget` tokens, at sentence boundaries when possible."""
    if count(paragraph) <= budget:
        return [paragraph]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        joined = f"{current} {sentence}".strip()
        if count(joined) <= budget:
            current = joined
            continue
        if current:
            pieces.append(current)
        current = sentence
        while count(current) > budget:  # one very long "sentence": cut between words
            words = current.split(" ")
            lo = 1
            while lo < len(words) and count(" ".join(words[: lo + 1])) <= budget:
                lo += 1
            pieces.append(" ".join(words[:lo]))
            current = " ".join(words[lo:])
    return pieces + ([current] if current else [])


def passages_for_section(header, paragraphs, budget, count):
    """Pack paragraphs into evenly sized passages of at most `budget` tokens each."""
    units = []  # (text, tokens, starts a new paragraph)
    for paragraph in paragraphs:
        for k, piece in enumerate(split_to_fit(paragraph, budget, count)):
            units.append((piece, count(piece), k == 0))
    total = sum(u[1] for u in units)
    if total == 0:
        return []
    target = math.ceil(total / math.ceil(total / budget))
    chunks, current, size = [], "", 0
    for text, tokens, new_paragraph in units:
        if current and size + tokens > target:
            chunks.append(current)
            current, size = "", 0
        current += ("\n\n" if new_paragraph else " ") + text if current else text
        size += tokens
    chunks.append(current)
    return [f"{header}\n{chunk}" for chunk in chunks]


def split_posts(src):
    tok, prefix = tokenizer()
    limit = tok.truncation["max_length"]
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)

    def page_ids(page):
        """Element ids on the rendered page, or an empty set if it isn't there to check."""
        rendered = src / "docs" / page
        return set(re.findall(r'\bid="([^"]+)"', rendered.read_text(errors="replace"))) if rendered.exists() else set()

    posts, passages, skipped = [], [], []
    anchors_checked = anchors_found = 0
    for folder in sorted(p for p in (src / "posts").iterdir() if p.is_dir()):
        post = read_post(folder)
        if post is None:
            continue
        source, meta, markdown = post
        if meta.get("draft") is True:
            skipped.append((folder.name, "draft"))
            continue
        title = " ".join(clean(str(meta.get("title") or folder.name)).split())
        page = f"posts/{folder.name}/{source.stem}.html"
        ids = page_ids(page)
        post_passages = []
        for heading, anchor, paragraphs in sections(markdown):
            header = f"{title}\n{heading}" if heading else title
            link = page
            if anchor:  # only link to anchors that exist on the rendered page
                anchors_checked += 1
                if anchor in ids:
                    anchors_found += 1
                    link = f"{page}#{anchor}"
            # The header and the prefix and special tokens count against the limit too.
            budget = limit - len(tok.encode(prefix + header).ids)
            for text in passages_for_section(header, paragraphs, budget, count):
                post_passages.append({"section": heading, "url": SITE + link, "text": text})
        if not post_passages:
            skipped.append((folder.name, "no prose"))
            continue
        for p in post_passages:
            p["post"] = len(posts)
            p["id"] = f"{folder.name}:{len(passages)}"
            passages.append(p)
        posts.append({
            "title": title,
            "date": str(meta.get("date") or folder.name[:10]),
            "url": SITE + page,
            "categories": [str(c) for c in meta.get("categories") or []],
        })

    for p in passages:  # nothing may be cut off by the model's length limit
        enc = tok.encode(prefix + p["text"])
        assert not enc.overflowing and len(enc.ids) <= limit, f"passage {p['id']} is too long"
    print(f"Split {len(posts)} posts into {len(passages):,} passages "
          f"(skipped: {', '.join(f'{n} ({why})' for n, why in skipped) or 'none'})")
    print(f"Section links: {anchors_found:,}/{anchors_checked:,} anchors found on the rendered pages; "
          "the rest link to the top of the post")
    return posts, passages


def pack(posts, passages, commit, keep_punctuation):
    """Write the published files: sign bits, token positions, and passage text and links."""
    meta, vecs, offsets = load_vectors(EMB)
    assert meta["ids"] == [p["id"] for p in passages], "cached vectors do not match the passages"
    tok, prefix = tokenizer()
    spans = []
    for p, n in zip(passages, np.diff(offsets)):
        enc = tok.encode(prefix + p["text"])
        assert len(enc.ids) == n, f"{p['id']}: {len(enc.ids)} tokens here, {n} vectors from pylate-rs"
        # Character offsets become UTF-16 offsets, the units JavaScript strings are indexed in.
        utf16 = np.concatenate([[0], np.cumsum([2 if ord(c) > 0xFFFF else 1 for c in p["text"]])])
        assert utf16[-1] < 2**16
        for s, e in enc.offsets:
            inside = e > len(prefix)  # special tokens and the prefix have no position in the text
            spans.append((utf16[s - len(prefix)], utf16[e - len(prefix)]) if inside else (0, 0))
    spans = np.array(spans, dtype="<u2")
    if not keep_punctuation:  # PyLate's default for documents; pylate-rs keeps them
        keep = punctuation_mask([p["text"] for p in passages], offsets)
        vecs, offsets = keep_tokens(vecs, offsets, keep)
        spans = spans[keep]

    DATA.mkdir(exist_ok=True)
    (DATA / "index.bin").write_bytes(np.packbits(vecs > 0, axis=1).tobytes())
    (DATA / "spans.bin").write_bytes(spans.tobytes())
    docs = {
        "format": 1,
        "model": meta["model"],
        "pylateRs": meta["pylateRs"],
        "dim": meta["dim"],
        "bits": 1,
        "bitOrder": "most significant bit first",
        "spans": "uint16 little-endian (start, end) per token vector, UTF-16 offsets into the passage text; (0, 0) for special tokens",
        "skippedTokens": None if keep_punctuation else "single ASCII punctuation marks",
        "site": SITE,
        "sourceCommit": commit,
        "built": date.today().isoformat(),
        "posts": posts,
        "passages": [{"post": p["post"], "section": p["section"], "url": p["url"], "text": p["text"]} for p in passages],
        "offsets": offsets.tolist(),
    }
    (DATA / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")))
    for name in ("index.bin", "spans.bin", "docs.json"):
        print(f"  data/{name}: {(DATA / name).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--blog-dir")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--keep-punctuation", action="store_true")
    args = parser.parse_args()

    src = blog_source(args.blog_dir, args.refresh)
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    posts, passages = split_posts(src)

    WORK.mkdir(parents=True, exist_ok=True)
    jsonl = WORK / "passages.jsonl"
    jsonl.write_text("".join(json.dumps({"id": p["id"], "text": p["text"]}, ensure_ascii=False) + "\n" for p in passages))
    if args.split_only:
        return
    cached = Path(f"{EMB}.json")
    fresh = cached.exists() and (m := json.loads(cached.read_text()))["inputSha256"] == sha256(jsonl) and m["model"] == MODEL
    if fresh:
        print("Passages and model unchanged: reusing cached vectors")
    else:
        run_encode(jsonl, EMB)

    print(f"Packing {sum(len(p['text']) for p in passages):,} characters of passages:")
    pack(posts, passages, commit or None, args.keep_punctuation)


if __name__ == "__main__":
    main()
