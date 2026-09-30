# multivector-rag-demo

Which files do what:

1. `index.html` is the page itself: search box (query vector is in fp32), score heatmap, collapsible JSON and the Generate answer button.
2. `scorer.js` ranks all 3,417 passages against your fp32 query using the 1-bit document vectors, and `llm-worker.js` runs Pleias-RAG-350M on the GPU in a background thread to stream the answer.
3. `data/` is the prebuilt index of your blog (`index.bin` holds the binary document vectors, `docs.json` the passages and links, `spans.bin` token positions for the offline tools), and `model.json` pins the ColBERT model version.
4. `scripts/` holds the offline tools, none of which the page needs: `build_index.py` rebuilds `data/` from your blog, `search.py` and `check_scorer.mjs` test search results, and `eval_quality.py` produced the SciFact numbers in results/scifact.md.
5. `package.json`, `package-lock.json` and `requirements.txt` list the Node and Python dependencies for those scripts.


