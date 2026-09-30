# Binary document vectors on SciFact

Run on 2026-09-27 with `scripts/eval_quality.py`.

- Model: `lightonai/answerai-colbert-small-v1` at revision `e507cd1`, encoded with pylate-rs (WebAssembly build, in Node)
- Corpus: BEIR SciFact, 5,183 documents, 1,388,618 token vectors of 96 numbers
  (167,230 of them punctuation); 300 test queries
- Scoring: exact MaxSim against every document. Quantization simulated in float32. int8 uses one scale per vector.
- "No punctuation" leaves out document tokens that are a single ASCII punctuation mark, PyLate's default.
  pylate-rs keeps them. Queries are unchanged.
- Published NDCG@10 for this model: 74.77

| Query x document | Document tokens | NDCG@10 | Change | Recall@100 | Same top 10 as fp32 | Document vectors |
|---|---|---:|---:|---:|---:|---:|
| fp32 x fp32 | all | 74.18 | – | 95.60 | 100.0% | 533.2 MB |
| fp32 x fp32 | no punctuation | 74.56 | +0.37 | 95.60 | 96.4% | 469.0 MB |
| int8 x int8 | all | 74.51 | +0.32 | 95.93 | 96.5% | 133.3 MB |
| int8 x int8 | no punctuation | 74.34 | +0.16 | 95.93 | 94.7% | 117.3 MB |
| fp32 x binary | all | 70.56 | -3.62 | 93.37 | 56.3% | 16.7 MB |
| fp32 x binary | no punctuation | 70.18 | -4.01 | 93.70 | 56.8% | 14.7 MB |
| int8 x binary | all | 70.56 | -3.62 | 93.37 | 56.2% | 16.7 MB |
| int8 x binary | no punctuation | 70.17 | -4.01 | 93.70 | 56.7% | 14.7 MB |
| binary x binary | all | 60.16 | -14.03 | 86.22 | 40.0% | 16.7 MB |
| binary x binary | no punctuation | 59.83 | -14.36 | 86.56 | 39.6% | 14.7 MB |

"Change" and "Same top 10 as fp32" compare against fp32 x fp32 with every token (the first row).
"Same top 10" is the average share of that top 10 each row also returns.
Document vector sizes count raw vector payload only.
