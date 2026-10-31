# M5 — large, held-out ModernBERT → SPARK benchmark

The GitHub Actions workflow `modernbert-large.yml` generates 5,000 training
and 500 validation masked examples from the published `Salesforce/wikitext`,
`wikitext-2-raw-v1` dataset. This dataset has independent original splits,
not a random split of a tiny sample.

## Reproducibility
- One deterministic word is masked per sentence. A SHA-256 seed of 20261031
  determines mask positions **before** looking at ModernBERT outputs.
- The collector uses `--honor-input-split` to preserve the source train and
  validation partitions. All masks derived from one source line share a group.
- The manifest stores dataset fingerprints, exact counts, and the SHA-256 hash
  of generated inputs. The trace audit rejects exact or group crossover.
- The student trains only on the train split.

## Evaluation
The workflow downloads and runs the **real** `answerdotai/ModernBERT-base`
teacher, trains a 256-bit supervised signature encoder, compiles a memory
index, and executes the matrix-free native C++ SPARK student. Reports include
held-out teacher top-1 agreement, 95% Wilson confidence interval, majority
baseline, retrieval/feature coverage, Python/native disagreement, latency,
artifact sizes, and process peak RSS.

The `modernbert-large` GitHub Actions artifact contains input masks,
trace/teacher observations, compiled student and a report (7-day retention).

## Run locally
```bash
python -m pip install 'transformers>=4.48,<5' 'datasets>=3,<5' torch
python tools/prepare_large_modernbert.py artifacts/large/masks.jsonl --train 5000 --validation 500
python tools/modernbert_teacher.py artifacts/large/masks.jsonl artifacts/large/teacher.jsonl --device cpu --emit-hidden --topk 16 --max-length 128 --honor-input-split
python tools/modernbert_semantic.py compile artifacts/large/teacher.jsonl artifacts/large/model
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
python tools/modernbert_benchmark.py artifacts/large/teacher.jsonl artifacts/large/model --build build --repeat 2 --output artifacts/large/report.json
```

**Limitations:** This is masked-token teacher agreement, **not** accuracy
against the original missing word, autoregressive text generation, or
Transformer-weight-equivalent compilation. Large near-duplicate contexts can
still occur. The current SPARK MLIR-to-LLVM pipeline is not independently
verified by this workflow.
