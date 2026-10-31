# SPARK × ModernBERT-base: masked-token distillation experiment

**Status:** Experimental baseline, not a weight compiler and not a working LLVM/MLIR lowering. The model is `answerdotai/ModernBERT-base` (149M parameters, bidirectional masked language model). It is **not** an autoregressive generator; measurements here are masked-token predictions, not tokens/second text generation.

## Intended architecture

- **Offline teacher**: `AutoModelForMaskedLM` with matrix operations is used to infer token probabilities at one `[MASK]` position per sentence. It produces teacher traces.
- **Matrix-free SPARK student**: stable 256-bit feature SimHash of nearby token IDs; eight 32-bit locality-sensitive-hash bands; vote among retrieved training contexts weighted by Hamming proximity and teacher's stored top-k probabilities.
- **No neural weights at inference**: student uses only Python standard-library integer, hash, dictionary and file operations. This is NOT semantic equivalence to ModernBERT. The token IDs at inference must use the same tokenizer; offline teacher collection needs Hugging Face packages.
- **Honest holdout**: identical tokenized prompts get a deterministic train/validation assignment. The evaluation reports top-1 teacher agreement, exact retrieval coverage and fallback use.

## Setup

Requires Python 3.10+, PyTorch and `transformers>=4.48.0` for teacher collection. CPU is the default; GPU is optional. The student and tests require no external packages.

```sh
python -m pip install 'transformers>=4.48.0' torch
python tools/modernbert_teacher.py examples/modernbert_masks.jsonl artifacts/modernbert_traces.jsonl --device cpu --topk 16
python tools/modernbert_spark.py compile artifacts/modernbert_traces.jsonl artifacts/modernbert.smlm
python tools/modernbert_spark.py evaluate artifacts/modernbert_traces.jsonl artifacts/modernbert.smlm
# Optional diagnostic only: full scan when no LSH bucket matches.
python tools/modernbert_spark.py evaluate artifacts/modernbert_traces.jsonl artifacts/modernbert.smlm --fallback-nearest
python -m unittest discover -s tests -v
```

The sample corpus contains only 20 examples: **it is a wiring smoke test, not a meaningful model-quality evaluation**. To study generalization, collect thousands of independently authored masked prompts or separate entire source documents between train and validation. Never use validation/test traces to build an index.

## Example teacher input

```json
{"text":"The capital of France is [MASK]."}
```

## Teacher trace schema

Each JSONL row is:

```json
{
  "input_ids": [101, 999, 102],
  "mask_index": 1,
  "topk": [{"token": 42, "prob": 0.9}, {"token": 43, "prob": 0.1}],
  "split": "train",
  "model": "answerdotai/ModernBERT-base",
  "source_text": "An example [MASK]."
}
```

Token IDs in this JSON are illustrative, not actual ModernBERT tokenizer outputs. `topk` probabilities are softmax values over the complete teacher vocabulary, **not renormalized within the top-k set**.

## Binary student format `SPARKMLM` v1

Little-endian:

| Structure | Format | Size | Meaning |
|---|---|---:|---|
| Header | `<8sHHIQQQQQQ` | 64 bytes | magic, major, minor, flags, rows, candidates, row offset, values offset, total bytes, reserved |
| Row | `<32sQII` | 48 bytes | 256-bit signature, candidate offset in *elements*, candidate count, reserved |
| Candidate | `<If` | 8 bytes | uint32 token id and float32 probability |

The index is read-only and deliberately simple. A production format needs checksums, tokenizer fingerprint, model metadata, sharding, concurrency/NUMA support, and a verifier for offsets and allocation limits. The loader verifies basic bounds, versions, and probabilities.

## What success looks like

- First: smoke tests pass, teacher traces can be collected locally and the SPARK student can be compiled.
- Then: report **validation** agreement, index size, coverage, latency and compare to exact-match context lookup and a majority-class baseline.
- Only then: try larger contexts, learned signatures, advanced retrieval or an MLIR `spark.mlm_probe` op.

**Important limitation:** Similar-looking token contexts are not the same as learned semantics. This implementation is an experimental matrix-free behavioral retrieval baseline; it does not demonstrate generalization equivalent to the teacher.
