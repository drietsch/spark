# ModernBERT-to-SPARK semantic signature experiment (M1)

**Status:** research prototype. The matrix-free student passed offline tests in
the development environment. The actual ModernBERT model has NOT been
downloaded or benchmarked there. The out-of-tree MLIR dialect has NOT been
compiled because the MLIR SDK was unavailable. No LLVM lowering exists yet.

## Concept

Use `answerdotai/ModernBERT-base` as an offline masked-LM teacher. Collect
top-k token probabilities and optionally final-layer mask embeddings.
Sparse seeded random projections of teacher mask embeddings produce 256-bit
binary training targets. The student learns an integer feature-to-bit mapping
from the **training** examples. At inference it uses ONLY token IDs, signed
integer votes, hash-based retrieval, bit comparisons and candidate voting.
It does not load ModernBERT or any teacher hidden vectors.

This is supervised associative distillation, not a weight compiler. SPARK
does not yet preserve the model semantics and the results are not comparable
to autoregressive LLM generation.

## Run on your machine

Python 3.10+, PyTorch, Transformers >=4.48:

```bash
python -m pip install 'transformers>=4.48.0' torch

# Teacher, offline conversion step: matrix operations permitted.
python tools/modernbert_teacher.py examples/modernbert_masks.jsonl \
  artifacts/modernbert_hidden.jsonl --device cpu --emit-hidden

# Matrix-free SPARK encoder and teacher-keyed index.
python tools/modernbert_semantic.py compile \
  artifacts/modernbert_hidden.jsonl artifacts/modernbert-semantic

# Held-out teacher-agreement benchmark.
python tools/modernbert_semantic.py evaluate \
  artifacts/modernbert_hidden.jsonl artifacts/modernbert-semantic

# Diagnostic FULL SCAN if LSH misses, potentially slow:
python tools/modernbert_semantic.py evaluate \
  artifacts/modernbert_hidden.jsonl artifacts/modernbert-semantic \
  --fallback-nearest

python -m unittest discover -s tests -v
```

The generated student files are `artifacts/modernbert-semantic.encoder.json`
(learned integer weights) and `artifacts/modernbert-semantic.smlm`
(teacher target codes and stored predictions). The binary index's header has
a fingerprint of the encoder and will refuse to load with the wrong encoder.
Version-1 baseline indexes remain readable, with `flags=0`; the semantic
variant uses `flags=1` and stores the fingerprint in the reserved header field.

Metrics include retrieved fraction, covered top-1 teacher agreement,
all-prompt top-1 teacher agreement, mean matched-feature fraction, and
teacher-signature bit agreement where teacher embeddings are available.
The validation split is excluded from fitting and index creation.

**Caveats:** The tiny example data is only for wiring smoke tests. For any
scientific comparison, split by source document and evaluate on thousands of
unseen prompts, comparing against the original SimHash and a majority-class
baseline. The sparse random projection is not an optimized semantic metric.
32x8-bit LSH bands are used for learned codes to avoid near-zero coverage,
with an accompanying candidate-set cost.

## Initial MLIR operator

`mlir/include/SparkOps.td` and `mlir/lib/SparkDialect.cpp` define a
prototype `spark.mlm_probe` with:

* `i64` opaque index handle, `tensor<32xi8>` 256-bit signature
* `i32 max_candidates` attribute (1..64)
* Outputs `tensor<64xi32>` token IDs and `tensor<64xi32>` Q16.16 scores

Unused token IDs use -1. This is a dialect/verification **skeleton only**:
the C++ source and CMake integration are supplied, but uncompiled and
not lowered to LLVM or linked to the student runtime.

On a workstation with a matching LLVM + MLIR development SDK:

```bash
cmake -S mlir -B build-mlir -DMLIR_DIR=/path/to/lib/cmake/mlir
cmake --build build-mlir -j
build-mlir/tools/spark-opt mlir/test/mlm_probe.mlir
```

The externally read-only memory effect must be modeled before implementing
optimizations; do not mark the operation as pure or permit unrestricted CSE.

**Security/privacy:** Teacher traces may contain source text and embeddings.
Keep them outside version control; do not distribute traces of private text.
