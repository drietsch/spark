# SPARK v0.1 – Design specification + executable reference baseline

**Status:** Research prototype. This contains a working C++17 reference runtime and
teacher-trace-to-index generator, plus a **proposed** MLIR dialect interface.
It is **not** a complete MLIR/LLVM model compiler, a transformer converter, or a
2T parameter LLM. The current baseline uses exact-context n-gram matching and
FNV-1a hashing, not learned associative representations.

## Run locally

```sh
python3 tools/train_from_traces.py examples/teacher_traces.jsonl model.idx
cmake -S . -B build && cmake --build build -j
./build/spark_demo model.idx 10 20 30
python3 -m unittest discover -s tests -v
```

Expected demo: `next=40 order=3 candidates=2`.

## Files

- `docs/SPEC.md` — compiler dialect, lowering, ABI and data contract.
- `include/spark_runtime.h` — C ABI v1.0 reference.
- `runtime/spark_runtime.cpp` — deterministic matrix-free runtime.
- `tools/train_from_traces.py` — tiny teacher-trace ingest baseline.
- `compiler/spark_example.mlir` — illustrative proposed dialect, not yet executable.

Production requirements not implemented: bit-256 signatures, LSH/ANN index,
MLIR TableGen compiler, semantic distillation, mmap and NUMA, distributed
execution, signed model manifests, sandboxing, and perf profiling.

## LFM3-1.7B-GGUF teacher experiment

To try this project with LiquidAI/LFM3-1.7B-GGUF through a local
llama.cpp server, see [`docs/LFM3_TRIAL.md`](docs/LFM3_TRIAL.md). This adds a
real-model HTTP trace capture adapter, a prompt-level holdout evaluation
and tests. The real GGUF download/inference must run in your environment.


## ModernBERT masked-token experiment (new)

[ModernBERT experiment documentation](docs/MODERNBERT_TRIAL.md) explains how
to use `answerdotai/ModernBERT-base` as an **offline** masked-LM teacher
and compile its top-k predictions into an experimental matrix-free 256-bit
SimHash/LSH student. The teacher uses PyTorch; the student has no neural
dependencies. The experiment **does not** convert transformer weights or
perform general-purpose text generation. It measures holdout teacher agreement.

```sh
python -m pip install 'transformers>=4.48.0' torch
python tools/modernbert_teacher.py examples/modernbert_masks.jsonl artifacts/modernbert_traces.jsonl
python tools/modernbert_spark.py compile artifacts/modernbert_traces.jsonl artifacts/modernbert.smlm
python tools/modernbert_spark.py evaluate artifacts/modernbert_traces.jsonl artifacts/modernbert.smlm
python -m unittest discover -s tests -v
```

## ModernBERT semantic-signature extension (M1)

[Semantic ModernBERT experiment](docs/SEMANTIC_MODERNBERT.md): the offline
teacher collector supports `--emit-hidden`, and
`tools/modernbert_semantic.py` learns 256-bit integer-vote signatures
from teacher embeddings, then creates a fingerprint-bound `SPARKMLM` index.

The `mlir/` subtree contains an experimental out-of-tree
`spark.mlm_probe` dialect definition and verifier. **It is not yet
build-tested against MLIR or lowered to LLVM**; the matrix-free Python
student is tested offline. This is a supervised retrieval experiment,
not verified semantic equivalence to ModernBERT.
