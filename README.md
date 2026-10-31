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

[`docs/SEMANTIC_MODERNBERT.md`](docs/SEMANTIC_MODERNBERT.md) adds a teacher-hidden-state-supervised
256-bit **integer-only** signature encoder and a binding between the learned
encoder and its SPARKMLM binary index. Run the teacher collector with
`--emit-hidden`, then `tools/modernbert_semantic.py compile` and `evaluate`.
The `mlir/` directory now contains an out-of-tree `spark.mlm_probe` ODS/C++
**dialect skeleton**, which requires an MLIR SDK to build. There is **no LLVM
lowering yet** and no claim of semantic equivalence to ModernBERT.

## M2: native MLM probe + MLIR lowering draft

A tested, matrix-free C++17 `SPARKMLM` index loader and probing ABI now lives in
`runtime/spark_mlm_runtime.cpp` and `include/spark_mlm_runtime.h`. The
`spark-probe-to-func` MLIR pass lowers `spark.mlm_probe` into a `func.call` of
`@spark_mlm_probe_v1`, which can subsequently be lowered to LLVM IR. The
**native runtime and Python/native parity tests are executable**; the MLIR C++
pass is **not yet build-tested** without the MLIR SDK. See
[`docs/M2_LLVM.md`](docs/M2_LLVM.md) for its ABI, assumptions and build steps.

## M3: matrix-free native semantic student

The semantic compiler now emits a **portable binary encoder**
`<prefix>.encoder.senc`, in addition to its JSON encoder and `.smlm`
index. C++17 `spark_semantic_runtime` generates teacher-supervised 256-bit
signatures using only integer votes and memory lookups; the composed
`spark_semantic_predict_v1` calls the native MLM index without Python or
PyTorch at inference time. The MLIR dialect includes `spark.semantic_encode`
and a func-call lowering; LLVM IR emission still requires SDK testing.

See [M3 native semantic experiment](docs/M3_NATIVE_SEMANTIC.md).

## M4: auditable ModernBERT teacher-versus-SPARK benchmark

The M4 experiment adds a real-model trace collection runner, group-aware
train/validation split checks, a training-majority baseline, masked-token
teacher agreement and native/Python comparison with latency and disk metrics.
No real ModernBERT quality score is claimed; offline CI uses synthetic fixtures.
See [M4 benchmark guide](docs/M4_BENCHMARK.md).

```sh
python -m pip install 'transformers>=4.48.0' torch
bash scripts/run_m4_modernbert.sh examples/modernbert_masks.jsonl artifacts/m4
# or benchmark already collected teacher traces without torch:
python tools/modernbert_benchmark.py artifacts/m4/teacher.jsonl artifacts/m4/student \
  --build build --output artifacts/m4/report.json
```

**ModernBERT is a bidirectional masked LM**, not an autoregressive LLM.
The current SPARK prototype remains a research baseline.


## M7: lexical memory quality breakthrough (experimental)

The corrected ModernBERT benchmark of 500 independent WikiText-2 validation
masks measured **15.2% held-out teacher top-1 agreement** for a matrix-free
lexical-plus-supervised-sparse student, up from **4.6%** for the M6 supervised
sparse baseline. Lexical-only agreement is **15.0%**. The new student builds
550,000 token observations from WikiText-2 **train only**; hyperparameter
tuning uses the teacher TRAIN split, not validation.

This is a Python prototype and **not a native C++ performance benchmark**.
See [M7 experiments](docs/M7_LEXICAL_MEMORY.md) and the
[successful GitHub run](https://github.com/drietsch/spark/actions/runs/41243701560).
