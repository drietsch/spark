# M4: ModernBERT teacher / SPARK student evaluation

This stage **does not** claim successful semantic translation of ModernBERT or
Transformer-level model quality. It implements a reproducible evaluation harness,
not an actual teacher-model benchmark result. The model could not be fetched in
the original offline development environment.

## Requirements

- Linux (for native shared libraries and meaningful `ru_maxrss` output).
- Python 3.10+, CMake, C++17 compiler.
- For teacher collection only: `torch`, `transformers>=4.48.0`, Internet on first model download.
- **No** PyTorch, Transformers or GPU at **student inference** time.

ModernBERT-base is a **masked-language encoder**, not an autoregressive
next-token model. Evaluated outputs are masked-token candidate IDs, not
conversation turns or tokens/second during text generation.

## Quick start

```bash
python3 -m pip install 'transformers>=4.48.0' torch
bash scripts/run_m4_modernbert.sh examples/modernbert_masks.jsonl artifacts/m4
cat artifacts/m4/report.json
python3 -m unittest discover -s tests -v
```

To collect on GPU set `SPARK_TEACHER_DEVICE=cuda`. The sample input only has 20
sentences; **results from it have no statistical reliability**. Bring a
substantially larger held-out corpus before drawing quality conclusions.

For custom datasets, one JSONL object per line:

```json
{"text":"The capital of France is [MASK].","group_id":"document-123"}
```

Each sentence must contain exactly one literal `[MASK]`. Use the same
`group_id` for all masks derived from the same underlying document; this helps
prevent document-level train/validation leakage. The trace collector stores
source sentences locally. Avoid sensitive or proprietary text unless you
intend to persist it. Traces include a per-example teacher-forward duration,
device and (when available) source-model commit revision.

## Commands, separately

```bash
python3 tools/modernbert_teacher.py my_masks.jsonl artifacts/teacher.jsonl \
    --model answerdotai/ModernBERT-base --emit-hidden --topk 16
python3 tools/modernbert_semantic.py compile artifacts/teacher.jsonl artifacts/student
cmake -S . -B build && cmake --build build -j2
python3 tools/modernbert_benchmark.py artifacts/teacher.jsonl artifacts/student \
    --build build --repeat 10 --output artifacts/benchmark.json
```

When experimenting **only** with synthetic fixture traces, append
`--allow-synthetic`. Never use that flag to report genuine teacher results.
For strict Python-vs-native top-token parity add `--require-native-parity`.
The C++ scoring uses fixed-point approximations and may differ on near ties.

## Metrics and interpretation

- **Teacher top-1 agreement:** comparison against the *teacher's own* top token.
  It is NOT label accuracy, general intelligence, or transformer equivalence.
- **Coverage:** fraction of masked-token contexts returning any candidate;
  missing predictions count as incorrect in the all-query agreement.
- **Training-majority baseline:** comparison against the most frequently
  predicted teacher token among training traces; distinguishes some trivial
  constant predictors from learned behavior.
- **Signature bit agreement:** how well the integer student reproduces teacher
  hidden-state random projections; this is not downstream task accuracy.
- **Known-feature fraction:** fraction of the query's features represented by
  the trained encoder. High coverage alone is not semantic generalization.
- **Native/Python disagreement:** differs because C++ uses Q16 approximations.
- **p50/p95 latency:** wall time for repeated queries, with model/index loading
  deliberately excluded; C-ABI calls include Python `ctypes` marshalling
  overhead, so this is not pure compiler-kernel timing.
- **Bytes and RSS:** disk artifact sizes and overall *process peak* RSS, not
  exact per-index resident memory. Native and Python in the same benchmark
  process share some allocations; make separate process-level measurements to
  compare memory use properly.

Both training and validation sets must be nonempty. Duplicate masks are rejected;
exact token-sequence and normalized source-text / explicit source-group overlap
across splits causes a hard failure. These checks cannot detect **near-duplicate
content** or verify a trace was genuinely produced by the named model; use
external checksums and independent test data.

## What M4 proves (and does not)

M4 tests the trace audit, benchmark statistics, native C ABI invocation and
report generation against **synthetic fixtures**. It **does not** demonstrate
ModernBERT-quality retrieval, new reasoning capabilities, or 30 token/s LLM
generation. A real-model report requires collecting actual teacher traces on
an Internet-connected machine and should include teacher revision, held-out
sources, at least several hundred evaluation masks, and confidence intervals.

The MLIR/LLVM SDK compilation remains a separate, unverified gate; see
`docs/M3_NATIVE_SEMANTIC.md` for how to run it with LLVM installed.
