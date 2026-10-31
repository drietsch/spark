# SPARK Compiler / Runtime v0.1 — Technical specification (proposal)

**Status:** Research and implementation proposal, 2026-10-31. The accompanying
C++ reference baseline is actually executable; the SPARK dialect, LLVM lowering,
ANN search, transformer converter and 1-TB distributed runtime are **not implemented**.
The baseline is **not** a quality-equivalent conversion of an existing LLM.

## 0. Purpose and exclusions

Transform observed behavior of a teacher LLM T into a **new** associative
model S, so that `P_S(next | context)` approximates `P_T(next | context)`.
Teacher inference may use normal matrix operations during **offline** data
collection. SPARK inference and its runtime execute no matrix multiplication.
Training / distillation and code generation are **separate** operations.

Non-goals for v0.1: exact weight conversion, guaranteed quality equivalence,
secret extraction from weights, automatic Dense-2T inference at 30 tok/s,
full ONNX-to-SPARK graph semantics, or transparent support for all tokenizers.

## 1. Front-end and converter

Input sources (planned): Hugging Face transformer + safetensors, ONNX model,
GGUF plus model-specific runtime adapter. **Model weights are not a lookup
index** and cannot be reinterpreted as transition records. An adapter must
execute the teacher to collect traces. An HF/ONNX parser can identify topology,
not magically translate dense matmuls into memory-only logic.

Trace schema (planned):

```json
{
  "schema":"spark.trace.v1", "teacher_id":"sha256:...",
  "tokenizer_sha256":"...", "context_ids":[10,20,30],
  "teacher_topk":[{"id":40,"logprob":-0.2},{"id":41,"logprob":-1.7}],
  "residual_prob":0.03, "metadata":{"split":"train"}
}
```

Capture greedy and sampled contexts; include rare terms, multi-turn sequences,
long-range dependency tasks, code, maths, negation and synthetic minimal pairs.
Use prompt/continuation disjointness and source-based holdout to reduce leakage.
Do not store user secrets, copyrighted corpora without rights, or personal data
without appropriate authority. Teacher provenance and tokenizer identity are
mandatory. Prefer a frozen teacher and fixed trace version.

Learning stages:
1. v0.1 exact-context n-gram histogram (implemented; limited quality).
2. Binary signature learning using feature hashing / hyperdimensional encoding;
   256/512/1024-bit codes, supervised by teacher next-token distributions.
3. Candidate retrieval with exact inverted index or LSH + bounded probing.
4. Tree/rule-based integer reranking and state transitions; avoid runtime
   matrix kernels. Calibrate probabilities offline.
5. Active distillation: query teacher on mismatches and ambiguous contexts.

At stage 2 onward, validate teacher/student drift. OOD accuracy is unknown and
cannot be inferred from index size or runtime throughput.

## 2. IR dialect and contracts

Dialect name: `spark`; namespace: `::spark`; TableGen/ODS operation definitions;
version `0.1`; use `BytecodeDialectInterface` for evolution; all implementations
must provide parse/print/verify and documented memory effects.

Custom types:
- `!spark.state`: immutable logical inference state (RNG seed and history in
  encoded form; implementation may use opaque reference with explicit lifetime).
- `!spark.sig<256>`: immutable 256-bit signature, bit 0 = lowest bit of word 0.
- `!spark.index`: read-only snapshot of an index, with model ownership.
- `!spark.candidates<64>`: bounded sequence (token `u32`, score `i32`, count `u32`).
- `!spark.distribution<64>`: bounded scored next-token candidates.
- `!spark.shard`: partition identifier, not an implicit network fetch.

Operations (proposed, not yet registered):

| Operation | Inputs -> outputs | Exact semantics / constraints |
|---|---|---|
| `spark.encode` | context, state -> sig | pure deterministic encoding; fixed `bits`, `scheme`; max context known |
| `spark.probe` | index, sig -> candidates | bounded retrieval from immutable snapshot; `max_candidates` and `probes` checked |
| `spark.similarity` | sig, sig -> i32 | Hamming distance = popcount(xor) |
| `spark.merge` | candidates, candidates -> candidates | sum same-token counts, stable ordering, saturate on overflow |
| `spark.score` | candidates -> distribution | integer fixed-point scoring, no model-dependent hidden calls |
| `spark.select` | distribution -> token | deterministic argmax with smallest ID tie-break, or explicitly seeded sampling |
| `spark.advance` | state, token -> state | append token/update rolling hashes without mutation of input state |
| `spark.route` | sig -> shard | deterministic partition mapping from manifest; no dynamic discovery |
| `spark.prefetch` | index, sig -> () | non-semantic hint; may be erased with no output change |

Verifier rules: signatures power-of-64 and 64..4096 bits; capacity 1..1024;
no unbounded buffers in `spark.probe`; explicit index snapshots; token IDs
below vocabulary limit; bounded full integer score; valid scheme and ABI
revision; no hidden mutation or unseeded randomness. Verify layouts and
endianness at model load; reject unrecognized op/revision. `spark.prefetch`
must be observably equivalent to a no-op.

See `compiler/spark_example.mlir` for illustrative **generic** assembly.
It is not valid executable MLIR until the types/ops have been registered.

## 3. Pass pipeline (proposed)

1. `spark-verify` — verify types, bounds, model/tokenizer compatibility.
2. `spark-canonicalize` — fold redundant state ops and constant routes.
3. `spark-fuse-probes` — merge compatible independent lookups, preserving ranking.
4. `spark-plan-index` — choose exact, bucketed, LSH and page layout from traces.
5. `spark-plan-placement` — assign fixed shards to NUMA nodes/servers.
6. `spark-prefetch-schedule` — software pipelining of independent lookups.
7. `spark-lower-high` — lower dialect into loops, memrefs, arithmetic and
   constrained calls to the SPARK C runtime.
8. `convert-scf-to-cf`, `finalize-memref-to-llvm`,
   `convert-arith-to-llvm`, `convert-func-to-llvm`, `convert-cf-to-llvm`,
   `reconcile-unrealized-casts` (subject to verified LLVM/MLIR version).
9. `translate-to-llvmir` → `opt` / `llc` or Clang/LLVM linker pipeline.
10. `spark-audit-matrix-free` — enforce allowlist of generated operations and
    runtime calls. Reject known dot/matmul/BLAS ops and unauthorized externals.

This audit is an engineering guard, **not a formal proof** that no optimized
piece of code performs matrix-like arithmetic. Lowering `spark.probe` to a
runtime C-call is preferred initially; replacing that call with inlined IR is
an optimization, not a requirement for LLVM compatibility.

C/C++ implementation: `lib/SPARK/IR`, `lib/SPARK/Transforms`,
`lib/SPARK/Conversion/SPARKToLLVM`, `tools/spark-opt`, `tools/sparkc`.
Use TableGen ODS for ops, types and generated verification; conversion
patterns + `ConversionTarget` / `TypeConverter`; `mlir-opt`-style pass manager.
Pin one LLVM/MLIR monorepo commit. Don't assume APIs will remain source-stable.

## 4. Runtime C ABI v1.0 (implemented subset)

Defined by `include/spark_runtime.h` (source of truth):

```c
#define SPARK_ABI_MAJOR 1u
#define SPARK_ABI_MINOR 0u

typedef struct spark_model spark_model_t;
typedef struct spark_session spark_session_t;
typedef enum spark_status { SPARK_OK = 0, SPARK_EINVAL = 1,
  SPARK_EIO = 2, SPARK_EFORMAT = 3, SPARK_ENOMEM = 4,
  SPARK_ENOMATCH = 5, SPARK_EUNSUPPORTED = 6 } spark_status_t;

typedef struct spark_step_stats {
  uint32_t abi_size;
  uint32_t matched_order;
  uint64_t matched_candidates;
  uint64_t reserved[2];
} spark_step_stats_t;

spark_status_t spark_model_open(const char *index_path, spark_model_t **out_model);
void spark_model_close(spark_model_t *model);
spark_status_t spark_session_open(const spark_model_t *model, spark_session_t **out_session);
void spark_session_close(spark_session_t *session);
spark_status_t spark_session_reset(spark_session_t *session);
spark_status_t spark_session_feed(spark_session_t *session, uint32_t token_id);
spark_status_t spark_session_step(spark_session_t *session, uint32_t *next_token,
                                  spark_step_stats_t *optional_stats);
```

ABI rules: C linkage, x86-64 System V for Linux MVP; fixed-width integer
fields; opaque handles; caller owns returned `spark_model_t`/`spark_session_t`;
model outlives sessions; each session bound to one worker, model can be read
concurrently after load; no exceptions cross ABI; no raw pointer on disk.
`optional_stats->abi_size` must match the header's struct size. Model memory
is immutable. `step` appends token to the session only on success. `ENOMATCH`
does not mutate session. Tokenizer is external for v0.1; caller passes token IDs.

LLVM codegen can initially emit `call @spark_session_step` via `LLVM::CallOp`
for a matrix-free host-side kernel, with an explicit context/session pointer.
Future v1.1 additions (NOT implemented) may expose `spark_rt_probe_v1`
for bulk candidate lookup, `spark_rt_merge` for federated reductions, and
per-shard node scheduling. Maintain ABI major compatibility or fail to load.

## 5. On-disk formats

### `*.idx` exact-index baseline — **implemented**

Little-endian. Header 64 bytes (`<8sHHIQQQQQQ` in Python):

| Offset | Field | Bytes | Value v0.1 |
|---:|---|---:|---|
| 0 | magic | 8 | ASCII `SPARKIDX` |
| 8 | version_major | 2 | 1 |
| 10 | version_minor | 2 | 0 |
| 12 | flags | 4 | 0 |
| 16 | bucket_count | 8 | 0 (not supported yet) |
| 24 | entry_count | 8 | number of records |
| 32 | seed | 8 | 0 (fixed FNV-1a keying) |
| 40 | bucket_offset | 8 | 0 |
| 48 | entry_offset | 8 | 64 |
| 56 | file_bytes | 8 | total file size |

Each following record: `<QII` = `uint64 key`, `uint32 token_id`,
`uint32 count`; sorted by `(key, token_id)`, no duplicate key/token pairs.
Key = 64-bit FNV-1a of the little-endian byte representation of the last
`n` context tokens (1 <= n <= 4). Query longest suffix first, then shorter.
The trainer aggregates teacher trace weights and saturates at `UINT32_MAX`.

**Security and scalability limitations of baseline:** Only weak 64-bit hash,
no collision verification, no cryptographic integrity inside the `.idx`,
loading into a `std::vector` rather than `mmap`, hard cap 100 million records,
no LSH, no tokenizer, no distributed sharding. A production loader must verify
cryptographic checksums and use read-only memory mapping and per-shard bounds.

### `*.sparkpkg` package — **proposed**

Directory (or a tar-like container) with versioned manifest:

```
model.sparkpkg/
  manifest.json
  tokenizer/tokenizer.json
  code/x86_64-linux-gnu.so
  ir/model.spark.mlirbc          # OPTIONAL debug/deployment artifact
  indexes/node-0000.idx
  indexes/node-0001.idx
  quality/teacher-student.json
  quality/dataset-card.json
```

Manifest fields: `format_version`, `runtime_abi`, `source_model_id`,
`teacher_sha256`, `tokenizer_sha256`, `context_limit`, `signature_scheme`,
`signature_bits`, `vocab_size`, `candidate_limit`, `routing_version`,
`target_triple`, `target_cpu_features`, `shards[]` (`filename`, `bytes`,
`sha256`, `node`, `numa`), `quality_metrics`, `build_provenance`.
Store exact tokenizer config, BOS/EOS/pad IDs, stop rules and normalization.
Reject tokenizer or ABI mismatch at load. Never assume MLIR bytecode from a
changing custom dialect is automatically backward-compatible: implement
dialect-specific upgrade/version hooks.

### Production memory index — **future**

A multi-level, pointer-free, immutable index with 64-bit offsets, fixed-width
bucket directories and contiguous candidate arrays; consider LSH/Hamming
lookup for 256-bit signatures. Flat exact index first for correct testing.
Allocate node-local memory on Linux NUMA; `mmap` immutable shards and prefault
as appropriate. Use bounded top-K output so remote messaging returns only
small candidate summaries, never entire 1-TB pages.

## 6. Model optimization and cluster execution

For each token, target < 33.33 ms total at the **application** layer for
30 tok/s. This is *not* guaranteed by aggregate bandwidth. Keep independent
probes in flight; split batches by index locality; merge local candidates
once per token. Assign shards statically to 4 EPYC nodes or NUMA domains.
Use an asynchronous fan-out/fan-in protocol with bounded responses and an
index version per request. If a shard misses the deadline, detect and report
a partial result; never silently reclassify it as an exact result.

A 1-TiB corpus is not the same as 2T learned parameters. Quality, recall
and token latency must be measured independently. Before purchasing multi-
server hardware, demonstrate a good held-out student on a 1–10 GB dataset.

## 7. Verification and success criteria

Implement in ascending order:

- [x] C ABI header, deterministic C++ baseline; exact-context trainer and demo.
- [x] Test one next-token prediction and suffix-backoff behavior.
- [ ] Build `spark` dialect through TableGen and register all v0.1 types/ops.
- [ ] Implement verifier / roundtrip printer-parser / negative malformed-IR tests.
- [ ] Interpreter matching reference runtime, then `spark-to-llvm` pass.
- [ ] Compare MLIR, LLVM and C runtime outputs on 10,000 fixed cases exactly.
- [ ] Implement 256-bit signatures with fixed bit semantics and LSH search.
- [ ] Implement teacher-distillation pipeline and held-out evaluation.
- [ ] Model packages with authenticated manifest and mmap/NUMA loader.
- [ ] Profile 50th/95th/99th-percentile token latency, DDR traffic, random
      access miss rate, memory footprint and inter-node traffic.
- [ ] Scale to 256 GiB, then 1 TiB, only if quality/latency gates pass.

Minimum **engineering** gates: exact deterministic compiled-vs-interpreter
match for reference traces; zero unknown runtime externals under matrix-free
mode; malformed/corrupt input rejected; ABI size/major version checked;
compilation determinism on fixed toolchain; no crashes under memory/index
fuzzing. **Model** gates: measured held-out teacher top-1 agreement,
KL/Jensen-Shannon on calibrated top-K+residual, perplexity only when full
probability distribution is available, long-context and compositional
out-of-distribution tasks, and general language-generation benchmarks.
No preset quality threshold is claimed to have been met.

## 8. MVP schedule / deliverables

1. **M0:** executable exact-context reference (delivered in this starter).
2. **M1:** proper TableGen types/ops and `spark-opt`; parse/print/verify.
3. **M2:** `spark-to-llvm` conversion with 3–5 ops to external runtime calls.
4. **M3:** `sparkc` CLI, trace-to-index conversion and package validation.
5. **M4:** learned signatures, bounded ANN search, teacher/student evaluation.
6. **M5:** SIMD prefetch, `mmap`, NUMA shards and distributed prototype.

M0 validates the software interfaces and deterministic retrieval, *not*
semantic LLM conversion. M4 is the research critical path.

## References used for the compiler design

- https://mlir.llvm.org/docs/DefiningDialects/Operations/
- https://mlir.llvm.org/docs/DialectConversion/
- https://mlir.llvm.org/docs/TargetLLVMIR/
- https://mlir.llvm.org/docs/PassManagement/
- https://mlir.llvm.org/docs/BytecodeFormat/
- https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
- https://github.com/onnx/onnx-mlir
- https://github.com/llvm/torch-mlir
- https://ai.meta.com/research/publications/generalization-through-memorization-nearest-neighbor-language-models/
