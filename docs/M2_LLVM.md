# SPARK M2: MLM probe to LLVM call ABI

**Status:** Native C++ ABI implementation is compiled and independently tested on
Linux x86-64. `spark-probe-to-func` is implemented as MLIR C++ source but **has
not been compiled or integration-tested** without the external MLIR SDK. It is
not a production compiler and does not transform ModernBERT weights.

## ABI and semantics

The signed-index parser accepts `SPARKMLM` v1.0, exact 64-byte header, 48-byte
records and 8-byte `(uint32 token, float32 probability)` candidates. Index
entries can be standard 32-bit x 8 LSH bands (SimHash) or 8-bit x 32 bands
(semantic encoder). The compiled runtime gets a **precomputed** 256-bit
signature, not raw token IDs. For semantic indexes, callers must separately
verify that their encoder fingerprint matches the index header; the C ABI
accepts a trusted prevalidated signature and does not load the JSON encoder.

`int32 spark_mlm_probe_v1(uint64 handle, uint64 signature_address,
 int32 max_candidates, uint64 token_out_address, uint64 score_out_address)`:

- Index handle is created with `spark_mlm_index_open_v1()`.
- Input signature is 32 bytes in little-endian bit order.
- Both output addresses reference separate writable arrays of **64 int32**.
- Returns number of active candidates (0..64); negative return means error.
- Unused token slots are `-1`, unused scores are 0.
- Ranked nearest neighbors: Hamming distance (ascending), then record ID
  (ascending). At most eight neighbors contribute to candidate votes.
- Probability is rounded to Q16 first. Hamming distance is weighted with
  `((512-distance)/256)^4` and rounded back to signed Q16.
- Candidate output: scores (descending), token ID (ascending). The C++ kernel
  intentionally uses integer rounding, so exact score equality with the
  Python floating-point implementation is **not** guaranteed.
- No cross-query fallback to full scans; zero retrieved records returns zero.

The `spark.mlm_probe` op takes four `i64` process-local addresses/handles and a
compile-time `max_candidates` attribute (1..64), and returns `i32`. The
`spark-probe-to-func` pass replaces it with a `func.call` of an automatically
declared ABI symbol. Standard `convert-func-to-llvm` subsequently lowers this
call to LLVM IR. This is an intentionally narrow 64-bit little-endian target
contract, not a portable pointer ABI. Future versions will use typed buffers,
end-to-end memory effects, an owned index handle and complete shape checking.

## Test C++ runtime (works without MLIR)

```sh
cmake -S . -B build
cmake --build build -j
python3 -m unittest discover -s tests -v
```

To probe from the demo, create an index via the Python `compile` command and
pass a 64-character hexadecimal signature in **byte order** as stored on disk:

```sh
./build/spark_mlm_demo /path/to/index.smlm 0000...0000
```

## Compile IR (requires an external MLIR / LLVM SDK)

```sh
cmake -S mlir -B build-mlir -DMLIR_DIR=/path/to/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/lib/cmake/llvm
cmake --build build-mlir -j
build-mlir/tools/spark-opt --spark-probe-to-func mlir/test/mlm_probe.mlir \
  > /tmp/spark-lowered.mlir
mlir-opt --convert-arith-to-llvm --convert-func-to-llvm /tmp/spark-lowered.mlir \
  > /tmp/spark-llvm.mlir
mlir-translate --mlir-to-llvmir /tmp/spark-llvm.mlir > /tmp/spark.ll
```

The generated LLVM IR calls `spark_mlm_probe_v1`. Link to
`build/libspark_mlm_runtime.so` and call with a live index handle and valid
in-process buffers. A separately built driver is needed; `mlir/test/mlm_probe.mlir`
only defines a callable function, not a standalone executable.

## Non-goals and next steps

- This does **not** lower transformer matrix operations; it only lowers a SPARK
  native retrieval primitive.
- The MLIR path, LLVM translation and linker path still need compilation and
  end-to-end testing with a pinned LLVM/MLIR release in CI.
- CPU feature-specific optimizations, mmap, NUMA sharding, thread-safe
  incremental model updates, and safety hardening are future work.
- `spark.mlm_probe` receives caller-generated binary signatures; the learned
  signature encoder is currently Python-based, not compiled to LLVM.
