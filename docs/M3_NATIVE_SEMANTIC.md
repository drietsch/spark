# M3: native learned semantic encoder and composed MLM inference

**Implemented:** portable `SPARKENC` binary export; C++17 learned-feature signature
encoder; versioned C ABI; end-to-end encode -> `SPARKMLM` retrieval; sample CLI;
Python/native parity and invalid-format tests; MLIR `spark.semantic_encode`
operation and lowering to `spark_semantic_encode_v1`.

**Not implemented:** Transformer-weight translation, true hidden-state
computation without teacher training, semantic equivalence guarantees, an
LLVM-generated executable validation in this environment, or multi-node runtime.
The MLIR integration is code/IR only until built and tested with an MLIR SDK.

## Fast local run

```bash
python -m pip install 'transformers>=4.48.0' torch
python tools/modernbert_teacher.py examples/modernbert_masks.jsonl \
  artifacts/traces.jsonl --emit-hidden
python tools/modernbert_semantic.py compile artifacts/traces.jsonl artifacts/model
cmake -S . -B build && cmake --build build -j
./build/spark_semantic_demo artifacts/model.encoder.senc artifacts/model.smlm \
  2 101 8 999 9 102
python tools/modernbert_semantic.py evaluate artifacts/traces.jsonl artifacts/model
python -m unittest discover -s tests -v
```

The sample `101 8 999 9 102` uses dummy IDs to illustrate ABI invocation;
use **actual IDs emitted by ModernBERT's tokenizer** for a meaningful result.
Teacher extraction needs the actual model downloaded separately; the offline tests
use synthetic traces and do **not** measure ModernBERT accuracy. A tiny dataset
is insufficient to claim generalization.

## Binary format SPARKENC / 1.0

Little-endian, 64-byte header (`<8sHHIQQQQIIQ`):

| Byte offset | Field | Type |
|---|---|---|
| 0 | magic `SPARKENC` | 8 bytes |
| 8 | major version = 1 | u16 |
| 10 | minor version = 0 | u16 |
| 12 | flags = 0 | u32 |
| 16 | number of feature records | u64 |
| 24 | encoder fingerprint | u64 |
| 32 | total bytes | u64 |
| 40 | payload offset = 64 | u64 |
| 48 | payload CRC32 | u32 |
| 52 | reserved = 0 | u32 |
| 56 | reserved = 0 | u64 |

Payload: **32 bytes fallback signature**, then for every feature sorted by
lexicographic ASCII key: `u16 key_length`, `key_length ASCII bytes`, exactly
**256 signed int8 weights**. Fallback and signature bit order are little-endian.
Feature names follow the Python `spark.modernbert._features()` convention:
`L{1..5}:id`, `R{1..5}:id`, `LB:id:id`, `RB:id:id`, `B:id`.

The CRC32 detects accidental corruption, *not malicious tampering*; the 64-bit
fingerprint only binds a compatible semantic index. Neither authenticates
teacher provenance. Both are verified at runtime where applicable.

## C ABI

`include/spark_semantic_runtime.h` provides:

- `spark_semantic_encoder_open_v1` / `spark_semantic_encoder_close_v1`;
- `spark_semantic_encode_v1` (produces 32-byte signature);
- `spark_semantic_predict_v1` (verifies matching encoder/index fingerprint,
  then calls `spark_mlm_probe_v1` and fills 64 output slots);
- `spark_semantic_encoder_fingerprint_v1`.

C ABI values are integer-sized **trusted in-process pointers**; never expose
these entry points directly to untrusted network clients. Current target is
64-bit little-endian Linux with 32-bit token IDs and at most 8192 input tokens.

## MLIR to LLVM IR (SDK required)

The proposed lowering is:

```
spark.semantic_encode -> func.call @spark_semantic_encode_v1
spark.mlm_probe      -> func.call @spark_mlm_probe_v1
convert-{arith,cf,func}-to-llvm -> mlir-translate --mlir-to-llvmir
```

The sample module branches on encoder status before probing. To build and
exercise the compiler path with an installed compatible LLVM/MLIR development
SDK:

```bash
cmake -S mlir -B build-mlir -DCMAKE_PREFIX_PATH=/path/to/llvm-install
cmake --build build-mlir -j
SPARK_OPT=build-mlir/tools/spark-opt bash scripts/check_m3_llvm.sh
```

The native C++ path **is** compiled and tested. The MLIR C++ component is not
yet SDK-verified in the supplied environment; its availability depends on the
installed LLVM/MLIR toolchain version. The high-level MLIR pass emits external
calls; it does not inline or independently synthesize the learned encoder.

## Reproducible test criteria

- Python/native bit-exact equality of 256-bit signatures on randomized contexts.
- Fingerprint mismatch rejected before index probe.
- Truncated/CRC-mismatched encoder artifact rejected at load.
- Native C++ winner matches Python reference on controlled examples.
- Held-out model-quality measurements separately reported; no training/validation leakage.
