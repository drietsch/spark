#!/usr/bin/env bash
# Requires an actual LLVM/MLIR SDK, unlike the portable native reference tests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_OPT="${SPARK_OPT:-$ROOT/build-mlir/tools/spark-opt}"
if [[ ! -x "$SPARK_OPT" ]]; then
  echo "Missing spark-opt at $SPARK_OPT. Build: cmake -S mlir -B build-mlir -DCMAKE_PREFIX_PATH=/path/to/llvm-install && cmake --build build-mlir" >&2
  exit 2
fi
command -v mlir-opt >/dev/null || { echo 'mlir-opt missing' >&2; exit 2; }
command -v mlir-translate >/dev/null || { echo 'mlir-translate missing' >&2; exit 2; }
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$SPARK_OPT" --spark-probe-to-func "$ROOT/mlir/test/semantic_encode.mlir" > "$out/lowered.mlir"
mlir-opt "$out/lowered.mlir" \
  --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts > "$out/llvm.mlir"
mlir-translate --mlir-to-llvmir "$out/llvm.mlir" > "$out/spark.ll"
grep -q '@spark_semantic_encode_v1' "$out/spark.ll"
grep -q '@spark_mlm_probe_v1' "$out/spark.ll"
echo 'M3 MLIR -> LLVM IR translation passed'
