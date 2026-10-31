#ifndef SPARK_OPS_TD
#define SPARK_OPS_TD
include "mlir/IR/OpBase.td"

def Spark_Dialect : Dialect {
  let name = "spark";
  let cppNamespace = "::spark::mlir";
  let summary = "Sparse predictive associative reasoning primitives";
  let description = [{
    Operations for immutable, matrix-free retrieval over SPARK memory indexes.
    Model conversion / teacher computation happens offline.
  }];
}

class Spark_Op<string mnemonic, list<Trait> traits = []>
    : Op<Spark_Dialect, mnemonic, traits>;

// M2 intentionally uses builtin integer values: ABI v1 is x86-64 LP64 and
// each 64-bit address is an in-process pointer to a fixed-size buffer.
// The runtime owns index lookup. No matrix operation is legal in this op.
def Spark_MLMProbeOp : Spark_Op<"mlm_probe"> {
  let summary = "Matrix-free masked-token probe through a versioned native ABI";
  let description = [{
    Arguments: a live index handle (i64), 32-byte signature buffer address
    (i64), top-k maximum (i32), output token buffer address (i64; 64 x i32),
    output signed-Q16 score buffer address (i64; 64 x i32).
    Attribute max_candidates limits active output count (1..64).
    Result is the number of candidates 0..64, or a negative error code.
    The operation reads external memory and writes output buffers. It is NOT
    pure, speculatable or safe for common-subexpression elimination.
    LLVM lowering calls spark_mlm_probe_v1(i64,i64,i32,i64,i64)->i32.
    Addresses must be non-zero; runtime validates only non-null, not ownership.
    Use only within a trusted local process on 64-bit little-endian systems.
  }];
  let results = (outs I32:$count);
  let arguments = (ins I64:$index_handle, I64:$signature_address,
                       I64:$token_out_address, I64:$score_out_address,
                       I32Attr:$max_candidates);
  let hasVerifier = 1;
}
#endif // SPARK_OPS_TD
