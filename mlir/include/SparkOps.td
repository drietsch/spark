#ifndef SPARK_OPS_TD
#define SPARK_OPS_TD
include "mlir/IR/OpBase.td"
def Spark_Dialect : Dialect {
  let name = "spark";
  let cppNamespace = "::spark::mlir";
  let summary = "Sparse predictive associative reasoning primitives";
}
class Spark_Op<string mnemonic, list<Trait> traits = []>
    : Op<Spark_Dialect, mnemonic, traits>;
def Spark_MLMProbeOp : Spark_Op<"mlm_probe"> {
  let summary = "Probe read-only masked-language-model index";
  let description = [{ signature: tensor<32xi8>; tokens and Q16.16 scores:
    tensor<64xi32>. Token -1 is unused. Opaque i64 index handle. Index is
    externally read-only. Lowering to LLVM/C ABI is not implemented. }];
  let arguments = (ins I64:$index_handle, AnyRankedTensor:$signature,
                       I32Attr:$max_candidates);
  let results = (outs AnyRankedTensor:$token_ids,
                      AnyRankedTensor:$scores);
  let hasVerifier = 1;
}
#endif
