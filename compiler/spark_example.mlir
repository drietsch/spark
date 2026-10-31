// SPARK-IR v0.1: illustrative *proposed* MLIR generic assembly.
// Not parsed without registering custom spark types and operations.
module attributes {spark.abi_major = 1 : i32, spark.abi_minor = 0 : i32} {
  func.func @predict(%tokens: memref<?xi32>, %state: !spark.state,
                     %index: !spark.index) -> (i32, !spark.state) {
    %sig = "spark.encode"(%tokens, %state) {bits = 256 : i32, scheme = "featurehash-v1"}
           : (memref<?xi32>, !spark.state) -> !spark.sig<256>
    %hits = "spark.probe"(%index, %sig) {max_candidates = 64 : i32, probes = 8 : i32}
            : (!spark.index, !spark.sig<256>) -> !spark.candidates<64>
    %dist = "spark.score"(%hits) {rule = "integer-vote-v1"}
            : (!spark.candidates<64>) -> !spark.distribution<64>
    %token = "spark.select"(%dist) {mode = "argmax"}
             : (!spark.distribution<64>) -> i32
    %next = "spark.advance"(%state, %token)
            : (!spark.state, i32) -> !spark.state
    return %token, %next : i32, !spark.state
  }
}
