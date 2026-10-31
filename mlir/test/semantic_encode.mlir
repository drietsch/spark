// RUN: spark-opt --spark-probe-to-func %s | FileCheck %s
// CHECK: func.func @spark_semantic_encode_v1(i64, i64, i32, i32, i64) -> i32
// CHECK: func.call @spark_semantic_encode_v1
// CHECK: func.call @spark_mlm_probe_v1
// CHECK-NOT: "spark.semantic_encode"
// CHECK-NOT: "spark.mlm_probe"
module {
  func.func @predict(%encoder: i64, %index: i64, %tokens: i64,
                     %size: i32, %mask: i32, %signature: i64,
                     %out_tokens: i64, %out_scores: i64) -> i32 {
    %status = "spark.semantic_encode"(%encoder, %tokens, %size, %mask, %signature)
      : (i64, i64, i32, i32, i64) -> i32
    %zero = arith.constant 0 : i32
    %valid = arith.cmpi eq, %status, %zero : i32
    cf.cond_br %valid, ^probe, ^fail
  ^probe:
    %count = "spark.mlm_probe"(%index, %signature, %out_tokens, %out_scores)
      {max_candidates = 32 : i32}
      : (i64, i64, i64, i64) -> i32
    return %count : i32
  ^fail:
    return %status : i32
  }
}
