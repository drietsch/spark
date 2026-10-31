// RUN: spark-opt --spark-probe-to-func %s | FileCheck %s
// CHECK: func.func @spark_mlm_probe_v1(i64, i64, i32, i64, i64) -> i32
// CHECK: func.call @spark_mlm_probe_v1
// CHECK-NOT: "spark.mlm_probe"
module {
  func.func @probe(%handle: i64, %signature_address: i64,
                   %token_out_address: i64, %score_out_address: i64) -> i32 {
    %count = "spark.mlm_probe"(%handle, %signature_address,
                                %token_out_address, %score_out_address)
      {max_candidates = 32 : i32}
      : (i64, i64, i64, i64) -> i32
    return %count : i32
  }
}
