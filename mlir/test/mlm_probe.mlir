module {
  func.func @probe(%handle: i64, %signature: tensor<32xi8>)
      -> (tensor<64xi32>, tensor<64xi32>) {
    %tokens, %scores = "spark.mlm_probe"(%handle, %signature)
      {max_candidates = 32 : i32}
      : (i64, tensor<32xi8>) -> (tensor<64xi32>, tensor<64xi32>)
    return %tokens, %scores : tensor<64xi32>, tensor<64xi32>
  }
}
