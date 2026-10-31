add_mlir_dialect_library(SparkMLIRDialect
  SparkDialect.cpp
  SparkLowerProbe.cpp
  DEPENDS SparkOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRFuncDialect MLIRArithDialect MLIRPass MLIRTransforms
)
