add_mlir_dialect_library(SparkMLIRDialect
  SparkDialect.cpp
  DEPENDS SparkOpsIncGen
  LINK_LIBS PUBLIC MLIRIR
)
