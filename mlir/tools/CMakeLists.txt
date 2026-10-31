add_executable(spark-opt spark-opt.cpp)
target_link_libraries(spark-opt PRIVATE SparkMLIRDialect MLIROptLib MLIRIR MLIRFuncDialect MLIRArithDialect MLIRControlFlowDialect)
