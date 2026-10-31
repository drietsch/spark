#include "SparkDialect.h"
#include "SparkPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  spark::mlir::registerSparkPasses();
  ::mlir::DialectRegistry registry;
  registry.insert<spark::mlir::SparkDialect, ::mlir::func::FuncDialect, ::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect>();
  return ::mlir::failed(::mlir::MlirOptMain(argc, argv, "SPARK IR tool\n", registry));
}
