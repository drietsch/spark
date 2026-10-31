#include "SparkDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  ::mlir::DialectRegistry registry;
  registry.insert<spark::mlir::SparkDialect, ::mlir::func::FuncDialect>();
  return ::mlir::failed(::mlir::MlirOptMain(argc, argv, "SPARK IR tool\n", registry));
}
