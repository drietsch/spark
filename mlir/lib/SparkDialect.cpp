#include "SparkDialect.h"
#include "SparkOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "SparkDialect.cpp.inc"

#define GET_OP_CLASSES
#include "SparkOps.cpp.inc"

namespace spark::mlir {
void SparkDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SparkOps.cpp.inc"
      >();
}

::mlir::LogicalResult MLMProbeOp::verify() {
  const auto limit = getMaxCandidates();
  if (limit < 1 || limit > 64)
    return emitOpError("max_candidates must be in [1, 64]");
  return ::mlir::success();
}
} // namespace spark::mlir
