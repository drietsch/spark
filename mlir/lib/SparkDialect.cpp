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
  auto valid = [&](::mlir::Type type, int64_t size, unsigned bits) {
    auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(type);
    return t && t.getRank() == 1 && t.getDimSize(0) == size &&
           t.getElementType().isInteger(bits);
  };
  if (!valid(getSignature().getType(), 32, 8))
    return emitOpError("signature must be tensor<32xi8>");
  if (!valid(getTokenIds().getType(), 64, 32) ||
      !valid(getScores().getType(), 64, 32))
    return emitOpError("results must be tensor<64xi32>");
  auto limit = getMaxCandidates();
  if (limit < 1 || limit > 64)
    return emitOpError("max_candidates outside [1,64]");
  return ::mlir::success();
}
}
