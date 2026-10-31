#include "SparkPasses.h"
#include "SparkOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace spark::mlir {
namespace {

class LowerMLMProbe final : public ::mlir::OpRewritePattern<MLMProbeOp> {
public:
  using ::mlir::OpRewritePattern<MLMProbeOp>::OpRewritePattern;

  ::mlir::LogicalResult matchAndRewrite(MLMProbeOp op,
                                        ::mlir::PatternRewriter &rewriter) const override {
    // The ABI symbol is inserted once per enclosing module by the pass.
    auto callee = op->getParentOfType<::mlir::ModuleOp>()
                      .lookupSymbol<::mlir::func::FuncOp>("spark_mlm_probe_v1");
    if (!callee) return rewriter.notifyMatchFailure(op, "runtime ABI symbol missing");
    auto limit = rewriter.create<::mlir::arith::ConstantIntOp>(
        op.getLoc(), op.getMaxCandidates(), 32);
    auto call = rewriter.create<::mlir::func::CallOp>(
        op.getLoc(), callee,
        ::mlir::ValueRange{op.getIndexHandle(), op.getSignatureAddress(),
                           limit.getResult(), op.getTokenOutAddress(),
                           op.getScoreOutAddress()});
    rewriter.replaceOp(op, call.getResults());
    return ::mlir::success();
  }
};

struct SparkProbeToFuncPass final
    : public ::mlir::PassWrapper<SparkProbeToFuncPass,
                                  ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparkProbeToFuncPass)
  ::llvm::StringRef getArgument() const final { return "spark-probe-to-func"; }
  ::llvm::StringRef getDescription() const final {
    return "Lower spark.mlm_probe to the spark_mlm_probe_v1 native C ABI";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    auto i64 = ::mlir::IntegerType::get(context, 64);
    auto i32 = ::mlir::IntegerType::get(context, 32);
    auto expected = ::mlir::FunctionType::get(context,
                        {i64, i64, i32, i64, i64}, {i32});
    auto existing = module.lookupSymbol<::mlir::func::FuncOp>("spark_mlm_probe_v1");
    if (existing && existing.getFunctionType() != expected) {
      module.emitError("spark_mlm_probe_v1 has incompatible ABI signature");
      signalPassFailure();
      return;
    }
    if (!existing) {
      ::mlir::OpBuilder builder(context);
      builder.setInsertionPointToStart(module.getBody());
      builder.create<::mlir::func::FuncOp>(module.getLoc(),
                                           "spark_mlm_probe_v1", expected);
    }
    ::mlir::RewritePatternSet patterns(context);
    patterns.add<LowerMLMProbe>(context);
    if (::mlir::failed(::mlir::applyPatternsAndFoldGreedily(module,
                                                             std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    bool remains = false;
    module.walk([&](MLMProbeOp) { remains = true; });
    if (remains) {
      module.emitError("unlowered spark.mlm_probe remains");
      signalPassFailure();
    }
  }
};
} // namespace

void registerSparkPasses() {
  ::mlir::PassRegistration<SparkProbeToFuncPass>();
}
} // namespace spark::mlir
