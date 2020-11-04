/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include "olap/operators/relbuilder.hpp"

#include <dlfcn.h>

#include <iomanip>
#include <lib/operators/bloom-filter/bloom-filter-build.hpp>
#include <lib/operators/bloom-filter/bloom-filter-probe.hpp>
#include <lib/operators/bloom-filter/bloom-filter-repack.hpp>
#include <lib/plugins/vector/vector-plugin.hpp>
#include <lib/util/flush-operator-tree.hpp>
#include <utility>

#include "block-to-tuples.hpp"
#include "cpu-to-gpu.hpp"
#include "flush.hpp"
#include "hash-join-chained-morsel.hpp"
#include "hash-join-chained.hpp"
#include "hash-rearrange.hpp"
#include "lib/operators/gpu/gpu-hash-group-by-chained.hpp"
#include "lib/operators/gpu/gpu-hash-join-chained.hpp"
#include "lib/operators/gpu/gpu-hash-rearrange.hpp"
#include "lib/operators/gpu/gpu-reduce.hpp"
#include "lib/operators/gpu/gpu-sort.hpp"
#include "lib/operators/gpu/gpu-to-cpu.hpp"
#include "lib/operators/mem-move/mem-broadcast-device.hpp"
#include "lib/operators/mem-move/mem-broadcast-scaleout.hpp"
#include "lib/operators/mem-move/mem-move-device.hpp"
#include "lib/operators/mem-move/mem-move-scaleout.hpp"
#include "lib/operators/router/router-scaleout.hpp"
#include "lib/operators/router/router.hpp"
#include "olap/plan/catalog-parser.hpp"
#include "olap/util/parallel-context.hpp"
#include "project.hpp"
#include "reduce-opt.hpp"
#include "scan.hpp"
#include "select.hpp"
#include "sort.hpp"
#include "unionall.hpp"
#include "unnest.hpp"
#include "update.hpp"

RelBuilder::RelBuilder(ParallelContext *ctx, Operator *root)
    : ctx(ctx), root(root) {}

RelBuilder::RelBuilder(const RelBuilder &builder, Operator *root)
    : RelBuilder(builder.ctx, root) {
  if (builder.root) builder.root->setParent(root);
}

std::string RelBuilder::getModuleName() const { return ctx->getModuleName(); }

expressions::InputArgument RelBuilder::getOutputArg() const {
  return new RecordType(root->getRowType());
}

expressions::InputArgument RelBuilder::getOutputArgUnnested() const {
  auto args = root->getRowType().getArgs();
  std::vector<RecordAttribute *> attrs;
  attrs.reserve(args.size());
  for (const auto &arg : args) {
    auto block = dynamic_cast<const BlockType *>(arg->getOriginalType());
    if (block) {
      attrs.emplace_back(
          new RecordAttribute(arg->getAttrNo(), arg->getRelationName(),
                              arg->getAttrName(), &(block->getNestedType())));
    } else {
      attrs.emplace_back(arg);
    }
  }
  return new RecordType(attrs);
}

RelBuilder::RelBuilder() : RelBuilder(new ParallelContext("main", false)) {}
RelBuilder::RelBuilder(ParallelContext *ctx) : RelBuilder(ctx, nullptr) {}

const RecordType &RelBuilder::getRecordType(CatalogParser &catalog,
                                            std::string relName) {
  auto inputInfo = catalog.getInputInfo(std::move(relName));

  auto &collType = dynamic_cast<CollectionType &>(*(inputInfo->exprType));

  const ExpressionType &nestedType = collType.getNestedType();
  return dynamic_cast<const RecordType &>(nestedType);
}

void RelBuilder::setOIDType(CatalogParser &catalog, std::string relName,
                            ExpressionType *type) {
  catalog.getInputInfo(std::move(relName))->oidType = type;
}

RelBuilder RelBuilder::apply(Operator *op) const {
  // Registered op's output relation, if it's not already registered
  auto args = op->getRowType().getArgs();
  if (!args.empty()) {
    auto &catalog = CatalogParser::getInstance();
    for (const auto a : args) {
      catalog.getOrCreateInputInfo(a->getRelationName(), ctx);
    }
  }
  return {*this, op};
}

RelBuilder RelBuilder::scan(Plugin &pg) const {
  return RelBuilder{ctx, new Scan(ctx, pg)};
}

RelBuilder RelBuilder::memmove(const vector<RecordAttribute *> &wantedFields,
                               size_t slack, DeviceType to) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemMoveDevice(root, wantedFields, slack, to == DeviceType::CPU);
  return apply(op);
}

RelBuilder RelBuilder::memmove_scaleout(
    const vector<RecordAttribute *> &wantedFields, size_t slack) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemMoveScaleOut(root, wantedFields, slack);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst(const vector<RecordAttribute *> &wantedFields,
                                 DegreeOfParallelism fanout, bool to_cpu,
                                 bool always_share) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op =
      new MemBroadcastDevice(root, wantedFields, fanout, to_cpu, always_share);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst(DegreeOfParallelism fanout, bool to_cpu,
                                 bool always_share) const {
  return membrdcst(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
        std::vector<RecordAttribute *> ret;
        for (const auto &attr : arg.getProjections()) {
          if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
            ret.emplace_back(new RecordAttribute{attr});
          }
        }
        assert(!ret.empty());
        return ret;
      },
      fanout, to_cpu, always_share);
}

RelBuilder RelBuilder::membrdcst_scaleout(
    const vector<RecordAttribute *> &wantedFields, size_t fanout, bool to_cpu,
    bool always_share) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemBroadcastScaleOut(root, ctx, wantedFields, fanout, to_cpu,
                                     always_share);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst_scaleout(size_t fanout, bool to_cpu,
                                          bool always_share) const {
  return membrdcst_scaleout(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
        std::vector<RecordAttribute *> ret;
        for (const auto &attr : arg.getProjections()) {
          if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
            ret.emplace_back(new RecordAttribute{attr});
          }
        }
        assert(!ret.empty());
        return ret;
      },
      fanout, to_cpu, always_share);
}

RelBuilder RelBuilder::memmove(size_t slack, DeviceType to) const {
  std::vector<RecordAttribute *> ret;
  for (const auto &attr : getOutputArg().getProjections()) {
    if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
      ret.emplace_back(new RecordAttribute{attr});
    }
  }
  assert(!ret.empty());
  return memmove(ret, slack, to);
}

RelBuilder RelBuilder::memmove_scaleout(const MultiAttributeFactory &attr,
                                        size_t slack) const {
  return memmove_scaleout(attr(getOutputArg()), slack);
}

RelBuilder RelBuilder::memmove_scaleout(size_t slack) const {
  return memmove_scaleout(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
        std::vector<RecordAttribute *> ret;
        for (const auto &attr : arg.getProjections()) {
          if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
            ret.emplace_back(new RecordAttribute{attr});
          }
        }
        assert(!ret.empty());
        return ret;
      },
      slack);
}

RelBuilder RelBuilder::to_gpu() const {
  const auto arg = getOutputArg();
  std::vector<RecordAttribute *> ret;
  for (const auto &attr : arg.getProjections()) {
    if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
      ret.emplace_back(new RecordAttribute{attr});
    }
  }
  assert(!ret.empty());
  return to_gpu(ret);
}

RelBuilder RelBuilder::to_cpu(gran_t granularity, size_t size) const {
  const auto arg = getOutputArg();
  std::vector<RecordAttribute *> ret;
  for (const auto &attr : arg.getProjections()) {
    if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
      ret.emplace_back(new RecordAttribute{attr});
    }
  }
  if (ret.empty()) {
    for (const auto &attr : arg.getProjections()) {
      ret.emplace_back(new RecordAttribute{attr});
    }
  }
  return to_cpu(ret, granularity, size);
}

RelBuilder RelBuilder::to_gpu(
    const vector<RecordAttribute *> &wantedFields) const {
  auto op = new CpuToGpu(root, wantedFields);
  return apply(op);
}

RelBuilder RelBuilder::to_cpu(const vector<RecordAttribute *> &wantedFields,
                              gran_t granularity, size_t size) const {
  auto op = new GpuToCpu(root, wantedFields, size, granularity);
  return apply(op);
}

RelBuilder RelBuilder::filter(expression_t pred) const {
  auto op = new Select(std::move(pred), root);
  return apply(op);
}

RelBuilder RelBuilder::project(const std::vector<expression_t> &proj) const {
  assert(!proj.empty());
  auto op = new Project(proj, proj[0].getRegisteredRelName(), root, ctx);
  return apply(op);
}

RelBuilder RelBuilder::unpack(const vector<expression_t> &projections) const {
  return unpack(projections, (root->getDeviceType() == DeviceType::GPU)
                                 ? gran_t::GRID
                                 : gran_t::THREAD);
}

RelBuilder RelBuilder::unpack(const vector<expression_t> &projections,
                              gran_t granularity) const {
  auto op = new BlockToTuples(
      root, projections, root->getDeviceType() == DeviceType::GPU, granularity);
  return apply(op);
}

RelBuilder RelBuilder::pack(const vector<expression_t> &projections,
                            expression_t hashExpr, size_t numOfBuckets) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new GpuHashRearrange(root, ctx, numOfBuckets, projections,
                                   std::move(hashExpr));
    return apply(op);
  } else {
    auto op =
        new HashRearrange(root, numOfBuckets, projections, std::move(hashExpr));
    return apply(op);
  }
}

RelBuilder RelBuilder::reduce(const vector<expression_t> &e,
                              const vector<Monoid> &accs) const {
  assert(e.size() == accs.size());
  std::vector<agg_t> aggs;
  aggs.reserve(e.size());
  for (size_t i = 0; i < e.size(); ++i) {
    aggs.emplace_back(e[i], accs[i]);
  }
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new opt::GpuReduce(std::move(aggs), true, root);
    return apply(op);
  } else {
    auto op = new opt::Reduce(std::move(aggs), true, root);
    return apply(op);
  }
}

RelBuilder RelBuilder::groupby(const std::vector<expression_t> &e,
                               const std::vector<GpuAggrMatExpr> &agg_exprs,
                               size_t hash_bits, size_t maxInputSize) const {
  switch (root->getDeviceType()) {
    case DeviceType::GPU: {
      auto op = new GpuHashGroupByChained(agg_exprs, e, root, hash_bits,
                                          maxInputSize);
      return apply(op);
    }
    case DeviceType::CPU: {
      auto op =
          new HashGroupByChained(agg_exprs, e, root, hash_bits, maxInputSize);
      return apply(op);
    }
  }
}

RelBuilder RelBuilder::sort(const vector<expression_t> &orderByFields,
                            const vector<direction> &dirs) const {
  switch (root->getDeviceType()) {
    case DeviceType::GPU: {
      auto op = new GpuSort(root, ctx, orderByFields, dirs);
      return apply(op);
    }
    case DeviceType::CPU: {
      auto op = new Sort(root, ctx, orderByFields, dirs);
      return apply(op)
          .unpack([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["__sorted"]};
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            std::vector<expression_t> attrs;
            for (const auto attr : dynamic_cast<const RecordType *>(
                                       arg["__sorted"].getExpressionType())
                                       ->getArgs()) {
              attrs.emplace_back(expression_t{arg["__sorted"]}[*attr]);
            }
            return attrs;
          });
    }
  }
}

RelBuilder RelBuilder::print(const vector<expression_t> &e, Plugin *pg,
                             bool may_overwrite) const {
  assert(!e.empty() && "Empty print");
  assert(e[0].isRegistered());
  std::string outrel = e[0].getRegisteredRelName();
  CatalogParser &catalog = CatalogParser::getInstance();
  LOG(INFO) << outrel;
  assert(may_overwrite || !catalog.getInputInfoIfKnown(outrel));

  std::vector<RecordAttribute *> args;
  args.reserve(e.size());

  for (const auto &e_e : e) {
    args.emplace_back(new RecordAttribute{e_e.getRegisteredAs()});
  }

  InputInfo *datasetInfo;
  if (pg == nullptr) {
    datasetInfo = catalog.getOrCreateInputInfo(outrel, ctx);
  } else {
    datasetInfo = new InputInfo();
    datasetInfo->path = outrel;
    assert(
        ctx &&
        "A ParallelContext is required to register relationships on the fly");

    catalog.setInputInfo(outrel, datasetInfo);
    setOIDType(catalog, outrel, pg->getOIDType());
  }

  datasetInfo->exprType =
      new BagType{RecordType{std::vector<RecordAttribute *>{args}}};

  auto op = new Flush(e, root, outrel);
  return apply(op);
}

RelBuilder RelBuilder::unnest(expression_t e) const {
  auto op = new Unnest(true, std::move(e), root);
  return apply(op);
}

PreparedStatement RelBuilder::prepare() {
  root->produce(ctx);
  ctx->prepareFunction(ctx->getGlobalFunction());
  ctx->compileAndLoad();
  auto p = dynamic_cast<Flush *>(root);
  std::string outputFile = (p) ? p->getOutputPath() : ctx->getModuleName();
  return {ctx->getPipelines(), outputFile};
}

RelBuilder RelBuilder::router(const vector<RecordAttribute *> &wantedFields,
                              std::optional<expression_t> hash,
                              DegreeOfParallelism fanout, size_t slack,
                              RoutingPolicy p, DeviceType target,
                              std::unique_ptr<Affinitizer> aff) const {
  if (aff) {
    auto op = new Router(root, fanout, wantedFields, slack, std::move(hash), p,
                         std::move(aff));
    return apply(op);
  } else {
    auto op = new Router(root, fanout, wantedFields, slack, std::move(hash), p,
                         target);
    return apply(op);
  }
}

RelBuilder RelBuilder::router_scaleout(
    const vector<RecordAttribute *> &wantedFields,
    std::optional<expression_t> hash, DegreeOfParallelism fanout, size_t slack,
    RoutingPolicy p, DeviceType targets) const {
  assert((p == RoutingPolicy::HASH_BASED) == (hash.has_value()));
  assert((p != RoutingPolicy::RANDOM) || (!hash.has_value()));
  auto op =
      new RouterScaleOut(root, DegreeOfParallelism{fanout}, wantedFields, slack,
                         std::move(hash), p, targets, root->getDOPServers());
  return apply(op);
}

RelBuilder RelBuilder::router_scaleout(const MultiAttributeFactory &attr,
                                       const OptionalExpressionFactory &hash,
                                       DegreeOfParallelism fanout, size_t slack,
                                       RoutingPolicy p,
                                       DeviceType targets) const {
  return router_scaleout(attr(getOutputArg()), hash(getOutputArg()), fanout,
                         slack, p, targets);
}

RelBuilder RelBuilder::router_scaleout(const OptionalExpressionFactory &hash,
                                       DegreeOfParallelism fanout, size_t slack,
                                       RoutingPolicy p,
                                       DeviceType target) const {
  return router_scaleout(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
        std::vector<RecordAttribute *> attrs;
        for (const auto &attr : arg.getProjections()) {
          if (p == RoutingPolicy::HASH_BASED &&
              attr.getAttrName() == "__broadcastTarget") {
            continue;
          }
          attrs.emplace_back(new RecordAttribute{attr});
        }
        return attrs;
      },
      hash, fanout, slack, p, target);
}

RelBuilder RelBuilder::router_scaleout(DegreeOfParallelism fanout, size_t slack,
                                       RoutingPolicy p,
                                       DeviceType target) const {
  assert(p != RoutingPolicy::HASH_BASED);
  return router_scaleout(
      [&](const auto &arg) -> std::optional<expression_t> {
        return std::nullopt;
      },
      fanout, slack, p, target);
}

RelBuilder RelBuilder::router(DegreeOfParallelism fanout, size_t slack,
                              RoutingPolicy p, DeviceType target,
                              std::unique_ptr<Affinitizer> aff) const {
  assert(p != RoutingPolicy::HASH_BASED);
  return router(
      [&](const auto &arg) -> std::optional<expression_t> {
        return std::nullopt;
      },
      fanout, slack, p, target, std::move(aff));
}

RelBuilder RelBuilder::router(size_t slack, RoutingPolicy p, DeviceType target,
                              std::unique_ptr<Affinitizer> aff) const {
  size_t dop = (target == DeviceType::CPU)
                   ? topology::getInstance().getCoreCount()
                   : topology::getInstance().getGpuCount();
  return router(DegreeOfParallelism{dop}, slack, p, target, std::move(aff));
}

class HintRowCount : public experimental::UnaryOperator {
 public:
  const double expected;

  HintRowCount(Operator *op, double expected)
      : UnaryOperator(op), expected(expected) {}

  [[nodiscard]] RecordType getRowType() const override {
    return getChild()->getRowType();
  }
  [[nodiscard]] DegreeOfParallelism getDOP() const override {
    return getChild()->getDOP();
  }
  void produce_(ParallelContext *context) override {
    return getChild()->produce(context);
  }
  void consume(ParallelContext *context,
               const OperatorState &binding) override {
    getParent()->consume(context, binding);
  }
  [[nodiscard]] bool isFiltering() const override { return false; }
};

[[nodiscard]] RelBuilder RelBuilder::hintRowCount(
    double expectedRowCount) const {
  auto op = new HintRowCount(root, expectedRowCount);
  return apply(op);
}

using v_t = std::variant<Router *, MemBroadcastDevice *, MemMoveDevice *,
                         BlockToTuples *, HintRowCount *, Select *, Project *>;
double expected(Operator *op);

class ExpectedTuplesOutputSize {
 public:
  double operator()(Router *op) const {
    return expected(op->getChild()) * op->getChild()->getDOP() / op->getDOP();
  }
  double operator()(MemBroadcastDevice *op) const {
    return expected(op->getChild()) * op->getNumberOfTargets();
  }
  double operator()(MemMoveDevice *op) const {
    return expected(op->getChild());
  }
  double operator()(BlockToTuples *op) const {
    return expected(op->getChild());
  }
  double operator()(HintRowCount *op) const { return op->expected; }
  double operator()(Select *op) const {
    double sel = 1;
    if (auto hint = dynamic_cast<const expressions::HintExpression *>(
            op->getFilter().getUnderlyingExpression())) {
      sel = hint->getSelectivity().sel;
    }

    return expected(op->getChild()) * sel;
  }
  double operator()(Project *op) const { return expected(op->getChild()); }
};

template <typename VT, size_t index = 0>
struct vinit {
  VT get(Operator *op) {
    if constexpr (index >= std::variant_size_v<VT>) {
      throw proteus::unsupported_operation{std::string{"unsupported type "} +
                                           typeid(*op).name()};
    } else {
      if (auto sop = dynamic_cast<std::variant_alternative_t<index, VT>>(op)) {
        return VT{sop};
      }
      return vinit<VT, index + 1>{}.get(op);
    }
  }
};

double expected(Operator *op) {
  auto obj = vinit<v_t>{}.get(op);
  return std::visit(ExpectedTuplesOutputSize{}, obj);
}

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            expression_t probe_k) const {
  size_t bsize = expected(build.root) * 1.1;  // 10% overestimate for safety

  return join(build, std::move(build_k), std::move(probe_k),
              static_cast<int>(std::ceil(std::log2(bsize))) + 1, bsize);
}

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            expression_t probe_k, int hash_bits,
                            size_t maxBuildInputSize) const {
  auto &llvmContext = ctx->getLLVMContext();
  std::vector<size_t> build_w;
  std::vector<GpuMatExpr> build_e;
  build_w.emplace_back(
      32 +
      ctx->getSizeOf(build_k.getExpressionType()->getLLVMType(llvmContext)) *
          8);
  size_t ind = 1;
  auto build_arg = build.getOutputArg();
  for (const auto &p : build_arg.getProjections()) {
    auto relName = build_k.getRegisteredRelName();
    auto e = build_arg[p];

    if (build_k.isRegistered() &&
        build_k.getRegisteredAs() == e.getRegisteredAs())
      continue;

    build_e.emplace_back(e, ind++, 0);
    build_w.emplace_back(
        ctx->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)) * 8);
  }
  std::vector<size_t> probe_w;
  std::vector<GpuMatExpr> probe_e;
  probe_w.emplace_back(
      32 +
      ctx->getSizeOf(probe_k.getExpressionType()->getLLVMType(llvmContext)) *
          8);
  ind = 1;
  auto probe_arg = getOutputArg();
  for (const auto &p : probe_arg.getProjections()) {
    auto relName = probe_k.getRegisteredRelName();
    auto e = probe_arg[p];

    if (probe_k.isRegistered() &&
        probe_k.getRegisteredAs() == e.getRegisteredAs())
      continue;

    probe_e.emplace_back(e, ind++, 0);
    probe_w.emplace_back(
        ctx->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)) * 8);
  }

  return join(build, std::move(build_k), build_e, build_w, std::move(probe_k),
              probe_e, probe_w, hash_bits, maxBuildInputSize);
}

RelBuilder RelBuilder::morsel_join(RelBuilder build, expression_t build_k,
                                   expression_t probe_k, int hash_bits,
                                   size_t maxBuildInputSize) const {
  auto &llvmContext = ctx->getLLVMContext();
  std::vector<size_t> build_w;
  std::vector<GpuMatExpr> build_e;
  build_w.emplace_back(
      32 +
      ctx->getSizeOf(build_k.getExpressionType()->getLLVMType(llvmContext)) *
          8);
  size_t ind = 1;
  auto build_arg = build.getOutputArg();
  for (const auto &p : build_arg.getProjections()) {
    auto relName = build_k.getRegisteredRelName();
    auto e = build_arg[p];

    if (build_k.isRegistered() &&
        build_k.getRegisteredAs() == e.getRegisteredAs())
      continue;

    auto rc = dynamic_cast<const expressions::RecordConstruction *>(
        build_k.getUnderlyingExpression());
    if (rc && dynamic_cast<const RecordType *>(rc->getExpressionType())
                  ->getArg(p.getAttrName()))
      continue;
    LOG(INFO) << p;

    build_e.emplace_back(e, ind++, 0);
    build_w.emplace_back(
        ctx->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)) * 8);
  }
  std::vector<size_t> probe_w;
  std::vector<GpuMatExpr> probe_e;
  probe_w.emplace_back(
      32 +
      ctx->getSizeOf(probe_k.getExpressionType()->getLLVMType(llvmContext)) *
          8);
  ind = 1;
  auto probe_arg = getOutputArg();
  for (const auto &p : probe_arg.getProjections()) {
    auto relName = probe_k.getRegisteredRelName();
    auto e = probe_arg[p];

    if (probe_k.isRegistered() &&
        probe_k.getRegisteredAs() == e.getRegisteredAs())
      continue;

    probe_e.emplace_back(e, ind++, 0);
    probe_w.emplace_back(
        ctx->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)) * 8);
  }

  return morsel_join(build, std::move(build_k), build_e, build_w,
                     std::move(probe_k), probe_e, probe_w, hash_bits,
                     maxBuildInputSize);
}

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            const std::vector<GpuMatExpr> &build_e,
                            const std::vector<size_t> &build_w,
                            expression_t probe_k,
                            const std::vector<GpuMatExpr> &probe_e,
                            const std::vector<size_t> &probe_w, int hash_bits,
                            size_t maxBuildInputSize) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new GpuHashJoinChained(
        build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
        std::move(probe_k), root, hash_bits, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  } else {
    auto op = new HashJoinChained(
        build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
        std::move(probe_k), root, hash_bits, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  }
}

RelBuilder RelBuilder::morsel_join(RelBuilder build, expression_t build_k,
                                   const std::vector<GpuMatExpr> &build_e,
                                   const std::vector<size_t> &build_w,
                                   expression_t probe_k,
                                   const std::vector<GpuMatExpr> &probe_e,
                                   const std::vector<size_t> &probe_w,
                                   int hash_bits,
                                   size_t maxBuildInputSize) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new GpuHashJoinChained(
        build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
        std::move(probe_k), root, hash_bits, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  } else {
    assert(this->root->getDOP() == build->getDOP());
    auto op = new HashJoinChainedMorsel(
        build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
        std::move(probe_k), root, hash_bits, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  }
}

void RelBuilder::registerPlugin(const std::string &relName, Plugin *pg) {
  Catalog::getInstance().registerPlugin(relName, pg);
}

typedef Plugin *(*plugin_creator_t)(ParallelContext *, std::string, RecordType,
                                    const std::vector<RecordAttribute *> &);

std::string hyphenatedPluginToCamel(const std::string &line);

auto getPluginFactory(const std::string &pgType) {
  auto name = hyphenatedPluginToCamel(pgType);
  std::string conv = "create" + name + "Plugin";

  LOG(INFO) << "PluginName: " << name;

  static auto handle = dlopen(nullptr, 0);

  auto create = (plugin_creator_t)dlsym(handle, conv.c_str());

  if (!create) {
    string err = string("Unknown Plugin Type: ") + pgType;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  return create;
}

Plugin *RelBuilder::createPlugin(const RecordType &rec,
                                 const std::vector<RecordAttribute *> &projs,
                                 const std::string &pgType) const {
  auto create = getPluginFactory(pgType);

  assert(!rec.getArgs().empty());
  auto fileName = rec.getArgs().front()->getRelationName();
  LOG(INFO) << fileName << " " << pgType;
  auto pg = create(ctx, fileName, rec, projs);

  auto &catalog = CatalogParser::getInstance();
  auto ii = catalog.getInputInfoIfKnown(fileName);
  if (!ii) {
    ii = new InputInfo;
    ii->exprType = new BagType(rec);
    ii->path = fileName;

    catalog.setInputInfo(fileName, ii);
  }

  ii->oidType = new RecordType(rec);

  Catalog::getInstance().registerPlugin(fileName, pg);

  return pg;
}

RelBuilder RelBuilder::scan(
    const std::vector<
        std::pair<RecordAttribute *, std::shared_ptr<proteus_any_vector>>>
        &data) const {
  auto pg = new VectorPlugin(ctx, data);

  auto fileName = data[0].first->getRelationName();
  auto &catalog = CatalogParser::getInstance();
  auto ii = catalog.getInputInfoIfKnown(fileName);
  if (!ii) {
    ii = new InputInfo;
    ii->exprType = new BagType(pg->getRowType());
    ii->path = fileName;

    catalog.setInputInfo(fileName, ii);
  }

  ii->oidType = new RecordType(pg->getRowType());

  Catalog::getInstance().registerPlugin(fileName, pg);
  return scan(*pg);
}

RelBuilder RelBuilder::scan(std::string relName,
                            const std::vector<std::string> &relAttrs,
                            CatalogParser &catalog, const pg &pgType) const {
  const RecordType &recType_ = getRecordType(catalog, std::move(relName));
  return scan(recType_, relAttrs, pgType.getType());
}

RelBuilder RelBuilder::scan(const RecordType &rec,
                            const std::vector<std::string> &relAttrs,
                            const std::string &pgType) const {
  std::vector<RecordAttribute *> projs;
  projs.reserve(relAttrs.size());
  for (const auto &attr : relAttrs) {
    projs.emplace_back(new RecordAttribute(*rec.getArg(attr)));
  }

  auto pg = createPlugin(rec, projs, pgType);
  return scan(*pg);
}

RelBuilder RelBuilder::print(pg pgType, std::string outrel) const {
  return print(
      [&](const auto &arg) {
        auto attrs = arg.getProjections();
        std::vector<expression_t> es;
        es.reserve(attrs.size());
        for (const auto &attr : attrs) {
          es.emplace_back(
              arg[attr.getAttrName()].as(outrel, attr.getAttrName()));
        }
        return es;
      },
      std::move(pgType), outrel);
}

RelBuilder RelBuilder::print(pg pgType) const {
  return print(std::move(pgType), getModuleName());
}

RelBuilder RelBuilder::print(
    std::function<std::vector<expression_t>(const expressions::InputArgument &)>
        exprs,
    pg pgType) const {
  return print(std::move(exprs), std::move(pgType), getModuleName());
}

RelBuilder RelBuilder::print(
    std::function<std::vector<expression_t>(const expressions::InputArgument &)>
        exprs,
    pg pgType, std::string outrel) const {
  LOG(INFO) << "registering";
  auto arg = getOutputArg();

  const auto &projs = arg.getProjections();

  std::vector<RecordAttribute *> attrs;
  attrs.reserve(projs.size());
  for (const auto &a : projs) {
    attrs.emplace_back(new RecordAttribute(
        attrs.size() + 1, outrel, a.getAttrName(), a.getOriginalType()));
  }

  auto pg = createPlugin({attrs}, {}, pgType.getType());

  auto v = exprs(arg);
  for (auto &e : v) e.as(outrel, e.getRegisteredAttrName());

  return print(v, pg, true);
}

RelBuilder RelBuilder::print(
    std::function<std::vector<expression_t>(const expressions::InputArgument &)>
        exprs) const {
  return print(std::move(exprs), pg("pm-csv"));
}

RelBuilder RelBuilder::split(size_t alternatives, size_t slack, RoutingPolicy p,
                             DeviceType target,
                             std::unique_ptr<Affinitizer> aff) const {
  // FIXME: implement!!!!
  throw proteus::unsupported_operation("unimplemented");
}

RelBuilder RelBuilder::unionAll(const std::vector<RelBuilder> &children) const {
  std::vector<RecordAttribute *> projections;
  for (const auto &attr : getOutputArg().getProjections()) {
    projections.emplace_back(new RecordAttribute{attr});
  }

  return unionAll(children, projections);
}

RelBuilder RelBuilder::unionAll(
    const std::vector<RelBuilder> &children,
    const vector<RecordAttribute *> &wantedFields) const {
  std::vector<Operator *> c2{root};
  c2.reserve(children.size() + 1);
  for (const auto &c : children) c2.emplace_back(c.root);
  auto op = new UnionAll(c2, wantedFields);
  for (const auto &c : children) c.apply(op);
  return apply(op);
}

RelBuilder RelBuilder::bloomfilter_probe(expression_t pred, size_t filterSize,
                                         uint64_t bloomId) const {
  auto op = new BloomFilterProbe(root, std::move(pred), filterSize, bloomId);
  return apply(op);
}

RelBuilder RelBuilder::bloomfilter_build(expression_t pred, size_t filterSize,
                                         uint64_t bloomId) const {
  auto op = new BloomFilterBuild(root, std::move(pred), filterSize, bloomId);
  return apply(op);
}

RelBuilder RelBuilder::bloomfilter_repack(expression_t pred,
                                          std::vector<expression_t> attr,
                                          size_t filterSize,
                                          uint64_t bloomId) const {
  auto op = new BloomFilterRepack(root, std::move(pred), std::move(attr),
                                  filterSize, bloomId);
  return apply(op);
}

RelBuilder RelBuilder::update(
    const std::function<expression_t(const expressions::InputArgument &)> &expr)
    const {
  return update(expr(getOutputArg()));
}

RelBuilder RelBuilder::update(expression_t e) const {
  auto op = new Update(root, std::move(e));
  return apply(op);
}

RelBuilder RelBuilder::unpack() const {
  return unpack([&](const auto &arg) -> std::vector<expression_t> {
    std::vector<expression_t> attrs;
    for (const auto &attr : arg.getProjections()) {
      attrs.emplace_back(arg[attr]);
    }
    return attrs;
  });
}

RelBuilder RelBuilder::pack() const {
  return pack(
      [&](const auto &arg) -> std::vector<expression_t> {
        std::vector<expression_t> attrs;
        for (const auto &attr : arg.getProjections()) {
          attrs.emplace_back(arg[attr]);
        }
        return attrs;
      },
      [](const auto &arg) { return expression_t{0}; }, 1);
}

std::ostream &operator<<(std::ostream &out, const RelBuilder &builder) {
  return out << *(builder.operator->());
}
