/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include <gflags/gflags.h>

#include <dlfcn.h>

#include <iomanip>
#include <optional>
#include <lib/operators/bloom-filter/bloom-filter-build.hpp>
#include <lib/operators/bloom-filter/bloom-filter-probe.hpp>
#include <lib/operators/bloom-filter/bloom-filter-repack.hpp>
#include <lib/operators/router/generalized-router.hpp>
#include <lib/plugins/vector/vector-plugin.hpp>
#include <lib/util/flush-operator-tree.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <utility>

#include "block-to-tuples.hpp"
#include "lib/operators/gpu/compressed-block-to-tuples.hpp"
#include "pre-decompress.hpp"
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
#include "split.hpp"
#include "unionall.hpp"
#include "unnest.hpp"
#include "update.hpp"

DECLARE_bool(decom_op);
DECLARE_bool(decom_reg);
DECLARE_bool(precompute_rl);
DECLARE_bool(pre_decom);
DECLARE_bool(use_metadata_in_block);
DECLARE_string(compression_type);

RelBuilder::RelBuilder(OlapParallelContext *ctx, Operator *root)
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

RelBuilder::RelBuilder() : RelBuilder(new OlapParallelContext("main", false)) {}
RelBuilder::RelBuilder(OlapParallelContext *ctx) : RelBuilder(ctx, nullptr) {}

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

Operator *RelBuilder::registerOutput(Operator *op) const {
  // Registered op's output relation, if it's not already registered
  auto args = op->getRowType().getArgs();
  if (!args.empty()) {
    auto &catalog = CatalogParser::getInstance();
    for (const auto a : args) {
      catalog.getOrCreateInputInfo(a->getRelationName(), ctx);
    }
  }
  return op;
}

RelBuilder RelBuilder::apply(Operator *op) const {
  return {*this, registerOutput(op)};
}

RelBuilder RelBuilder::scan(Plugin &pg) const {
  return RelBuilder{ctx, new Scan(pg)};
}

RelBuilder RelBuilder::memmove(
    const std::vector<RecordAttribute *> &wantedFields, size_t slack,
    DeviceType to) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemMoveDevice(root, wantedFields, slack, to == DeviceType::CPU);
  return apply(op);
}

RelBuilder RelBuilder::memmove_scaleout(
    const std::vector<RecordAttribute *> &wantedFields, size_t slack) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemMoveScaleOut(root, wantedFields, slack);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst(
    const std::vector<RecordAttribute *> &wantedFields,
    DegreeOfParallelism fanout, bool to_cpu, bool always_share) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op =
      new MemBroadcastDevice(root, wantedFields, fanout, to_cpu, always_share);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst(DegreeOfParallelism fanout, DeviceType target,
                                 bool always_share) const {
  return membrdcst(fanout, target == DeviceType::CPU, always_share);
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

RelBuilder RelBuilder::membrdcst(DeviceType target, bool always_share) const {
  return membrdcst(getDefaultDOP(target), target == DeviceType::CPU,
                   always_share);
}

RelBuilder RelBuilder::membrdcst_scaleout(
    const std::vector<RecordAttribute *> &wantedFields, size_t fanout,
    bool to_cpu, bool always_share) const {
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

RelBuilder RelBuilder::memmove(const std::vector<RecordAttribute *> &wantedFields,
                               size_t slack, DeviceType to,
                               const std::vector<bool> &do_transfer) const {
  for (const auto &attr : wantedFields) {
    assert(dynamic_cast<const BlockType *>(attr->getOriginalType()));
  }
  auto op = new MemMoveDevice(root, wantedFields, slack, to == DeviceType::CPU,
                              do_transfer);
  return apply(op);
}

RelBuilder RelBuilder::memmove(
    size_t slack, DeviceType to,
    std::optional<std::vector<bool>> do_transfer) const {
  assert(root->isPacked() && "MemMove is not applicable to unpacked inputs");
  std::vector<RecordAttribute *> ret;
  for (const auto &attr : getOutputArg().getProjections()) {
    if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
      ret.emplace_back(new RecordAttribute{attr});
    }
  }
  DCHECK(!ret.empty()) << "No attributes to move!";
  std::vector<bool> do_transfer_vec;
  if (do_transfer.has_value()) {
    do_transfer_vec = do_transfer.value();
    DCHECK_EQ(do_transfer->size(), ret.size())
        << "Invalid do_transfer vector! do_transfer.size() must be equal to "
           "the number of attributes to transfer!";
    if (to == DeviceType::CPU) {
      LOG(WARNING) << "do_transfer provided for CPU target. Ignoring it!";
    }
  } else {
    do_transfer_vec = std::vector<bool>(ret.size(), true);
  }
  return memmove(ret, slack, to, do_transfer_vec);
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
    //    if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
    ret.emplace_back(new RecordAttribute{attr});
    //    }
  }
  //  if (ret.empty()) {
  //    for (const auto &attr : arg.getProjections()) {
  //      ret.emplace_back(new RecordAttribute{attr});
  //    }
  //  }
  return to_cpu(ret, granularity, size);
}

RelBuilder RelBuilder::to_gpu(
    const std::vector<RecordAttribute *> &wantedFields) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    return *this;
  }
  auto op = new CpuToGpu(root, wantedFields);
  return apply(op);
}

RelBuilder RelBuilder::to_cpu(
    const std::vector<RecordAttribute *> &wantedFields, gran_t granularity,
    size_t size) const {
  if (root->getDeviceType() == DeviceType::CPU) {
    return *this;
  }
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

RelBuilder RelBuilder::unpack(
    const std::vector<expression_t> &projections) const {
  return unpack(projections, (root->getDeviceType() == DeviceType::GPU)
                                 ? gran_t::GRID
                                 : gran_t::THREAD);
}

RelBuilder RelBuilder::unpack(const std::vector<expression_t> &projections,
                              gran_t granularity) const {
  CHECK(!(FLAGS_decom_op && FLAGS_decom_reg))
      << "--decom_op and --decom_reg are mutually exclusive";
  CHECK(!(FLAGS_precompute_rl && !FLAGS_decom_reg))
      << "--precompute_rl requires --decom_reg";
  CHECK(!(FLAGS_precompute_rl && FLAGS_compression_type != "RFOR"))
      << "--precompute_rl currently supports only --compression_type=RFOR";
  CHECK(!(FLAGS_decom_op && FLAGS_pre_decom))
      << "--decom_op and --pre_decom are mutually exclusive";
  CHECK(!(FLAGS_decom_reg && FLAGS_pre_decom))
      << "--decom_reg and --pre_decom are mutually exclusive";
  CHECK(!(FLAGS_pre_decom && FLAGS_compression_type == "NONE"))
      << "--pre_decom is incompatible with --compression_type=NONE";
  CHECK(!(FLAGS_pre_decom && FLAGS_compression_type != "RFOR" &&
          FLAGS_compression_type != "FLBP"))
      << "--pre_decom currently supports only --compression_type=RFOR/FLBP";
  CHECK(!(FLAGS_decom_reg && FLAGS_compression_type != "RFOR" &&
          FLAGS_compression_type != "FLBP"))
      << "--decom_reg currently supports only --compression_type=RFOR/FLBP";

  const bool gpu_unpack = (root->getDeviceType() == DeviceType::GPU);
  const bool use_decom_reg_for_this_unpack = FLAGS_decom_reg && gpu_unpack;

  bool scan_emits_compressed_blocks = false;
  if (gpu_unpack) {
    for (const auto &proj : projections) {
      const auto *block_type =
          dynamic_cast<const BlockType *>(proj.getExpressionType());
      if (!block_type) continue;
      auto *pg = Catalog::getInstance().getPlugin(proj.getRegisteredRelName());
      if (auto *nvme_pg = dynamic_cast<NvmePlugin *>(pg)) {
        scan_emits_compressed_blocks =
            scan_emits_compressed_blocks ||
            nvme_pg->scanEmitsCompressedBlocks();
      }
    }
  }

  const bool supports_gpu_compressed_unpack = FLAGS_compression_type != "NONE";
  const bool can_defer_rfor_decode =
      FLAGS_compression_type == "RFOR" &&
      (scan_emits_compressed_blocks || FLAGS_use_metadata_in_block);
  const bool can_defer_flbp_decode =
      FLAGS_compression_type == "FLBP" &&
      (scan_emits_compressed_blocks || FLAGS_use_metadata_in_block);
  const bool can_defer_codec_decode =
      can_defer_rfor_decode || can_defer_flbp_decode;
  CHECK(!(use_decom_reg_for_this_unpack && !can_defer_codec_decode))
      << "--decom_reg requires the deferred decode "
         "path (compressed scan blocks or --use_metadata_in_block=true)";

  if (gpu_unpack && scan_emits_compressed_blocks &&
      supports_gpu_compressed_unpack && FLAGS_pre_decom) {
    auto predecompressed =
        apply(new PreDecompress(root, projections, true, granularity));
    return predecompressed.apply(
        new BlockToTuples(predecompressed.root, projections, true, granularity));
  }

  if (gpu_unpack && scan_emits_compressed_blocks &&
      supports_gpu_compressed_unpack) {
    if (use_decom_reg_for_this_unpack) {
      return apply(new CompressedBlockToTuplesReg(
          root, projections, true, granularity));
    }
    // Canonical semantics:
    // - --decom_op=false: materialize decoded tiles to GPU global memory.
    // - --decom_op=true : keep deferred decode path when supported.
    if (!FLAGS_decom_op) {
      return apply(new CompressedBlockToTuplesMaterialize(
          root, projections, true, granularity));
    }

    // Deferred mode requires codec metadata path support.
    if (can_defer_codec_decode) {
      return apply(new CompressedBlockToTuplesDeferred(
          root, projections, true, granularity));
    }

    // If deferred decode is not available, fall back to materialize so plan
    // remains executable.
    return apply(new CompressedBlockToTuplesMaterialize(
        root, projections, true, granularity));
  }

  // Out-of-GPU deferred path: MemMove can stage compressed blocks and
  // CompressedBlockToTuples decodes them on GPU even when scan does not emit
  // preloaded compressed pointers.
  if (gpu_unpack && supports_gpu_compressed_unpack && can_defer_codec_decode) {
    if (use_decom_reg_for_this_unpack) {
      return apply(new CompressedBlockToTuplesReg(
          root, projections, true, granularity));
    }
    if (FLAGS_decom_op) {
      return apply(new CompressedBlockToTuplesDeferred(
          root, projections, true, granularity));
    }
  }

  return apply(new BlockToTuples(root, projections, gpu_unpack, granularity));
}

RelBuilder RelBuilder::pack(const std::vector<expression_t> &projections,
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

RelBuilder RelBuilder::reduce(const std::vector<expression_t> &e,
                              const std::vector<Monoid> &accs) const {
  CHECK_EQ(e.size(), accs.size());
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

RelBuilder RelBuilder::sort(const std::vector<expression_t> &orderByFields,
                            const std::vector<direction> &dirs) const {
  if (isPacked()) {
    std::string error = "Cannot sort packed input";
    LOG(ERROR) << error;
    throw runtime_error(error);
  }
  if (orderByFields.size() != dirs.size()) {
    std::string error =
        "Number of fields must equal number of directions. Num fields: " +
        std::to_string(orderByFields.size()) +
        ", num sort directions: " + std::to_string(dirs.size());
    LOG(ERROR) << error;
    throw runtime_error(error);
  }
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

RelBuilder RelBuilder::print(const std::vector<expression_t> &e, Plugin *pg,
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
    assert(ctx &&
           "A OlapParallelContext is required to register relationships on the "
           "fly");

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

PreparedStatement RelBuilder::prepare(
    std::shared_ptr<ParallelContext> compiled_context) {
  event_range<range_log_op::BUILDER_PREPARE> er{root->getUUID()};
  LOG(INFO) << '\n' << *this;
  root->produce(ctx);
  ctx->prepareFunction(ctx->getGlobalFunction());
  ctx->compileAndLoad();
  auto p = dynamic_cast<Flush *>(root);
  std::string outputFile = (p) ? p->getOutputPath() : ctx->getModuleName();
  auto pipelines = compiled_context ? ctx->getPipelinesRetained()
                                    : ctx->getPipelines();
  return {std::move(pipelines), outputFile,
          // FIXME: we have no lifetime specs for ops, so here we leak
          //  root on purpose. We should fix the issue on its root.
          std::shared_ptr<Operator>(std::shared_ptr<Operator>{}, root),
          std::move(compiled_context)};
}

RelBuilder RelBuilder::router(
    const std::vector<RecordAttribute *> &wantedFields,
    std::optional<expression_t> hash, DegreeOfParallelism fanout, size_t slack,
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
    const std::vector<RecordAttribute *> &wantedFields,
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

RelBuilder RelBuilder::router_scaleout(const ExpressionFactory &hash,
                                       DegreeOfParallelism fanout, size_t slack,
                                       DeviceType target) const {
  return router_scaleout(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
        std::vector<RecordAttribute *> attrs;
        for (const auto &attr : arg.getProjections()) {
          if (attr.getAttrName() == "__broadcastTarget") {
            continue;
          }
          attrs.emplace_back(new RecordAttribute{attr});
        }
        return attrs;
      },
      [&](const auto &arg) -> std::optional<expression_t> { return hash(arg); },
      fanout, slack, RoutingPolicy::HASH_BASED, target);
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
      [&](const auto &arg) -> std::optional<expression_t> {
        return std::nullopt;
      },
      fanout, slack, p, target, std::move(aff));
}

DegreeOfParallelism RelBuilder::getDefaultDOP(DeviceType targetType) {
  return DegreeOfParallelism{(targetType == DeviceType::CPU)
                                 ? topology::getInstance().getCoreCount()
                                 : topology::getInstance().getGpuCount()};
}

RelBuilder RelBuilder::router(size_t slack, RoutingPolicy p, DeviceType target,
                              std::unique_ptr<Affinitizer> aff) const {
  return router(getDefaultDOP(target), slack, p, target, std::move(aff));
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
  void produce_(OlapParallelContext *context) override {
    return getChild()->produce(context);
  }
  void consume(OlapParallelContext *context,
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

using v_t =
    std::variant<RouterScaleOut *, Router *, DeviceCross *,
                 MemBroadcastScaleOut *, MemBroadcastDevice *, MemMoveDevice *,
                 HashRearrange *, BloomFilterBuild *, PreDecompress *,
                 BlockToTuples *,
                 CompressedBlockToTuplesDeferred *,
                 CompressedBlockToTuplesMaterialize *,
                 CompressedBlockToTuplesReg *, HintRowCount *, Select *,
                 Project *, BloomFilterRepack *>;
double expected(Operator *op);
double expectedNoSelectivity(Operator *op);

static bool hasDStringField(const RecordType &rt) {
  for (const auto *attr : rt.getArgs()) {
    const auto *type = attr->getOriginalType();
    if (type->getTypeID() == DSTRING) return true;
    if (type->getTypeID() == BLOCK) {
      auto *bt = dynamic_cast<const BlockType *>(type);
      if (bt && bt->getNestedType().getTypeID() == DSTRING) return true;
    }
  }
  return false;
}

static bool containsDStringInSubtree(const Operator *op) {
  if (!op) return false;

  if (hasDStringField(op->getRowType())) return true;

  if (auto *uall = dynamic_cast<const UnionAll *>(op)) {
    for (const auto *child : uall->getChildren()) {
      if (containsDStringInSubtree(child)) return true;
    }
    return false;
  }

  if (auto *b = dynamic_cast<const BinaryOperator *>(op)) {
    return containsDStringInSubtree(b->getLeftChild()) ||
           containsDStringInSubtree(b->getRightChild());
  }

  if (auto *u = dynamic_cast<const UnaryOperator *>(op)) {
    try {
      return containsDStringInSubtree(u->getChild());
    } catch (...) {
      return false;
    }
  }

  return false;
}

class ExpectedTuplesOutputSize {
 public:
  double operator()(Router *op) const {
    return expected(op->getChild()) * op->getChild()->getDOP() / op->getDOP();
  }
  double operator()(RouterScaleOut *op) const {
    return expected(op->getChild()) * op->getChild()->getDOPServers() /
           op->getDOPServers();
  }
  double operator()(MemBroadcastDevice *op) const {
    return expected(op->getChild()) * op->getNumberOfTargets();
  }
  double operator()(MemBroadcastScaleOut *op) const {
    return expected(op->getChild()) * op->getDOPServers();
  }
  double operator()(MemMoveDevice *op) const {
    return expected(op->getChild());
  }
  double operator()(HashRearrange *op) const {
    return expected(op->getChild());
  }
  double operator()(BloomFilterBuild *op) const {
    return expected(op->getChild());
  }
  double operator()(PreDecompress *op) const {
    return expected(op->getChild());
  }
  double operator()(BloomFilterRepack *op) const {
    return expected(op->getChild());
  }
  double operator()(DeviceCross *op) const { return expected(op->getChild()); }
  double operator()(BlockToTuples *op) const {
    return expected(op->getChild());
  }
  double operator()(CompressedBlockToTuplesDeferred *op) const {
    return expected(op->getChild());
  }
  double operator()(CompressedBlockToTuplesMaterialize *op) const {
    return expected(op->getChild());
  }
  double operator()(CompressedBlockToTuplesReg *op) const {
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

class ExpectedTuplesOutputSizeNoSelectivity {
 public:
  double operator()(Router *op) const {
    return expectedNoSelectivity(op->getChild()) * op->getChild()->getDOP() /
           op->getDOP();
  }
  double operator()(RouterScaleOut *op) const {
    return expectedNoSelectivity(op->getChild()) * op->getChild()->getDOPServers() /
           op->getDOPServers();
  }
  double operator()(MemBroadcastDevice *op) const {
    return expectedNoSelectivity(op->getChild()) * op->getNumberOfTargets();
  }
  double operator()(MemBroadcastScaleOut *op) const {
    return expectedNoSelectivity(op->getChild()) * op->getDOPServers();
  }
  double operator()(MemMoveDevice *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(HashRearrange *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(BloomFilterBuild *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(PreDecompress *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(BloomFilterRepack *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(DeviceCross *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(BlockToTuples *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(CompressedBlockToTuplesDeferred *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(CompressedBlockToTuplesMaterialize *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(CompressedBlockToTuplesReg *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(HintRowCount *op) const { return op->expected; }
  double operator()(Select *op) const {
    return expectedNoSelectivity(op->getChild());
  }
  double operator()(Project *op) const { return expectedNoSelectivity(op->getChild()); }
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

double expectedNoSelectivity(Operator *op) {
  auto obj = vinit<v_t>{}.get(op);
  return std::visit(ExpectedTuplesOutputSizeNoSelectivity{}, obj);
}

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            expression_t probe_k) const {
  double expct = expected(build.root);

  const bool needsDopScaling =
      !(build.root->getHomReplication() ==
            proteus::traits::HomReplication::BRDCST ||
        root->getHomParallelization() ==
            proteus::traits::HomParallelization::SINGLE);
  if (needsDopScaling) {
    expct = expct * build.root->getDOP();
  }

  // String dictionaries may be cached/rotated across query batches and can
  // cause hinted selectivities to under-estimate build sizes. For DString
  // build sides, allocate against a conservative no-selectivity estimate.
  if (root->getDeviceType() == DeviceType::GPU &&
      containsDStringInSubtree(build.root)) {
    double conservative = expectedNoSelectivity(build.root);
    if (needsDopScaling) conservative = conservative * build.root->getDOP();
    expct = std::max(expct, conservative);
  }

  expct = std::max(1.0, expct);
  double buildCapacity = expct * 1.1;  // 10% overestimate for safety
  if (root->getDeviceType() == DeviceType::GPU) {
    // GPU joins allocate fixed build buffers up front. Hinted selectivities are
    // often coarse on small dimension-table builds, so leave more headroom to
    // avoid aborting valid executions on minor under-estimates.
    buildCapacity = std::max(buildCapacity, expct * 1.25);
    if (expct < 4096.0) {
      buildCapacity = std::max(buildCapacity, expct + 128.0);
    }
  }
  size_t bsize = static_cast<size_t>(std::ceil(buildCapacity));

  return join(build, std::move(build_k), std::move(probe_k),
              static_cast<int>(std::ceil(std::log2(expct))) + 1, bsize);
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

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            const std::vector<GpuMatExpr> &build_e,
                            const std::vector<size_t> &build_w,
                            expression_t probe_k,
                            const std::vector<GpuMatExpr> &probe_e,
                            const std::vector<size_t> &probe_w, int hash_bits,
                            size_t maxBuildInputSize) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    assert(build.root->getHomReplication() ==
               proteus::traits::HomReplication::BRDCST ||
           root->getHomParallelization() ==
               proteus::traits::HomParallelization::SINGLE);

    auto op = new GpuHashJoinChained(
        build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
        std::move(probe_k), root, hash_bits, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  } else {
    if (build.root->getHomReplication() ==
            proteus::traits::HomReplication::BRDCST ||
        root->getHomParallelization() ==
            proteus::traits::HomParallelization::SINGLE) {
      auto op = new HashJoinChained(
          build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
          std::move(probe_k), root, hash_bits, maxBuildInputSize);
      build.apply(op);
      return apply(op);
    } else {
      auto op = new HashJoinChainedMorsel(
          build_e, build_w, std::move(build_k), build.root, probe_e, probe_w,
          std::move(probe_k), root, hash_bits, maxBuildInputSize);
      build.apply(op);
      return apply(op);
    }
  }
}

void RelBuilder::registerPlugin(const std::string &relName, Plugin *pg) {
  Catalog::getInstance().registerPlugin(relName, pg);
}

typedef Plugin *(*plugin_creator_t)(OlapParallelContext *, std::string,
                                    RecordType,
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

RelBuilder RelBuilder::scan(
    const std::vector<
        std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
        &fields) const {
  auto pg = new NvmePlugin(ctx, fields);

  auto fileName = fields[0].first->getRelationName();
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

RelBuilder RelBuilder::scan(
    const std::vector<
        std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
        &fields,
    bool preload_to_gpu, bool emit_compressed_blocks) const {
  auto pg = new NvmePlugin(ctx, fields, preload_to_gpu, emit_compressed_blocks);

  auto fileName = fields[0].first->getRelationName();
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
    auto arg = rec.getArg(attr);
    if (!arg) {
      throw proteus::runtime_error() << "Attribute " << attr << " not found";
    }
    projs.emplace_back(new RecordAttribute(*arg));
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
                             std::unique_ptr<Affinitizer> aff) const {
  return apply(new Split(
      root, alternatives,
      [&] {
        std::vector<RecordAttribute *> attrs;
        for (const auto &attr : getOutputArg().getProjections()) {
          if (p == RoutingPolicy::HASH_BASED &&
              attr.getAttrName() == "__broadcastTarget") {
            continue;
          }
          attrs.emplace_back(new RecordAttribute{attr});
        }
        return attrs;
      }(),
      slack,
      (p == RoutingPolicy::HASH_BASED)
          ? std::optional<expression_t>{getOutputArg()["__broadcastTarget"]}
          : std::nullopt,
      p));
}

SplitRelBuilder RelBuilder::gsplit(size_t slack,
                                   GeneralizedRoutingPolicy p) const {
  return SplitRelBuilder{apply(new proteus::GeneralizedRouter(
      root, slack,
      [&] {
        std::vector<RecordAttribute *> attrs;
        for (const auto &attr : getOutputArg().getProjections()) {
          if (p == GeneralizedRoutingPolicy::SHARED_HASH_BASED &&
              attr.getAttrName() == "__broadcastTarget") {
            continue;
          }
          attrs.emplace_back(new RecordAttribute{attr});
        }
        return attrs;
      }(),
      p))};
}

RelBuilder RelBuilder::unionAll(const std::vector<RelBuilder> &children,
                                DegreeOfParallelism fanout) const {
  std::vector<RecordAttribute *> projections;
  for (const auto &attr : getOutputArg().getProjections()) {
    projections.emplace_back(new RecordAttribute{attr});
  }

  return unionAll(children, projections, fanout);
}

RelBuilder RelBuilder::unionAll(
    const std::vector<RelBuilder> &children,
    const std::vector<RecordAttribute *> &wantedFields,
    DegreeOfParallelism fanout) const {
  std::vector<Operator *> c2{root};
  c2.reserve(children.size() + 1);
  for (const auto &c : children) c2.emplace_back(c.root);
  auto op = new UnionAll(c2, wantedFields, fanout);
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
  CHECK(root->isPacked()) << "unpack is not applicable to unpacked inputs";
  return unpack(
      [&](const expressions::InputArgument &arg) -> std::vector<expression_t> {
        std::vector<expression_t> attrs;
        for (const auto &attr : arg.getProjections()) {
          attrs.emplace_back(arg[attr]);
        }
        return attrs;
      });
}

RelBuilder RelBuilder::bloomfilter_repack(
    std::function<expression_t(expressions::InputArgument)> pred,
    size_t filterSize, uint64_t bloomId) const {
  CHECK(root->isPacked()) << "unpack is not applicable to unpacked inputs";
  auto arg = getOutputArg();
  std::vector<expression_t> attrs;
  for (const auto &attr : arg.getProjections()) {
    attrs.emplace_back(arg[attr]);
  }
  return bloomfilter_repack(pred(arg), attrs, filterSize, bloomId);
}

RelBuilder RelBuilder::pack() const {
  return pack([](const auto &arg) { return expression_t{0}; }, 1);
}

[[nodiscard]] RelBuilder RelBuilder::pack(ExpressionFactory hashExpr,
                                          size_t numOfBuckets) const {
  return pack(
      [&](const expressions::InputArgument &arg) -> std::vector<expression_t> {
        if (hashExpr(arg).isRegistered()) {
          auto relName = hashExpr(arg).getRegisteredRelName();
          std::vector<expression_t> attrs;
          for (const auto &attr : arg.getProjections()) {
            LOG(INFO) << arg[attr].as(relName, attr.getAttrName());
            attrs.emplace_back(arg[attr].as(relName, attr.getAttrName()));
          }
          return attrs;
        } else {
          std::vector<expression_t> attrs;
          for (const auto &attr : arg.getProjections()) {
            attrs.emplace_back(arg[attr]);
          }
          return attrs;
        }
      },
      hashExpr, numOfBuckets);
}

[[nodiscard]] bool RelBuilder::isPacked() const { return root->isPacked(); }

std::ostream &operator<<(std::ostream &out, const RelBuilder &builder) {
  return out << *(builder.operator->());
}

SplitRelBuilder::SplitRelBuilder(RelBuilder builder) : src(std::move(builder)) {
  CHECK_NE(dynamic_cast<proteus::GeneralizedRouter *>(src.root), nullptr)
      << "SplitRelBuilder can only be constructed if the root operator of the "
         "provided RelBuilder is a GeneralizedRouter";
}

RelBuilder SplitRelBuilder::path(DeviceType target,
                                 std::unique_ptr<Affinitizer> aff) const {
  return path(target, RelBuilder::getDefaultDOP(target), std::move(aff));
}

RelBuilder SplitRelBuilder::path(DeviceType target, DegreeOfParallelism dop,
                                 std::unique_ptr<Affinitizer> aff) const {
  return {src.ctx,
          dynamic_cast<proteus::GeneralizedRouter *>(src.root)->appendConsumer(
              target, dop, std::move(aff))};
}
