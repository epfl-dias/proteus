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

#include "operators/relbuilder.hpp"

#include <iomanip>

#include "operators/block-to-tuples.hpp"
#include "operators/cpu-to-gpu.hpp"
#include "operators/exchange.hpp"
#include "operators/flush.hpp"
#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#include "operators/hash-join-chained.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/mem-broadcast-device.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plan/plan-parser.hpp"

const RecordType &RelBuilder::getRecordType(CatalogParser &catalog,
                                            std::string relName) const {
  auto inputInfo = catalog.getInputInfoIfKnown(relName);

  CollectionType &collType =
      dynamic_cast<CollectionType &>(*(inputInfo->exprType));

  const ExpressionType &nestedType = collType.getNestedType();
  return dynamic_cast<const RecordType &>(nestedType);
}

void RelBuilder::setOIDType(CatalogParser &catalog, std::string relName,
                            ExpressionType *type) const {
  catalog.getInputInfo(relName)->oidType = type;
}

RelBuilder RelBuilder::apply(Operator *op) const { return {*this, op}; }

RelBuilder RelBuilder::scan(Plugin &pg) const {
  return RelBuilder{ctx, new Scan(ctx, pg)};
}

RelBuilder RelBuilder::memmove(const vector<RecordAttribute *> &wantedFields,
                               size_t slack, bool to_cpu) const {
  auto op = new MemMoveDevice(root, ctx, wantedFields, slack, to_cpu);
  return apply(op);
}

RelBuilder RelBuilder::membrdcst(const vector<RecordAttribute *> &wantedFields,
                                 size_t fanout, bool to_cpu,
                                 bool always_share) const {
  auto op = new MemBroadcastDevice(root, ctx, wantedFields, fanout, to_cpu,
                                   always_share);
  return apply(op);
}

RelBuilder RelBuilder::to_gpu() const {
  return to_gpu([&](const auto &arg) -> std::vector<RecordAttribute *> {
    std::vector<RecordAttribute *> ret;
    for (const auto &attr : arg.getProjections()) {
      if (dynamic_cast<const BlockType *>(attr.getOriginalType())) {
        ret.emplace_back(new RecordAttribute{attr});
      }
    }
    assert(ret.size() != 0);
    return ret;
  });
}

RelBuilder RelBuilder::to_cpu(gran_t granularity, size_t size) const {
  return to_cpu(
      [&](const auto &arg) -> std::vector<RecordAttribute *> {
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
        return ret;
      },
      granularity, size);
}

RelBuilder RelBuilder::to_gpu(
    const vector<RecordAttribute *> &wantedFields) const {
  auto op = new CpuToGpu(root, ctx, wantedFields);
  return apply(op);
}

RelBuilder RelBuilder::to_cpu(const vector<RecordAttribute *> &wantedFields,
                              gran_t granularity, size_t size) const {
  auto op = new GpuToCpu(root, ctx, wantedFields, size, granularity);
  return apply(op);
}

RelBuilder RelBuilder::filter(expression_t pred) const {
  auto op = new Select(pred, root);
  return apply(op);
}

RelBuilder RelBuilder::unpack(const vector<expression_t> &projections) const {
  return unpack(projections, (root->getDeviceType() == DeviceType::GPU)
                                 ? gran_t::GRID
                                 : gran_t::THREAD);
}

RelBuilder RelBuilder::unpack(const vector<expression_t> &projections,
                              gran_t granularity) const {
  auto op =
      new BlockToTuples(root, ctx, projections,
                        root->getDeviceType() == DeviceType::GPU, granularity);
  return apply(op);
}

RelBuilder RelBuilder::pack(const vector<expression_t> &projections,
                            expression_t hashExpr, size_t numOfBuckets) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op =
        new GpuHashRearrange(root, ctx, numOfBuckets, projections, hashExpr);
    return apply(op);
  } else {
    auto op = new HashRearrange(root, ctx, numOfBuckets, projections, hashExpr);
    return apply(op);
  }
}

RelBuilder RelBuilder::reduce(const vector<expression_t> &e,
                              const vector<Monoid> &accs) const {
  assert(e.size() == accs.size());
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new opt::GpuReduce(accs, e, expression_t{true}, root, ctx);
    return apply(op);
  } else {
    auto op =
        new opt::Reduce(accs, e, expression_t{true}, root, ctx, false, "");
    return apply(op);
  }
}

RelBuilder RelBuilder::print(const vector<expression_t> &e) const {
  auto op = new Flush(e, root, ctx);
  return apply(op);
}

RelBuilder RelBuilder::unnest(expression_t e) const {
  auto op = new Unnest(true, e, root);
  return apply(op);
}

PreparedStatement RelBuilder::prepare() {
  root->produce();
  ctx->prepareFunction(ctx->getGlobalFunction());
  ctx->compileAndLoad();
  return {ctx->getPipelines()};
}

RelBuilder RelBuilder::router(const vector<RecordAttribute *> &wantedFields,
                              std::optional<expression_t> hash, size_t fanout,
                              size_t fanin, size_t slack, RoutingPolicy p,
                              bool cpu_targets, int numa_socket_id) const {
  assert((p == RoutingPolicy::HASH_BASED) == (hash.has_value()));
  assert((p == RoutingPolicy::RANDOM) == (!hash.has_value()));
  auto op = new Exchange(root, ctx, fanout, wantedFields, slack, hash,
                         p == RoutingPolicy::GPU_NUMA_LOCAL,
                         p == RoutingPolicy::RAND_LOCAL_CPU, fanin, cpu_targets,
                         numa_socket_id);
  return apply(op);
}

RelBuilder RelBuilder::join(RelBuilder build, expression_t build_k,
                            const std::vector<GpuMatExpr> &build_e,
                            const std::vector<size_t> &build_w,
                            expression_t probe_k,
                            const std::vector<GpuMatExpr> &probe_e,
                            const std::vector<size_t> &probe_w, int hash_bits,
                            size_t maxBuildInputSize) const {
  if (root->getDeviceType() == DeviceType::GPU) {
    auto op = new GpuHashJoinChained(build_e, build_w, build_k, build.root,
                                     probe_e, probe_w, probe_k, root, hash_bits,
                                     ctx, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  } else {
    auto op = new HashJoinChained(build_e, build_w, build_k, build.root,
                                  probe_e, probe_w, probe_k, root, hash_bits,
                                  ctx, maxBuildInputSize);
    build.apply(op);
    return apply(op);
  }
}
