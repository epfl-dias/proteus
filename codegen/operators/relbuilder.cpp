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
#include "operators/flush.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"

RelBuilder RelBuilder::apply(Operator *op) const { return {*this, op}; }

RelBuilder RelBuilder::scan(Plugin &pg) const {
  return RelBuilder{ctx, new Scan(ctx, pg)};
}

RelBuilder RelBuilder::memmove(const vector<RecordAttribute *> &wantedFields,
                               size_t slack, bool to_cpu) const {
  auto op = new MemMoveDevice(root, ctx, wantedFields, slack, to_cpu);
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

PreparedStatement RelBuilder::prepare() {
  root->produce();
  ctx->prepareFunction(ctx->getGlobalFunction());
  ctx->compileAndLoad();
  return {ctx->getPipelines()};
}
