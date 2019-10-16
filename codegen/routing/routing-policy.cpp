/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "routing-policy.hpp"

#include "expressions/expressions-generator.hpp"
#include "topology/device-manager.hpp"
#include "topology/topology.hpp"

class CpuNumaNodeAffinitizer : public Affinitizer {
 public:
  virtual size_t getAvailableCUIndex(size_t i) const override {
    return DeviceManager::getInstance()
        .getAvailableCPUNumaNode(this, i)
        .index_in_topo;
  }

  virtual size_t size() const override {
    return topology::getInstance().getCpuNumaNodeCount();
  }

  virtual size_t getLocalCUIndex(void *p) const override {
    auto &topo = topology::getInstance();
    const auto *g = topo.getGpuAddressed(p);
    if (g) return g->getLocalCPUNumaNode().index_in_topo;
    auto *c = topo.getCpuNumaNodeAddressed(p);
    assert(c);
    return c->index_in_topo;
  }
};

class GPUAffinitizer : public Affinitizer {
 public:
  virtual size_t getAvailableCUIndex(size_t i) const override {
    return DeviceManager::getInstance().getAvailableGPU(this, i).index_in_topo;
  }
  virtual size_t size() const override {
    return topology::getInstance().getGpuCount();
  }
  virtual size_t getLocalCUIndex(void *p) const override {
    auto &topo = topology::getInstance();
    const auto *g = topo.getGpuAddressed(p);
    if (g) return g->index_in_topo;
    auto *c = topo.getCpuNumaNodeAddressed(p);
    assert(c);
    const auto &gpus = c->local_gpus;
    return gpus[rand() % gpus.size()];
  }
};

class CpuCoreAffinitizer : public CpuNumaNodeAffinitizer {
 public:
  virtual size_t getAvailableCUIndex(size_t i) const override {
    return DeviceManager::getInstance()
        .getAvailableCPUCore(this, i)
        .getLocalCPUNumaNode()
        .index_in_topo;
  }
};

AffinityPolicy::AffinityPolicy(size_t fanout, const Affinitizer *aff)
    : aff(aff) {
  indexes.resize(aff->size());
  for (size_t i = 0; i < fanout; ++i) {
    indexes[aff->getAvailableCUIndex(i)].emplace_back(i);
  }
  for (auto &ind : indexes) {
    if (!ind.empty()) continue;
    for (size_t i = 0; i < fanout; ++i) ind.emplace_back(i);
  }
}

size_t AffinityPolicy::getIndexOfRandLocalCU(void *p) const {
  auto r = rand();

  auto index_in_topo = aff->getLocalCUIndex(p);

  const auto &ind = indexes[index_in_topo];
  return ind[r % ind.size()];
}

extern "C" size_t random_local_cu(void *ptr, AffinityPolicy *aff) {
  return aff->getIndexOfRandLocalCU(ptr);
}

namespace routing {
::routing_target Random::evaluate(ParallelContext *const context,
                                  const OperatorState &childState) {
  if (fanout == 1) return {context->createInt64(0), false};
  auto Builder = context->getBuilder();

  auto crand = context->getFunction("rand");
  auto target = Builder->CreateCall(crand, {});
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), true};
}

::routing_target HashBased::evaluate(ParallelContext *const context,
                                     const OperatorState &childState) {
  auto Builder = context->getBuilder();

  ExpressionGeneratorVisitor exprGenerator{context, childState};
  auto target = e.accept(exprGenerator).value;
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), false};
}

::routing_target Local::evaluate(ParallelContext *const context,
                                 const OperatorState &childState) {
  if (fanout == 1) return {context->createInt64(0), false};

  auto Builder = context->getBuilder();
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto ptr = Builder->CreateLoad(childState[wantedField].mem);
  auto ptr8 = Builder->CreateBitCast(ptr, charPtrType);

  auto this_ptr = Builder->CreateIntToPtr(context->createInt64((uintptr_t)aff),
                                          charPtrType);

  auto target = context->gen_call("random_local_cu", {ptr8, this_ptr},
                                  context->createSizeType());

  return {target, true};
}

Local::Local(size_t fanout, DeviceType dev,
             const std::vector<RecordAttribute *> &wantedFields)
    : fanout(fanout),
      wantedField(*wantedFields[0]),
      aff(new AffinityPolicy(
          fanout, (dev == DeviceType::CPU)
                      ? static_cast<Affinitizer *>(new CpuCoreAffinitizer)
                      : static_cast<Affinitizer *>(new GPUAffinitizer))) {}

}  // namespace routing
