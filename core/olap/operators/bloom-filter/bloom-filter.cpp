/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "bloom-filter.hpp"

#include <expressions/expressions-generator.hpp>
#include <memory/memory-manager.hpp>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>
#include <values/indexed-seq.hpp>

#include "llvm/IR/Intrinsics.h"

std::map<std::pair<uint64_t, decltype(topology::cpunumanode::id)>, void *>
    registry;

extern "C" void setBloomFilter(Pipeline *pip, void *s, uint64_t bloomId) {
  const auto &cpu = affinity::get();
  auto k = std::make_pair(bloomId, cpu.id);
  if (registry.count(k)) MemoryManager::freePinned(registry[k]);
  registry[k] = s;
}

extern "C" void *getBloomFilter(Pipeline *pip, uint64_t bloomId) {
  const auto &cpu = affinity::get();
  auto k = std::make_pair(bloomId, cpu.id);
  // FIXME: how often is this called?
  assert(registry.count(k) > 0);
  assert(registry[k]);
  return registry[k];
}

void cleanBloomFilterRegistry() {
  for (const auto &r : registry) MemoryManager::freePinned(r.second);
}

BloomFilter::BloomFilter(Operator *child, expression_t e, size_t filterSize,
                         uint64_t bloomId)
    : experimental::UnaryOperator(child),
      e(std::move(e)),
      filterSize(filterSize),
      bloomId(bloomId) {}

llvm::Type *BloomFilter::getFilterType(ParallelContext *context) const {
  return llvm::PointerType::getUnqual(llvm::ArrayType::get(
      llvm::Type::getInt1Ty(context->getLLVMContext()), filterSize));
}

expressions::RefExpression BloomFilter::findInFilter(
    ParallelContext *context, const OperatorState &childState) const {
  auto fptr_v = context->getStateVar(filter_ptr);

  auto btype = new BoolType();
  expressions::ProteusValueExpression fptr{new type::IndexedSeq{*btype},
                                           {fptr_v, context->createFalse()}};

  // If probe
  //  auto ref = fptr[expressions::HashExpression{e}]
  assert(!(filterSize & (filterSize - 1)) &&
         "Filter size is expectd to be a power of 2");
  //  auto f = llvm::Intrinsic::getDeclaration(context->getModule(),
  //  llvm::Intrinsic::x86_pclmulqdq); assert(f); ExpressionGeneratorVisitor
  //  vis{context, childState}; f->dump(); auto hvpv = e.accept(vis); auto hv =
  //  context->getBuilder()->CreateCall(f, {hvpv.value,
  //  context->createInt32(0x75ebca6b)}); auto h =
  //  expressions::ProteusValueExpression{new IntType, {hv, hvpv.isNull}} +
  //  0x85ebca6; auto h = expressions::HashExpression{e}; expression_t h = 0;
  //  uint32_t mul = 0x75ebca6bu & ((uint32_t)filterSize - 1);
  //  for (uint32_t i = 0 ; i < sizeof(uint32_t) * 8 ; ++i){
  //    if ((mul >> i) & 1u) h = h ^ (e << ((int32_t) (i)));
  //  }
  auto h = e;

  auto hash = (filterSize & (filterSize - 1))
                  ? expression_t{h % ((int32_t)filterSize)}
                  : expression_t{h & ((int32_t)filterSize - 1)};
  //
  //  auto h = expressions::HashExpression{e};
  //
  //  auto hash = (filterSize & (filterSize - 1))
  //              ? expression_t{h % ((int64_t)filterSize)}
  //              : expression_t{h & ((int64_t)filterSize - 1)};

  return fptr[hash];
}
