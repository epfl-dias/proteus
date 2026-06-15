/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include "generalized-router.hpp"

#include <codegen/jit/pipeline.hpp>
#include <lib/expressions/expressions-generator.hpp>
#include <lib/util/catalog.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/util/demangle.hpp>
#include <platform/util/timing.hpp>

#include "lib/operators/router/routing-policy.hpp"
#include "router.hpp"

namespace proteus {

int64_t getGroupId(Pipeline *pip) { return pip->getGroup(); }

[[nodiscard]] void *acquireBufferGeneralized(int target, GeneralizedRouter *xch,
                                             int64_t groupId) {
  return xch->acquireBufferGeneralized(target, false, groupId).release();
}

[[nodiscard]] void *try_acquireBufferGeneralized(int target,
                                                 GeneralizedRouter *xch,

                                                 int64_t groupId) {
  return xch->acquireBufferGeneralized(target, true, groupId).release();
}

void releaseBufferGeneralized(int target, GeneralizedRouter *xch, void *buff) {
  xch->releaseBufferGeneralized(target, proteus::managed_ptr{buff});
}

void GeneralizedRouterConsumer::produce_(OlapParallelContext *context) {
  consume(context, {*this, {}});

  catch_pip = context->operator->();

  producer.produceForConsumer(*this, context);
}

void GeneralizedRouterConsumer::consume(OlapParallelContext *context,
                                        const OperatorState &childState) {
  auto &llvmContext = context->getLLVMContext();

  Plugin *pg = Catalog::getInstance().getPlugin(
      producer.wantedFields[0]->getRelationName());

  const ExpressionType *ptoid = pg->getOIDType();

  llvm::Type *oidType = ptoid->getLLVMType(llvmContext);

  std::vector<llvm::Type *> param_typelist;
  for (auto field : producer.wantedFields) {
    auto *wtype = field->getLLVMType(llvmContext);
    if (wtype == nullptr)
      wtype = oidType;  // FIXME: dirty hack for JSON inner lists

    param_typelist.push_back(wtype);
    producer.need_cnt =
        producer.need_cnt || (field->getOriginalType()->getTypeID() == BLOCK);
  }

  param_typelist.push_back(oidType);                              // oid
  param_typelist.push_back(llvm::Type::getInt64Ty(llvmContext));  // srcServer
  if (producer.need_cnt) param_typelist.push_back(oidType);       // cnt

  producer.params_type = llvm::StructType::get(llvmContext, param_typelist);
  producer.buf_size = context->getSizeOf(producer.params_type);

  RecordAttribute tupleCnt(producer.wantedFields[0]->getRelationName(),
                           "activeCnt",
                           pg->getOIDType());  // FIXME: OID type for blocks ?
  RecordAttribute tupleIdentifier(producer.wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType());
  RecordAttribute srcServer{producer.wantedFields[0]->getRelationName(),
                            "srcServer",
                            new Int64Type()};  // FIXME: OID type for blocks ?

  // Generate catch code
  auto p = context->appendParameter(
      llvm::PointerType::get(producer.params_type, 0), true, true);
  context->setGlobalFunction();

  auto *Builder = context->getBuilder();
  auto *entryBB = Builder->GetInsertBlock();
  auto *F = entryBB->getParent();

  context->setCurrentEntryBlock(entryBB);

  auto *mainBB = llvm::BasicBlock::Create(llvmContext, "main", F);

  auto *endBB = llvm::BasicBlock::Create(llvmContext, "end", F);
  context->setEndingBlock(endBB);

  Builder->SetInsertPoint(entryBB);

  auto *arg = context->getArgument(p);
  auto *params =
      Builder->CreateLoad(arg->getType()->getNonOpaquePointerElementType(),
                          context->getArgument(p));

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  for (size_t i = 0; i < producer.wantedFields.size(); ++i) {
    auto *param = Builder->CreateExtractValue(params, i);

    // FIMXE: should we alse transfer this information ?
    variableBindings[*(producer.wantedFields[i])] =
        context->toMem(param, context->createFalse());
  }
  auto *oid = Builder->CreateExtractValue(params, producer.wantedFields.size());
  variableBindings[tupleIdentifier] =
      context->toMem(oid, context->createFalse());

  auto *srv =
      Builder->CreateExtractValue(params, producer.wantedFields.size() + 1);
  variableBindings[srcServer] =
      context->toMem(srv, context->createFalse(), "srcServer");

  if (producer.need_cnt) {
    auto *cnt =
        Builder->CreateExtractValue(params, producer.wantedFields.size() + 2);
    variableBindings[tupleCnt] = context->toMem(cnt, context->createFalse());
  }

  Builder->SetInsertPoint(mainBB);

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  Builder->CreateBr(endBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(mainBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  // Builder->CreateRetVoid();
}

std::unique_ptr<routing::RoutingPolicy> GeneralizedRouter::getPolicy(
    GeneralizedRoutingPolicy p, DegreeOfParallelism dop,
    const std::vector<RecordAttribute *> &_wantedFields) {
  switch (p) {
      //    case RoutingPolicy::HASH_BASED: {
      //      assert(hashExpr.has_value());
      //      return std::make_unique<routing::HashBased>(fanout,
      //      hashExpr.value());
      //    }
      //      //    case RoutingPolicy::LOCAL: {
      //      //      return std::make_unique<routing::Local>(
      //      //          fanout, wantedFields, new AffinityPolicy(fanout,
      //      //          aff.get()));
      //      //      //      return std::make_unique<routing::PreferLocal>(
      //      //      //          fanout, wantedFields, new
      //      AffinityPolicy(fanout,
      //      //      aff.get()));
      //      //    }
      //      //    case RoutingPolicy::FORCE_LOCAL: {
      //      //      return std::make_unique<routing::Local>(
      //      //          fanout, wantedFields, new AffinityPolicy(fanout,
      //      //          aff.get()));
      //      //    }
    case GeneralizedRoutingPolicy::SHARED_LOCAL:  // FIXME: add the flexible
                                                  // version
    case GeneralizedRoutingPolicy::SHARED_FORCE_LOCAL: {
      return std::make_unique<routing::Local>(
          dop, _wantedFields,
          new AffinityPolicy(dop, new CpuNumaNodeAffinitizer()));
    }
    case GeneralizedRoutingPolicy::SHARED_RANDOM: {
      // TODO getPolicy isn't used for SHARED_RANDOM
      return std::make_unique<routing::Random>(dop);
    }
    case GeneralizedRoutingPolicy::DISTINCT_RANDOM_SPLIT_DATA_LOCAL: {
      std::vector<AffinityPolicy *> affs;
      std::vector<DeviceType> device_types;
      for (auto &consumer : consumers) {
        affs.emplace_back(consumer->aff_policy.get());
        device_types.emplace_back(consumer->target_device);
      }
      return std::make_unique<routing::RandomSplitDataLocal>(
          _wantedFields, affs, device_types);
    }
    default: {
      assert(false && "Unimplemented");  // FIXME: rest of the policies
    }
  }
}

void GeneralizedRouter::produceForConsumer(
    const GeneralizedRouterConsumer &cons, OlapParallelContext *context) {
  assert(cons.producer == *this);
  if (&cons == consumers.back().get()) {
    produce(context);
  }
}

void GeneralizedRouter::produce_(OlapParallelContext *context) {
  context->popPipeline();

  // push new pipeline for the throw part
  context->pushPipeline();

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  groupVar = context->appendStateVar(
      llvm::IntegerType::getIntNTy(context->getLLVMContext(),
                                   sizeof(int64_t) * 8),
      [=](llvm::Value *pip) {
        return context->gen_call(proteus::getGroupId, {pip});
      },
      [=](llvm::Value *, llvm::Value *s) {});

  getChild()->produce(context);
}

llvm::Value *GeneralizedRouter::createTaskDescription(
    OlapParallelContext *context, const OperatorState &childState) {
  auto *Builder = context->getBuilder();

  llvm::Value *params = llvm::UndefValue::get(params_type);

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());

  auto rec = childState.getProducer().getRowType();
  ExpressionGeneratorVisitor vis{context, childState};
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    auto v =
        expressions::InputArgument{&rec}[*wantedFields[i]].accept(vis).value;

    auto vi =
        (v->getType()->isPointerTy() &&
         v->getType()->getPointerElementType()->isArrayTy() &&
         v->getType()->getPointerElementType()->getArrayElementType() ==
             params_type->getStructElementType(i)->getPointerElementType())
            ? Builder->CreateInBoundsGEP(
                  v->getType()->getNonOpaquePointerElementType(), v,
                  {context->createInt32(0), context->createInt32(0)})
            : v;  // Is this still relevant?

    params = Builder->CreateInsertValue(params, vi, i);
  }

  RecordAttribute tupleIdentifier(wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType());

  ProteusValueMemory mem_oidWrapper = childState[tupleIdentifier];
  params = Builder->CreateInsertValue(
      params,
      Builder->CreateLoad(
          mem_oidWrapper.mem->getType()->getPointerElementType(),
          mem_oidWrapper.mem),
      wantedFields.size());

  auto srcServer = [&]() -> llvm::Value * {
    try {
      return Builder->CreateLoad(childState[{wantedFields[0]->getRelationName(),
                                             "srcServer", new Int64Type()}]
                                     .mem->getType()
                                     ->getPointerElementType(),
                                 childState[{wantedFields[0]->getRelationName(),
                                             "srcServer", new Int64Type()}]
                                     .mem);
    } catch (const std::out_of_range &) {
      return context->createInt64(InfiniBandManager::server_id());
    }
  }();

  params =
      Builder->CreateInsertValue(params, srcServer, wantedFields.size() + 1);

  if (need_cnt) {
    RecordAttribute tupleCnt(wantedFields[0]->getRelationName(), "activeCnt",
                             pg->getOIDType());  // FIXME: OID type for blocks ?

    ProteusValueMemory mem_cntWrapper = childState[tupleCnt];
    params = Builder->CreateInsertValue(
        params,
        Builder->CreateLoad(
            mem_cntWrapper.mem->getType()->getPointerElementType(),
            mem_cntWrapper.mem),
        wantedFields.size() + 2);
  }

  return params;
}

void GeneralizedRouter::consume(OlapParallelContext *context,
                                const OperatorState &childState) {
  /*
   * At this point we are compiling, so no more GeneralizedRouterConsumers
   * should be added. This means we know how many pipelines we are splitting
   * into. So we construct the routing policy here to enable policies that wish
   * to use that the number of split pipelines.
   */
  DCHECK_EQ(routing, nullptr);
  routing = getPolicy(
      policy_type,
      DegreeOfParallelism{topology::getInstance().getCpuNumaNodes().size()},
      wantedFields);

  // Warmup threads to avoid thread creation overhead
  for (const auto &cons : consumers) {
    for (int i = 0; i < cons->fanout; i++) {
      firers.emplace_back([]() {});
    }
  }
  for (auto &t : firers) t.get();
  firers.clear();

  llvm::LLVMContext &llvmContext = context->getLLVMContext();
  llvm::IRBuilder<> *Builder = context->getBuilder();

  llvm::PointerType *charPtrType = llvm::Type::getInt8PtrTy(llvmContext);
  auto groupId = context->getStateVar(groupVar);

  auto params = createTaskDescription(context, childState);

  llvm::Value *exchangePtr =
      llvm::ConstantInt::get(llvmContext, llvm::APInt(64, ((uint64_t)this)));
  llvm::Value *exchange = Builder->CreateIntToPtr(exchangePtr, charPtrType);

  auto retry_cnt =
      context->toMem(context->createInt32(0), context->createFalse());

  llvm::Value *param_ptr;
  llvm::Value *target;
  context
      ->gen_do([&]() {
        // FIXME: rest of policies
        switch (policy_type) {
          case GeneralizedRoutingPolicy::SHARED_RANDOM: {
            DCHECK_EQ(getNumberOfQueues(), 1);
            target = context->createInt32(0);
            param_ptr = context->gen_call(proteus::acquireBufferGeneralized,
                                          {target, exchange, groupId});
            break;
          }
          default: {
            auto srcServer = [&]() -> llvm::Value * {
              try {
                return Builder->CreateLoad(
                    childState[{wantedFields[0]->getRelationName(), "srcServer",
                                new Int64Type()}]
                        .mem->getType()
                        ->getPointerElementType(),
                    childState[{wantedFields[0]->getRelationName(), "srcServer",
                                new Int64Type()}]
                        .mem);
              } catch (const std::out_of_range &) {
                return context->createInt64(InfiniBandManager::server_id());
              }
            }();

            bool may_retry = false;

            auto phi_type =
                llvm::IntegerType::getInt32Ty(context->getLLVMContext());
            llvm::BasicBlock *b1;
            llvm::BasicBlock *b2;
            llvm::Value *p1;
            llvm::Value *p2;
            context
                ->gen_if({Builder->CreateICmpEQ(
                              srcServer, context->createInt64(
                                             InfiniBandManager::server_id())),
                          context->createFalse()})([&]() {
                  auto &x = *routing;
                  LOG(INFO) << demangle(typeid(x).name());
                  auto r = routing->evaluate(context, childState, retry_cnt);

                  r.target->setName("target");
                  p1 = Builder->CreateTruncOrBitCast(
                      r.target, llvm::Type::getInt32Ty(llvmContext));
                  may_retry = r.may_retry;

                  b1 = Builder->GetInsertBlock();
                })
                .gen_else([&]() {
                  p2 = Builder->CreateURem(
                      Builder->CreateTruncOrBitCast(
                          context->gen_call(rand, {}),
                          llvm::Type::getInt32Ty(llvmContext)),
                      context->createInt32(
                          topology::getInstance()
                              .getCpuNumaNodeCount()));  // TODO assumes CPU
                                                         // NUMA affinitization
                                                         // only for now
                  may_retry = false;

                  b2 = Builder->GetInsertBlock();
                });

            auto phi = Builder->CreatePHI(phi_type, 2);
            phi->addIncoming(p1, b1);
            phi->addIncoming(p2, b2);
            target = phi;

            assert(!may_retry &&
                   "Unimplemented, needs to take another path above due to the "
                   "mismatch of the two may_retry paths");
            param_ptr = context->gen_call(
                (may_retry)
                    ? (proteus::try_acquireBufferGeneralized /* FIXME */)
                    : (proteus::acquireBufferGeneralized),
                {target, exchange, groupId});

            break;
          }
        }

        Builder->CreateStore(
            Builder->CreateAdd(
                Builder->CreateLoad(
                    retry_cnt.mem->getType()->getPointerElementType(),
                    retry_cnt.mem),
                context->createInt32(1)),
            retry_cnt.mem);
      })
      .gen_while([&]() {
        llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
            ((llvm::PointerType *)param_ptr->getType()));
        llvm::Value *is_null = Builder->CreateICmpEQ(param_ptr, null_ptr);

        return ProteusValue{is_null, context->createFalse()};
      });

  param_ptr = Builder->CreateBitCast(
      param_ptr, llvm::PointerType::getUnqual(params->getType()));

  Builder->CreateStore(params, param_ptr);

  context->gen_call(
      proteus::releaseBufferGeneralized,
      {target, exchange, Builder->CreateBitCast(param_ptr, charPtrType)});
}

size_t GeneralizedRouter::getNumberOfQueues() const {
  switch (policy_type) {
    case GeneralizedRoutingPolicy::SHARED_LOCAL:  // FIMXE
    case GeneralizedRoutingPolicy::SHARED_FORCE_LOCAL: {
      auto &topo = topology::getInstance();
      return topo.getCpuNumaNodeCount() + topo.getGpuCount();
    }
    case GeneralizedRoutingPolicy::SHARED_RANDOM: {
      return 1;
    }
    case GeneralizedRoutingPolicy::SHARED_HASH_BASED: {
      // Note: code path is not tested at the moment
      size_t consumer_dop = consumers.front()->getDOP();
      for (const auto &cons : consumers) {
        CHECK_EQ(cons->getDOP(), consumer_dop)
            << "All consumers must have the "
               "same DOP for SHARED_HASH_BASED";
      }
      return consumer_dop;
    }
    case GeneralizedRoutingPolicy::DISTINCT_RANDOM_SPLIT_DATA_LOCAL: {
      auto &topo = topology::getInstance();
      return consumers.size() *
             (topo.getCpuNumaNodeCount() + topo.getGpuCount());
    }
    default: {
      LOG(FATAL) << "Unimplemented";
    }
  }
}

DegreeOfParallelism GeneralizedRouter::getDOP() const {
  size_t total = 0;
  for (auto &cons : consumers) total += cons->getDOP();
  return DegreeOfParallelism{total};
}

void *GeneralizedRouter::allocate_buffers_for_queue(size_t queue) {
  DCHECK_NE(buf_size, 0);
  void *mem = MemoryManager::mallocPinned(buf_size * slack);
  for (int j = 0; j < slack; ++j) {
    freeBufferGeneralized(queue,
                          proteus::managed_ptr{((char *)mem) + j * buf_size});
  }
  const int fifo_size = ready_fifo.at(queue).size_unsafe();
  counterlogger.log(getUUID(), counter_type::ROUTER_READY_QUEUE_SIZE, fifo_size,
                    queue);
  const int free_size = free_pool.at(queue).size_unsafe();
  counterlogger.log(getUUID(), counter_type::ROUTER_FREE_POOL_SIZE, free_size,
                    queue);
  return mem;
}

void GeneralizedRouter::create_queues() {
  const auto queueCnt = getNumberOfQueues();
  if (free_pool.size() != queueCnt) {
    free_pool.clear();
    ready_fifo.clear();
    for (size_t i = 0; i < queueCnt; ++i) {
      // note, queues currently don't use numa affinity for memory allocation
      free_pool.emplace_back(1);
      ready_fifo.emplace_back(1);
    }
  } else {
    for (auto &f2 : free_pool) f2.reset();
    for (auto &f2 : ready_fifo) f2.reset();
  }
  return;
}

void GeneralizedRouter::open(Pipeline *pip) {
  event_range<range_log_op::GROUTER_OPEN> er{id, pip->getGeneratorUUID(),
                                             pip->getGroup()};
  std::lock_guard<std::mutex> guard(init_mutex);

  if (firers.empty()) {
    {
      event_range<range_log_op::GROUTER_CREATE_QUEUES> e{
          id, pip->getGeneratorUUID()};
      create_queues();
    }
    remaining_producers = producers;
    auto &topo = topology::getInstance();
    auto queue_offset = [policy = policy_type,
                         num_numa_nodes =
                             topo.getCpuNumaNodeCount() + topo.getGpuCount()](
                            size_t consumer_index) -> size_t {
      switch (policy) {
        case GeneralizedRoutingPolicy::SHARED_LOCAL:
        case GeneralizedRoutingPolicy::SHARED_FORCE_LOCAL:
        case GeneralizedRoutingPolicy::SHARED_RANDOM:
        case GeneralizedRoutingPolicy::SHARED_HASH_BASED: {
          return 0;
        }
        case GeneralizedRoutingPolicy::DISTINCT_RANDOM_SPLIT_DATA_LOCAL: {
          return consumer_index * num_numa_nodes;
        }
        default: {
          LOG(FATAL) << "Unimplemented";
        }
      }
    };
    size_t consumer_index = 0;
    for (auto &cons : consumers) {
      event_range<range_log_op::GROUTER_INIT_CONS> e{
          id, cons->catch_pip->getUUID()};
      cons->spawnWorker(pip->getSession(), queue_offset(consumer_index),
                        firers);
      consumer_index += 1;
    }
  }
}

void GeneralizedRouter::close(Pipeline *pip) {
  time_block t("Tterm_exchange: ");

  int rem = --remaining_producers;
  CHECK_GE(rem, 0);

  if (rem == 0) {
    event_range<range_log_op::GROUTER_CLOSE> er{id, pip->getUUID(),
                                                pip->getGroup()};
    for (auto &r : ready_fifo) {
      r.close();
    }

    nvtxRangePushA("Exchange_waiting_to_close");
    for (auto &thread : firers) thread.get();
    nvtxRangePop();
    firers.clear();
    for (auto &r : free_pool) r.close();
  }
}

GeneralizedRouterConsumer *GeneralizedRouter::appendConsumer(
    DeviceType target_device, DegreeOfParallelism dop,
    std::unique_ptr<Affinitizer> aff) {
  if (dop < aff->size()) {
    if (policy_type == GeneralizedRoutingPolicy::SHARED_LOCAL ||
        policy_type == GeneralizedRoutingPolicy::SHARED_FORCE_LOCAL) {
      LOG(WARNING) << "Degree of parallelism of this consumer is less than the "
                      "number of available CUs in the affinitizer. This may "
                      "lead to data being  routed to a queue which no workers "
                      "are consuming from.";
    }
  }

  consumers.emplace_back(std::make_unique<GeneralizedRouterConsumer>(
      *this, dop, std::move(aff), target_device));
  return consumers.back().get();
}

bool GeneralizedRouterConsumer::isFiltering() const {
  return producer.isFiltering();
}

RecordType GeneralizedRouterConsumer::getRowType() const {
  return producer.getRowType();
}

DegreeOfParallelism GeneralizedRouterConsumer::getDOPServers() const {
  return producer.getDOPServers();
}

DeviceType GeneralizedRouterConsumer::getDeviceType() const {
  return producer.getDeviceType();
}

bool GeneralizedRouterConsumer::isPacked() const { return producer.isPacked(); }

proteus::traits::HomReplication GeneralizedRouterConsumer::getHomReplication()
    const {
  return producer.getHomReplication();
}

void GeneralizedRouterConsumer::foreachTaskDo(int target_queue, Pipeline *pip,
                                              PipelineGen *pipGen,
                                              std::function<void(void *)> f) {
  DCHECK_LE(target_queue, producer.ready_fifo.size());
  producer.ready_fifo.at(target_queue)
      .foreachItemDo(
          [&]() {
            const int fifo_size =
                producer.ready_fifo.at(target_queue).size_unsafe();
            counterlogger.log(producer.getUUID(),
                              counter_type::ROUTER_READY_QUEUE_SIZE, fifo_size,
                              target_queue);
            const int free_size =
                producer.free_pool.at(target_queue).size_unsafe();
            counterlogger.log(producer.getUUID(),
                              counter_type::ROUTER_FREE_POOL_SIZE, free_size,
                              target_queue);
            return event_range<range_log_op::ROUTER_WAITING_FOR_TASK>{
                id, pipGen->getUUID(), pip->getGroup()};
          },
          [&](void *ptr) {
            f(ptr);

            {
              producer.freeBufferGeneralized(target_queue,
                                             proteus::managed_ptr{ptr});
              std::this_thread::yield();
            }
          });
}

void GeneralizedRouterConsumer::fire(int target_queue, int local_target,
                                     PipelineGen *pipGen, const void *session,
                                     bool should_allocate_queue_buffs) {
  pthread_setname_np(pthread_self(),
                     (pipGen->getName() + ":" + std::to_string(target_queue) +
                      ":" + std::to_string(target_queue))
                         .c_str());
  auto &cu = aff->getAvailableCU(local_target);

  auto exec_affinity = cu.set_on_scope();
  auto pip = pipGen->getPipeline(local_target);
  event_range<range_log_op::GROUTER_CONS_FIRE> e{id, pip->getGeneratorUUID(),
                                                 pip->getGroup()};
  // if we remove that, following opens may allocate memory to wrong socket!
  std::this_thread::yield();
  void *buffer_mem = nullptr;
  if (should_allocate_queue_buffs) {
    event_range<range_log_op::GROUTER_ALLOC_QUEUE_BUFFS> e{
        id, pip->getGeneratorUUID(), pip->getGroup()};
    buffer_mem = producer.allocate_buffers_for_queue(target_queue);
  }

  pip->open(session);
  {
    foreachTaskDo(target_queue, pip.get(), pipGen, [&](void *ptr) {
      pip->consume((void *)(((uintptr_t)ptr) & ~uintptr_t(1)));
    });
  }

  pip->close();
  if (buffer_mem != nullptr) {
    for (int j = 0; j < producer.slack; ++j) {
      /* Release and ignore, it will be handled by the following freePinned */
      ((void)(producer
                  .acquireBufferGeneralized(target_queue, false,
                                            pip->getGroup())
                  .release()));
    }
    MemoryManager::freePinned(buffer_mem);
  }
}

void GeneralizedRouterConsumer::spawnWorker(const void *session,
                                            size_t queue_offset,
                                            threadvector &firers) {
  /// local_targets holds the offsets for the target queues (i.e numa
  /// nodes/gpus) ignoring which consumer this is which is then accounted for by
  /// the queue_offset. When the queues are shared between consumers
  /// queue_offset will be 0. When the queues are distinct queue_offset will be
  /// a multiple of the total number of CPU numa nodes + gpus in the system.
  /// This is important for the case where a consumer may only run on a subset
  /// of nodes/gpus
  const std::vector<int> local_targets =
      [dop = getDOP(), routing_policy = producer.policy_type,
       device_type = target_device,
       affinitizer = aff.get()]() -> std::vector<int> {
    if (routing_policy == GeneralizedRoutingPolicy::SHARED_RANDOM) {
      return {0};
    }
    std::vector<int> temp_local_targets;
    if (routing_policy == GeneralizedRoutingPolicy::SHARED_HASH_BASED) {
      // note: not a tested code path
      for (int i = 0; i < dop; ++i) {
        temp_local_targets.emplace_back(i);
      }
      return temp_local_targets;
    }

    // This handles SHARED_LOCAL and DISTINCT_RANDOM_SPLIT_DATA_LOCAL as later
    // we account for the difference with queue_offset
    switch (device_type) {
      case DeviceType::CPU: {
        for (int i = 0; i < affinitizer->size(); i++) {
          const auto *node = dynamic_cast<const topology::cpunumanode *>(
              &affinitizer->getAvailableCU(i));
          CHECK_NE(node, nullptr)
              << "Affinitizer for a consumer targeting DeviceType::CPU must "
                 "affinitize to a topology::cpunumanode";
          temp_local_targets.emplace_back(node->index_in_topo);
        }
        return temp_local_targets;
      }
      case DeviceType::GPU: {
        const auto cpu_node_count =
            topology::getInstance().getCpuNumaNodeCount();
        for (int i = 0; i < affinitizer->size(); i++) {
          const auto *node = dynamic_cast<const topology::gpunode *>(
              &affinitizer->getAvailableCU(i));
          CHECK_NE(node, nullptr)
              << "Affinitizer for a consumer targeting DeviceType::GPU must "
                 "affinitize to a topology::cpunode";
          temp_local_targets.emplace_back(node->index_in_topo + cpu_node_count);
          CHECK_LE(temp_local_targets.back(),
                   topology::getInstance().getGpuCount() + cpu_node_count);
        }
        return temp_local_targets;
      }
      default: {
        LOG(FATAL) << "Unimplemented";
      }
    }
  }();

  for (size_t i = 0; i < fanout; ++i) {
    // The firers allocate buffers so that open can be parallelized
    const bool alloc_buffers = [&]() -> bool {
      switch (producer.policy_type) {
        case GeneralizedRoutingPolicy::SHARED_RANDOM: {
          return firers.empty();
        }
        case GeneralizedRoutingPolicy::SHARED_HASH_BASED: {
          /// Untested code path
          return firers.size() < fanout;
        }
        case GeneralizedRoutingPolicy::SHARED_LOCAL: {
          // this _may_ be a race condition
          if (i < local_targets.size()) {
            return producer.free_pool
                .at(local_targets[i % local_targets.size()])
                .empty();
          } else {
            return false;
          }
        }
        case GeneralizedRoutingPolicy::DISTINCT_RANDOM_SPLIT_DATA_LOCAL: {
          return i < local_targets.size();
        }
        default:
          return false;
      }
    }();

    firers.emplace_back(&GeneralizedRouterConsumer::fire, this,
                        queue_offset + local_targets[i % local_targets.size()],
                        i, catch_pip, session, alloc_buffers);
  }
}

proteus::managed_ptr GeneralizedRouter::acquireBufferGeneralized(
    int target, bool polling, int64_t groupId) {
  DCHECK_LT(target, free_pool.size());
  if (free_pool.at(target).empty_unsafe() && polling) {
    LOG(INFO) << free_pool.at(target).size_unsafe();
    nvtxRangePop();
    return nullptr;
  }
  void *buff = nullptr;
  DCHECK_LE(target, ready_fifo.size());
  auto x = free_pool.at(target).pop(buff);
  DCHECK(x);

  return proteus::managed_ptr{buff};
}

void GeneralizedRouter::releaseBufferGeneralized(int target,
                                                 proteus::managed_ptr buff) {
  assert(target < ready_fifo.size());
  ready_fifo.at(target).push(buff.release());
}

void GeneralizedRouter::freeBufferGeneralized(int target,
                                              proteus::managed_ptr buff) {
  DCHECK_LE(target, free_pool.size());
  free_pool.at(target).emplace(buff.release());
}

bool GeneralizedRouter::get_readyGeneralized(int target,
                                             proteus::managed_ptr &buff) {
  assert(target < ready_fifo.size() && target >= 0);

  void *ptr;
  bool r = ready_fifo.at(target).pop2(ptr);
  if (r) buff = proteus::managed_ptr{ptr};
  return r;
}

}  // namespace proteus
