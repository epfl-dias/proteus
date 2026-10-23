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

#ifndef PROTEUS_GENERALIZED_ROUTER_HPP
#define PROTEUS_GENERALIZED_ROUTER_HPP

#include <olap/routing/affinitizers.hpp>
#include <platform/memory/managed-pointer.hpp>
#include <platform/threadpool/threadvector.hpp>
#include <platform/util/datastructures/threadsafe-set.hpp>

#include "lib/operators/operators.hpp"
#include "lib/operators/router/routing-policy.hpp"

namespace proteus {

class GeneralizedRouter;

[[nodiscard]] void *acquireBufferGeneralized(int target, GeneralizedRouter *xch,
                                             int64_t groupId);
[[nodiscard]] void *try_acquireBufferGeneralized(int target,
                                                 GeneralizedRouter *xch,
                                                 int64_t groupId);
void releaseBufferGeneralized(int target, GeneralizedRouter *xch, void *buff);

class GeneralizedRouterConsumer final : public experimental::Operator {
 protected:
  GeneralizedRouter &producer;

  const DegreeOfParallelism fanout;
  PipelineGen *catch_pip;

  std::unique_ptr<Affinitizer> aff;
  std::unique_ptr<AffinityPolicy> aff_policy;
  const DeviceType target_device;

 public:
  GeneralizedRouterConsumer(GeneralizedRouter &producer,
                            DegreeOfParallelism fanout,
                            std::unique_ptr<Affinitizer> aff,
                            DeviceType target_device)
      : producer(producer),
        fanout(fanout),
        aff(std::move(aff)),
        aff_policy(std::make_unique<AffinityPolicy>(this->aff->size(),
                                                    this->aff.get())),
        target_device(target_device) {}

  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] bool isFiltering() const override;

  [[nodiscard]] RecordType getRowType() const override;

  [[nodiscard]] DegreeOfParallelism getDOP() const override { return fanout; }
  [[nodiscard]] DeviceType getDeviceType() const override;

  [[nodiscard]] DegreeOfParallelism getDOPServers() const override;
  [[nodiscard]] bool isPacked() const override;
  [[nodiscard]] proteus::traits::HomReplication getHomReplication()
      const override;

  [[nodiscard]] GeneralizedRouter &getProducer() const { return producer; }

 protected:
  void produce_(OlapParallelContext *context) override;
  virtual void spawnWorker(const void *session, size_t queue_offset,
                           threadvector &firers);
  virtual void fire(int target_queue, int local_target, PipelineGen *pipGen,
                    const void *session, bool should_allocate_queue_buffs);
  virtual void foreachTaskDo(int target, Pipeline *pip, PipelineGen *pipGen,
                             std::function<void(void *)> f);

  friend class GeneralizedRouter;
};

class GeneralizedRouter final : public experimental::UnaryOperator {
  std::vector<std::unique_ptr<GeneralizedRouterConsumer>> consumers;

  threadvector firers;

  const size_t slack;

  const std::vector<RecordAttribute *> wantedFields;
  llvm::Type *params_type;

  StateVar groupVar;
  size_t buf_size;
  bool need_cnt = false;

  std::atomic<int> remaining_producers;
  int producers;

  std::mutex init_mutex;

  //  std::deque<AsyncQueueMPMC<void *>,
  //  proteus::memory::ExplicitSocketPinnedMemoryAllocator<AsyncQueueMPMC<void
  //  *>>>
  //  ready_fifo{proteus::memory::ExplicitSocketPinnedMemoryAllocator<AsyncQueueMPMC<void
  //  *>>{1}};
  std::deque<AsyncQueueMPMC<void *>> ready_fifo{};

  //  std::deque<AsyncQueueMPMC<void *>,
  //  proteus::memory::ExplicitSocketPinnedMemoryAllocator<AsyncQueueMPMC<void
  //  *>>>
  //  free_pool{proteus::memory::ExplicitSocketPinnedMemoryAllocator<AsyncQueueMPMC<void
  //  *>>{1}};
  std::deque<AsyncQueueMPMC<void *>> free_pool{};
  //  std::deque<AsyncQueueMPMC<void *>,
  //             proteus::memory::ExplicitSocketPinnedMemoryAllocator<
  //                 AsyncQueueMPMC<void *>>>
  //      ready_fifo{proteus::memory::ExplicitSocketPinnedMemoryAllocator<
  //          AsyncQueueMPMC<void *>>{1}};
  //
  //  std::deque<AsyncQueueMPMC<void *>,
  //             proteus::memory::ExplicitSocketPinnedMemoryAllocator<
  //                 AsyncQueueMPMC<void *>>>
  //      free_pool{proteus::memory::ExplicitSocketPinnedMemoryAllocator<
  //          AsyncQueueMPMC<void *>>{1}};
  //  //  threadsafe_set<void *> *free_pool = nullptr;

  const GeneralizedRoutingPolicy policy_type;
  std::unique_ptr<routing::RoutingPolicy> routing;

 public:
  GeneralizedRouter(Operator *child, size_t slack,
                    std::vector<RecordAttribute *> attrs,
                    GeneralizedRoutingPolicy policy_type)
      : experimental::UnaryOperator(child),
        slack(slack),
        wantedFields(std::move(attrs)),
        policy_type(policy_type),
        producers(child->getDOP()) {}

  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override { return wantedFields; }

  [[nodiscard]] DegreeOfParallelism getDOP() const override;

  GeneralizedRouterConsumer *appendConsumer(DeviceType target_device,
                                            DegreeOfParallelism dop,
                                            std::unique_ptr<Affinitizer> aff);

 protected:
  /**
   * Aquire a buffer from a free_pool
   * @param target The free pool to aquire a buffer from
   * @param polling If false, the function will block until a buffer is
   * available. If true, will return nullptr if the target free pool is empty
   * @param groupId unsure why needed
   * @return nullptr or a buffer
   */
  [[nodiscard]] virtual proteus::managed_ptr acquireBufferGeneralized(
      int target, bool polling, int64_t groupId);
  /**
   * Release a buffer to the target ready queue
   * @param target target queue
   * @param buff buffer previously acquired from the target free pool
   */
  virtual void releaseBufferGeneralized(int target, proteus::managed_ptr buff);
  virtual void freeBufferGeneralized(int target, proteus::managed_ptr buff);
  virtual bool get_readyGeneralized(int target, proteus::managed_ptr &buff);

  friend void *acquireBufferGeneralized(int target, GeneralizedRouter *xch,
                                        int64_t groupId);
  friend void *try_acquireBufferGeneralized(int target, GeneralizedRouter *xch,

                                            int64_t groupId);
  friend void releaseBufferGeneralized(int target, GeneralizedRouter *xch,
                                       void *buff);

  void produce_(OlapParallelContext *context) override;

  void produceForConsumer(const GeneralizedRouterConsumer &cons,
                          OlapParallelContext *context);

  virtual void open(Pipeline *pip);
  /**
   * helper function to create queues
   * Assumes init_mutex is held when called.
   */
  virtual void create_queues();
  virtual void *allocate_buffers_for_queue(size_t queue);
  virtual void close(Pipeline *pip);

  virtual llvm::Value *createTaskDescription(OlapParallelContext *context,
                                             const OperatorState &childState);

  [[nodiscard]] virtual size_t getNumberOfQueues() const;

  std::unique_ptr<routing::RoutingPolicy> getPolicy(
      GeneralizedRoutingPolicy p, DegreeOfParallelism dop,
      const std::vector<RecordAttribute *> &wantedFields);

  friend class GeneralizedRouterConsumer;
};

}  // namespace proteus

#endif /* PROTEUS_GENERALIZED_ROUTER_HPP */
