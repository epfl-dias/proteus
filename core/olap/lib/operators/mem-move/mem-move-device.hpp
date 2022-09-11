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
#ifndef MEM_MOVE_DEVICE_HPP_
#define MEM_MOVE_DEVICE_HPP_

#include <future>
#include <olap/util/parallel-context.hpp>
#include <platform/memory/managed-pointer.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/async_containers.hpp>
#include <thread>

#include "lib/operators/operators.hpp"

struct buff_pair {
  proteus::managed_ptr new_buff;
  proteus::managed_ptr old_buff;

  static buff_pair not_moved(proteus::managed_ptr buff);
  [[nodiscard]] bool moved() const;
};

class MemMoveDevice : public experimental::UnaryOperator {
 public:
  struct workunit {
    void *data;
    cudaEvent_t event;
    [[maybe_unused]] bool unused;  // FIXME: remove
  };

  /**
   * Between opening and closing of a pipeline MemMoveDevice itself is largely
   * stateless. All of the state is instead stored in MemMoveConf.
   */
  class MemMoveConf {
   public:
    // Idle workunits ready to be aquired
    AsyncQueueSPSC<workunit *>
        idle;  //_lockfree is slower and seems to have a bug
    /**
     * tran holds workunits that have been propogated (pushed) while they wait
     * to be reaped with getPropagated (pulled)
     */
    AsyncQueueSPSC<workunit *> tran;
    /**
     * Handle for the thread catching propagated workunits and calling `pull`
     */
    std::future<void> worker;
    cudaStream_t strm;
    cudaEvent_t *lastEvent;
    // The number of workunits in circulation
    size_t slack;
    size_t cnt = 0;
    // The pinned data backing the workunits
    void *data_buffs;

   public:
    virtual ~MemMoveConf() = default;

    virtual buff_pair push(proteus::managed_ptr src, size_t bytes,
                           int target_device, uint64_t srcServer);
    virtual proteus::managed_ptr force_push(const proteus::managed_ptr &src,
                                            size_t bytes, int target_device,
                                            uint64_t srcServer,
                                            cudaStream_t movestrm);
    virtual proteus::managed_ptr pull(proteus::managed_ptr buff) {
      return buff;
    }

    /**
     * acquire an idle workunit
     * @return a workunit
     */
    virtual MemMoveDevice::workunit *acquire();
    /**
     * Propagate a workunit onto the tran queue so that he can be caught with
     * `getPropagated` The work unit must be previously obtained with `acquire`
     */
    virtual void propagate(MemMoveDevice::workunit *buff, bool is_noop);

    /**
     * This will block until either a workunit has been obtained, or the
     * operator is terminating.
     * @param ret pointer to a workunit pointer which will be set to the value
     * of a propagated workunit.
     * @return true if a workunit has been obtained, and false if there are no
     * more workunits to obtain and the operator is terminating.
     */
    virtual bool getPropagated(MemMoveDevice::workunit **ret);

    /**
     * Release a workunit into the idle pool
     * @param buff a workunit
     */
    virtual void release(MemMoveDevice::workunit *buff);
  };

  MemMoveDevice(Operator *const child,
                const vector<RecordAttribute *> &wantedFields, size_t slack,
                bool to_cpu, std::vector<bool> do_transfer)
      : UnaryOperator(child),
        wantedFields(wantedFields),
        slack(slack),
        to_cpu(to_cpu),
        do_transfer(std::move(do_transfer)) {}

  MemMoveDevice(Operator *const child,
                const vector<RecordAttribute *> &wantedFields, size_t slack,
                bool to_cpu)
      : MemMoveDevice(child, wantedFields, slack, to_cpu,
                      //                      {true, true, false, false}
                      std::vector<bool>(wantedFields.size(), true)) {}

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override { return wantedFields; }

 protected:
  [[nodiscard]] virtual MemMoveConf *createMoveConf() const;
  virtual void destroyMoveConf(MemMoveConf *mmc) const;

  [[nodiscard]] virtual ProteusValueMemory getServerId(
      ParallelContext *context, const OperatorState &childState) const;

  virtual void genReleaseOldBuffer(ParallelContext *context,
                                   llvm::Value *pValue) const;

  const vector<RecordAttribute *> wantedFields;
  StateVar device_id_var;
  StateVar memmvconf_var;

  PipelineGen *catch_pip;
  llvm::Type *data_type;

  size_t slack;
  bool to_cpu;

  std::vector<bool> do_transfer;

  void catcher(MemMoveConf *conf, int group_id, exec_location target_dev,
               const void *session);

 protected:
  [[nodiscard]] int getTargetDevice() const;
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);
};

extern "C" {
MemMoveDevice::workunit *acquireWorkUnit(
    MemMoveDevice::MemMoveConf *mmc) noexcept;
}

#endif /* MEM_MOVE_DEVICE_HPP_ */
