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
#include <thread>

#include "lib/operators/operators.hpp"
#include "olap/util/parallel-context.hpp"
#include "topology/affinity_manager.hpp"
#include "util/async_containers.hpp"

struct buff_pair {
  void *new_buff;
  void *old_buff;

  static buff_pair not_moved(void *buff);
};

class MemMoveDevice : public experimental::UnaryOperator {
 public:
  struct workunit {
    void *data;
    cudaEvent_t event;
    // cudaStream_t strm;
  };

  class MemMoveConf {
   public:
    AsyncQueueSPSC<workunit *>
        idle;  //_lockfree is slower and seems to have a bug
    AsyncQueueSPSC<workunit *> tran;

    std::future<void> worker;
    // std::thread               * worker   ;
    cudaStream_t strm;
    // cudaStream_t                strm2    ;

    cudaEvent_t *lastEvent;

    size_t slack;
    // cudaEvent_t               * events   ;
    // void                     ** old_buffs;
    size_t next_e;
    size_t cnt = 0;

    void *data_buffs;
    workunit *wus;

   public:
    virtual ~MemMoveConf() = default;

    virtual MemMoveDevice::workunit *acquire();
    virtual void propagate(MemMoveDevice::workunit *buff, bool is_noop);

    virtual buff_pair push(void *src, size_t bytes, int target_device,
                           uint64_t srcServer);
    virtual void *pull(void *buff) { return buff; }

    virtual bool getPropagated(MemMoveDevice::workunit **ret);
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

  const vector<RecordAttribute *> wantedFields;
  StateVar device_id_var;
  StateVar memmvconf_var;

  PipelineGen *catch_pip;
  llvm::Type *data_type;

  size_t slack;
  bool to_cpu;

  std::vector<bool> do_transfer;

  void catcher(MemMoveConf *conf, int group_id, exec_location target_dev);

 protected:
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);
};

#endif /* MEM_MOVE_DEVICE_HPP_ */
