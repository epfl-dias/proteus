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
#ifndef MEM_MOVE_LOCAL_TO_HPP_
#define MEM_MOVE_LOCAL_TO_HPP_

#include <future>
#include <thread>

#include "operators/operators.hpp"
#include "topology/affinity_manager.hpp"
#include "util/async_containers.hpp"
#include "util/parallel-context.hpp"

// void * make_mem_move_device(char * src, size_t bytes, int target_device,
// cudaStream_t strm);

class MemMoveLocalTo : public UnaryOperator {
 public:
  struct workunit {
    void *data;
    cudaEvent_t event;
  };

  struct MemMoveConf {
    AsyncStackSPSC<workunit *> idle;
    AsyncQueueSPSC<workunit *> tran;

    std::future<void> worker;
    cudaStream_t strm;
    cudaStream_t strm2;

    size_t slack;
    cudaEvent_t *events;
    void **old_buffs;
    size_t next_e;
  };

  MemMoveLocalTo(Operator *const child, ParallelContext *const context,
                 const vector<RecordAttribute *> &wantedFields,
                 size_t slack = 8)
      : UnaryOperator(child),
        context(context),
        wantedFields(wantedFields),
        slack(slack) {}

  virtual ~MemMoveLocalTo() {
    LOG(INFO) << "Collapsing MemMoveLocalTo operator";
  }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

 private:
  const vector<RecordAttribute *> wantedFields;
  size_t device_id_var;
  size_t memmvconf_var;

  PipelineGen *catch_pip;
  llvm::Type *data_type;

  size_t slack;

  ParallelContext *const context;

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  void catcher(MemMoveConf *conf, int group_id,
               const exec_location &target_dev);
};

#endif /* MEM_MOVE_LOCAL_TO_HPP_ */
