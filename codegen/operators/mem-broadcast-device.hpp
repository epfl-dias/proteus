/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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
#ifndef MEM_BROADCAST_DEVICE_HPP_
#define MEM_BROADCAST_DEVICE_HPP_

#include <future>
#include <thread>
#include <unordered_map>
#include "codegen/util/parallel-context.hpp"
#include "operators/operators.hpp"
#include "topology/affinity_manager.hpp"
#include "util/async_containers.hpp"

// void * make_mem_move_device(char * src, size_t bytes, int target_device,
// cudaStream_t strm);

class MemBroadcastDevice : public UnaryOperator {
 public:
  struct workunit {
    void *data;
    cudaEvent_t event;
    // cudaStream_t strm;
  };

  struct MemMoveConf {
    AsyncQueueSPSC<workunit *> idle;
    AsyncQueueSPSC<workunit *> tran;

    std::future<void> worker;
    std::unordered_map<int, cudaStream_t> strm;

    int num_of_targets;
    bool to_cpu;
    bool always_share;

    void *targetbuffer[16];

    // cudaStream_t                strm2    ;

    // cudaEvent_t               * lastEvent;

    // size_t                      slack    ;
    // // cudaEvent_t               * events   ;
    // void                     ** old_buffs;
    // size_t                      next_e   ;
  };

  MemBroadcastDevice(Operator *const child, ParallelContext *const context,
                     const vector<RecordAttribute *> &wantedFields,
                     int num_of_targets, bool to_cpu, bool always_share = false)
      : UnaryOperator(child),
        context(context),
        wantedFields(wantedFields),
        slack(8 * num_of_targets),
        to_cpu(to_cpu),
        always_share(always_share) {
    for (int i = 0; i < num_of_targets; ++i) {
      targets.push_back(
          always_share ? 0 : i);  // FIXME: this is not correct, only to be used
                                  // for SSB-broadcast benchmark!!!!!!!!!
    }
  }

  virtual ~MemBroadcastDevice() {
    LOG(INFO) << "Collapsing MemBroadcastDevice operator";
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

  std::vector<int> targets;

  size_t slack;
  bool to_cpu;
  bool always_share;

  ParallelContext *const context;

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  void catcher(MemMoveConf *conf, int group_id,
               const exec_location &target_dev);
};

#endif /* MEM_BROADCAST_DEVICE_HPP_ */
