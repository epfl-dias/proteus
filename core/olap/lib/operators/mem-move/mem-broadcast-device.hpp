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
#ifndef MEM_BROADCAST_DEVICE_HPP_
#define MEM_BROADCAST_DEVICE_HPP_

#include <future>
#include <olap/util/parallel-context.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/async_containers.hpp>
#include <thread>
#include <unordered_map>

#include "mem-move-device.hpp"

// void * make_mem_move_device(char * src, size_t bytes, int target_device,
// cudaStream_t strm);

class MemBroadcastDevice : public MemMoveDevice {
 public:
  class MemBroadcastConf : public MemMoveDevice::MemMoveConf {
   public:
    std::unordered_map<int, cudaStream_t> strm;

    int num_of_targets;
    bool to_cpu;
    bool always_share;

    void *targetbuffer[16];

    virtual void prepareNextBatch();

    virtual void propagateBroadcast(MemMoveDevice::workunit *buff,
                                    int target_device);

    virtual buff_pair pushBroadcast(void *src, size_t bytes, int target_device,
                                    bool disable_noop);
    //  virtual buff_pair push(void *src, size_t bytes, int target_device);
    //  virtual void *pull(void *buff) { return buff; }
  };

  MemBroadcastDevice(Operator *const child,
                     const vector<RecordAttribute *> &wantedFields,
                     int num_of_targets, bool to_cpu, bool always_share = false)
      : MemMoveDevice(child, wantedFields, 8 * num_of_targets, to_cpu),
        always_share(always_share) {
    for (int i = 0; i < num_of_targets; ++i) {
      targets.push_back(
          always_share ? 0 : i);  // FIXME: this is not correct, only to be used
                                  // for SSB-broadcast benchmark!!!!!!!!!
    }
  }

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] RecordType getRowType() const override {
    std::string relName = wantedFields[0]->getRelationName();

    std::vector<RecordAttribute *> ret;
    ret.reserve(wantedFields.size() + 1);
    for (const auto &f : wantedFields) {
      ret.emplace_back(new RecordAttribute{*f});
      assert(dynamic_cast<const BlockType *>(ret.back()->getOriginalType()));
    }

    auto *reg_as =
        new RecordAttribute(relName, "__broadcastTarget", new IntType());
    ret.emplace_back(reg_as);
    return ret;
  }

  [[nodiscard]] MemBroadcastConf *createMoveConf() const override;

 private:
  std::vector<int> targets;

  bool always_share;

  void open(Pipeline *pip) override;
  void close(Pipeline *pip) override;
};

#endif /* MEM_BROADCAST_DEVICE_HPP_ */
