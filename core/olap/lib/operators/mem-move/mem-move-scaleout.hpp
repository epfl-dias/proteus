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
#ifndef MEM_MOVE_SCALEOUT_HPP_
#define MEM_MOVE_SCALEOUT_HPP_

#include <future>
#include <thread>

#include "mem-move-device.hpp"
#include "olap/util/parallel-context.hpp"

class MemMoveScaleOut : public MemMoveDevice {
 public:
  class MemMoveConf : public MemMoveDevice::MemMoveConf {
    std::queue<void *> pending;
    std::mutex pending_m;

   public:
    void propagate(MemMoveDevice::workunit *buff, bool is_noop) override;

    buff_pair push(proteus::managed_ptr src, size_t bytes, int target_device,
                   uint64_t srcServer) override;
    proteus::managed_ptr pull(proteus::managed_ptr buff) override;

    bool getPropagated(MemMoveDevice::workunit **ret) override;
  };

  MemMoveScaleOut(Operator *const child,
                  const vector<RecordAttribute *> &wantedFields, size_t slack);

 protected:
  [[nodiscard]] MemMoveScaleOut::MemMoveConf *createMoveConf() const override;

  [[nodiscard]] ProteusValueMemory getServerId(
      ParallelContext *context, const OperatorState &childState) const override;

  void genReleaseOldBuffer(ParallelContext *context,
                           llvm::Value *src) const override;

  // virtual void open(Pipeline *pip);
  void close(Pipeline *pip) override;
};

#endif /* MEM_MOVE_SCALEOUT_HPP_ */
