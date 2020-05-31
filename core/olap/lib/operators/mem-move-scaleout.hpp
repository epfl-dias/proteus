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
    virtual void propagate(MemMoveDevice::workunit *buff, bool is_noop);

    virtual buff_pair push(void *src, size_t bytes, int target_device,
                           uint64_t srcServer);
    virtual void *pull(void *buff);

    virtual bool getPropagated(MemMoveDevice::workunit **ret);
  };

  MemMoveScaleOut(Operator *const child, ParallelContext *const context,
                  const vector<RecordAttribute *> &wantedFields, size_t slack)
      : MemMoveDevice(child, context, wantedFields, slack, true) {}

 protected:
  virtual MemMoveScaleOut::MemMoveConf *createMoveConf() const;

  // virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);
};

#endif /* MEM_MOVE_SCALEOUT_HPP_ */
