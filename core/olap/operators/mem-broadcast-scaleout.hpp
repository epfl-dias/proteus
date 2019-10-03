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
#ifndef MEM_BROADCAST_SCALEOUT_HPP_
#define MEM_BROADCAST_SCALEOUT_HPP_

#include <future>
#include <thread>
#include <unordered_map>

#include "operators/mem-broadcast-device.hpp"

class MemBroadcastScaleOut : public MemBroadcastDevice {
 public:
  class MemBroadcastConf : public MemBroadcastDevice::MemBroadcastConf {
   public:
    virtual void propagateBroadcast(MemMoveDevice::workunit *buff,
                                    int target_device);

    virtual buff_pair pushBroadcast(void *src, size_t bytes, int target_device,
                                    bool disable_noop);

    virtual void *pull(void *buff);

    virtual bool getPropagated(MemMoveDevice::workunit **ret);
  };

  MemBroadcastScaleOut(Operator *const child, ParallelContext *const context,
                       const vector<RecordAttribute *> &wantedFields,
                       int num_of_targets, bool to_cpu,
                       bool always_share = false)
      : MemBroadcastDevice(child, context, wantedFields, num_of_targets, to_cpu,
                           always_share) {}

  virtual MemBroadcastConf *createMoveConf() const;
};

#endif /* MEM_BROADCAST_SCALEOUT_HPP_ */
