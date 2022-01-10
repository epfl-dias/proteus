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
#include <platform/memory/managed-pointer.hpp>
#include <thread>
#include <unordered_map>

#include "mem-broadcast-device.hpp"

class MemBroadcastScaleOut : public MemBroadcastDevice {
 public:
  class MemBroadcastConf : public MemBroadcastDevice::MemBroadcastConf {
   public:
    void propagateBroadcast(MemMoveDevice::workunit *buff,
                            int target_device) override;

    buff_pair pushBroadcast(const proteus::managed_ptr &src, size_t bytes,
                            int target_device, bool disable_noop) override;

    proteus::managed_ptr pull(proteus::managed_ptr buff) override;

    bool getPropagated(MemMoveDevice::workunit **ret) override;
  };

  MemBroadcastScaleOut(Operator *const child, ParallelContext *const context,
                       const vector<RecordAttribute *> &wantedFields,
                       int num_of_targets, bool to_cpu,
                       bool always_share = false)
      : MemBroadcastDevice(child, wantedFields, num_of_targets, to_cpu,
                           always_share) {}

  [[nodiscard]] MemBroadcastConf *createMoveConf() const override;
};

#endif /* MEM_BROADCAST_SCALEOUT_HPP_ */
