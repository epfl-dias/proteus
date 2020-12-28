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
#include <olap/util/parallel-context.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/async_containers.hpp>
#include <thread>

#include "lib/operators/operators.hpp"
#include "mem-move-device.hpp"

class MemMoveLocalTo : public MemMoveDevice {
 public:
  class MemMoveConf : public MemMoveDevice::MemMoveConf {
   public:
    void propagate(MemMoveDevice::workunit *buff, bool is_noop) override;

    buff_pair push(proteus::managed_ptr src, size_t bytes, int target_device,
                   uint64_t srcServer) override;

    bool getPropagated(MemMoveDevice::workunit **ret) override;
  };

 public:
  MemMoveLocalTo(Operator *const child,
                 const vector<RecordAttribute *> &wantedFields,
                 size_t slack = 8)
      : MemMoveDevice(child, wantedFields, slack, true) {}

  [[nodiscard]] MemMoveConf *createMoveConf() const override;
};

#endif /* MEM_MOVE_LOCAL_TO_HPP_ */
