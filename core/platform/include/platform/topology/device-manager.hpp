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

#ifndef DEVICE_MANAGER_HPP_
#define DEVICE_MANAGER_HPP_

#include <platform/topology/topology.hpp>

class DeviceManager {
 private:
  DeviceManager() = default;

 public:
  DeviceManager(DeviceManager &&) noexcept = delete;
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(DeviceManager &&) noexcept = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;

  ~DeviceManager() = default;

  static DeviceManager &getInstance();

  const topology::cpunumanode &getAvailableCPUNumaNode(const void *,
                                                       size_t cpu_req);
  const topology::core &getAvailableCPUCore(const void *, size_t cpu_req);
  const topology::gpunode &getAvailableGPU(const void *, size_t gpu_req);
};

#endif /* DEVICE_MANAGER_HPP_ */
