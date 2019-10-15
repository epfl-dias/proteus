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
#include "device-manager.hpp"

DeviceManager &DeviceManager::getInstance() {
  static DeviceManager instance;
  return instance;
}

const topology::cpunumanode &DeviceManager::getAvailableCPUNumaNode(
    void *, size_t cpu_req) {
  return topology::getInstance()
      .getCpuNumaNodes()[cpu_req %
                         topology::getInstance().getCpuNumaNodeCount()];
}

const topology::core &DeviceManager::getAvailableCPUCore(void *,
                                                         size_t cpu_req) {
  return topology::getInstance()
      .getCores()[cpu_req % topology::getInstance().getCoreCount()];
}

const topology::gpunode &DeviceManager::getAvailableGPU(void *,
                                                        size_t gpu_req) {
  return topology::getInstance()
      .getGpus()[gpu_req % topology::getInstance().getGpuCount()];
}
