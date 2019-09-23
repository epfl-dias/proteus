/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "memory/block-manager.hpp"

template <>
size_t buffer_manager<int32_t>::h_size;

template <>
void **buffer_manager<int32_t>::h_h_buff_start;

void BlockManager::reg(MemoryRegistry &registry) {
  size_t bytes = block_size * h_size;
  for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
#ifndef NDEBUG
    auto ptr =
#endif
        registry.reg(h_h_buff_start[cpu.id], bytes);
    assert(ptr == h_h_buff_start[cpu.id]);
  }

  // TODO: also register GPU memory
}

void BlockManager::unreg(MemoryRegistry &registry) {
  for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
    registry.unreg(BlockManager::h_h_buff_start[cpu.id]);
  }

  // TODO: also unregister GPU memory
}
