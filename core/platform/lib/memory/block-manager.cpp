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

#include <platform/memory/block-manager.hpp>

template <>
size_t *buffer_manager<int32_t>::h_size;

template <>
void **buffer_manager<int32_t>::h_h_buff_start;

template <>
void **buffer_manager<int32_t>::h_buff_start;

template <>
void **buffer_manager<int32_t>::h_buff_end;

void BlockManager::reg(MemoryRegistry &registry) {
  for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
    size_t bytes = block_size * h_size[cpu.id];
    registry.reg(h_h_buff_start[cpu.id], bytes);
  }

  for (const auto &gpu : topology::getInstance().getGpus()) {
    size_t bytes =
        ((char *)h_buff_end[gpu.id]) - ((char *)h_buff_start[gpu.id]);
    registry.reg(h_buff_start[gpu.id], bytes);
  }
}

void BlockManager::unreg(MemoryRegistry &registry) {
  for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
    registry.unreg(BlockManager::h_h_buff_start[cpu.id]);
  }

  for (const auto &gpu : topology::getInstance().getGpus()) {
    registry.unreg(h_buff_start[gpu.id]);
  }
}

proteus::managed_ptr BlockManager::get_buffer(decltype(__builtin_FILE()) file,
                                              decltype(__builtin_LINE()) line) {
  return proteus::managed_ptr{buffer_manager<int32_t>::get_buffer()};
}

proteus::managed_ptr BlockManager::get_buffer_numa(
    const topology::cpunumanode &cpu) {
  return proteus::managed_ptr{buffer_manager<int32_t>::get_buffer_numa(cpu)};
}

proteus::managed_ptr BlockManager::h_get_buffer(int dev) {
  return proteus::managed_ptr{buffer_manager<int32_t>::h_get_buffer(dev)};
}

proteus::managed_ptr BlockManager::share_host_buffer(
    const proteus::managed_ptr &p) {
  buffer_manager<int32_t>::share_host_buffer((int32_t *)p.get());
  return proteus::managed_ptr{p.get()};
}
