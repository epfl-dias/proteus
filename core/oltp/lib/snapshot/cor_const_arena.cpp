/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include "oltp/snapshot/cor_const_arena.hpp"

#include <glog/logging.h>
#include <sys/mman.h>

#include <cassert>
#include <cstring>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <stdexcept>
#include <string>

namespace aeolus {
namespace snapshot {

void CORConstProvider::handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  // printf("\nAttempt to access memory at address %p\n", siginfo->si_addr);
  // #ifdef LINUX_64BIT
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
  // #elif defined(LINUX_32BIT)
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
  // #endif

  void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

  const auto &instance = getInstance(addr);

  if (addr >= instance->olap_start &&
      addr < instance->olap_start + instance->size_bytes) {
    auto offset = (((int8_t *)addr) - instance->olap_start) / page_size;

    mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    mprotect(instance->start + offset * page_size, page_size, PROT_READ);

    std::memcpy(addr, instance->start + offset * page_size, page_size);
    uint64_t mask = ((uint64_t)1) << (offset % 64);
    assert(offset / 64 < instance->dirty_segs);
    instance->dirty[offset / 64] &= ~mask;
    instance->new_dirty[offset / 64] |= mask;

    // fix_olap((int8_t *)addr);
  } else {
    auto offset = (((int8_t *)addr) - instance->start) / page_size;

    mprotect(instance->olap_start + offset * page_size, page_size,
             PROT_READ | PROT_WRITE);
    mprotect(addr, page_size, PROT_READ | PROT_WRITE);

    uint64_t mask = ((uint64_t)1) << (offset % 64);
    assert(offset / 64 < instance->dirty_segs);
    if (!(instance->new_dirty[offset / 64] & mask)) {
      std::memcpy(instance->olap_start + offset * page_size, addr, page_size);
    }
    instance->dirty[offset / 64] |= mask;
  }
#pragma clang diagnostic pop
}

CORConstArena::CORConstArena(size_t size, guard g)
    : size_bytes(size),
      olap_arena((int *)MemoryManager::mallocPinned(2 * size)),
      oltp_arena((int *)(((int8_t *)olap_arena) + size)) {
  shm_fd = memfd_create("CORConstProvider", 0);
  if (shm_fd < 0) {
    auto msg = std::string{"memfd failed"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
  if (ftruncate(shm_fd, size_bytes) < 0) {
    auto msg = std::string{"ftruncate failed"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
  size_t num_of_pages = size_bytes / CORConstProvider::page_size;

  dirty_segs = (num_of_pages + 63) / 64;
  dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
  new_dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));

  olap_start = (int8_t *)olap_arena;

  start = (int8_t *)oltp_arena;
  // Create snapshot
}

CORConstArena::~CORConstArena() {
  CORConstProvider::remove(olap_arena);
  MemoryManager::freePinned(olap_arena);
}

void CORConstProvider::init() {
  page_size = getpagesize();

  // 0. Install handler to lazily steal and save data to-be-overwritten by
  // OLTP
  struct sigaction act {};
  sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  act.sa_sigaction = CORConstProvider::handler;
#pragma clang diagnostic pop
  act.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigaction(SIGSEGV, &act, nullptr);
}

void CORConstArena::create_snapshot_() {
  // Should we mprotect before or after?
  // ANSWER:
  //    If we mprotect after, we may loose some updates!
  //    If we mprotect before, our handler may not know where to copy some
  //    updates!

  // 1. Mark OLTP side as ROdirty
  // assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

  // TODO: fix dirty pages

  // 2. Create a second mapping of the OLTP data for the OLAP side
  // 2.i   The mapping can not be MAP_SHARED as we do not want to create
  // changes private to the OLTP side.
  // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
  // side, we have to catch OLTP writes, still the underlying data and
  // replicate the, to the OLAP side.
  // assert(mprotect(olap_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
  assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);
  for (size_t i = 0; i < dirty_segs; ++i) {
    new_dirty[i] = 0;
    uint64_t d = dirty[i];
    while (d) {
      uint64_t bit = d & -d;
      d ^= bit;
      size_t bit_index = __builtin_popcountll(bit - 1);
      size_t page_offset = i * 64 + bit_index;
      size_t offset = page_offset * CORConstProvider::page_size;
      assert(mprotect(((int8_t *)olap_arena) + offset,
                      CORConstProvider::page_size,
                      // PROT_WRITE would be sufficient and prefered, but on
                      // some architectures in also implies PROT_READ!
                      PROT_NONE) >= 0);
      // assert(mprotect(((int8_t *)oltp_arena) + offset, page_size,
      //                 // PROT_WRITE would be sufficient and prefered, but
      //                 on
      //                 // some architectures in also implies PROT_READ!
      //                 PROT_READ) >= 0);
    }
  }

  // return {olap_arena, oltp_arena};
}

size_t CORConstProvider::page_size;
std::map<void *, CORConstArena *, std::greater<>> CORConstProvider::instances;

}  // namespace snapshot
}  // namespace aeolus
