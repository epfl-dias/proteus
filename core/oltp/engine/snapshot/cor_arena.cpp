/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#include "cor_arena.hpp"

#include <glog/logging.h>
#include <signal.h>
#include <sys/mman.h>

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace aeolus {
namespace snapshot {

void CORArena::fix_olap(int8_t *addr) {
  auto offset = (addr - olap_start) / page_size;
  mprotect(addr, page_size, PROT_READ | PROT_WRITE);
  std::memcpy(addr, find_at[offset], page_size);

  uint64_t mask = ((uint64_t)1) << (offset % 64);
  dirty[offset / 64] &= ~mask;
  find_at[offset] = nullptr;
}

void CORArena::handler2(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  //     printf("\nAttempt to access memory at address %p\n",
  //     siginfo->si_addr);
  // #ifdef LINUX_64BIT
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
  // #elif defined(LINUX_32BIT)
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
  // #endif

  void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

  if (addr >= olap_start && addr < olap_start + size_bytes) {
    fix_olap((int8_t *)addr);
  } else {
    auto offset = (((int8_t *)addr) - start) / page_size;
    uint64_t mask = ((uint64_t)1) << (offset % 64);
    assert(offset / 64 < dirty_segs);
    if (dirty[offset / 64] & mask) {
      fix_olap(((int8_t *)olap_start) + offset * page_size);
      mprotect(addr, page_size, PROT_READ);
    } else {
      // dirty[offset / 64] |= mask;
      mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    }
  }
#pragma clang diagnostic pop
}

void CORArena::handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  //     printf("\nAttempt to access memory at address %p\n",
  //     siginfo->si_addr);
  // #ifdef LINUX_64BIT
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
  // #elif defined(LINUX_32BIT)
  //     printf("Instruction pointer: %p\n",
  //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
  // #endif

  void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

  if (addr >= olap_start && addr < olap_start + size_bytes) {
    fix_olap((int8_t *)addr);
  } else {
    auto offset = (((int8_t *)addr) - start) / page_size;
    uint64_t mask = ((uint64_t)1) << (offset % 64);
    assert(offset / 64 < dirty_segs);
    new_dirty[offset / 64] |= mask;
    if (dirty[offset / 64] & mask) {
      {
        // auto offset = (addr - olap_start) / page_size;
        mprotect(addr, page_size, PROT_READ | PROT_WRITE);
        std::memcpy(addr, find_at[offset], page_size);

        // uint64_t mask = ((uint64_t)1) << (offset % 64);
        // dirty[offset / 64] &= ~mask;
      }
      // mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    } else {
      // dirty[offset / 64] |= mask;
      mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    }
  }
#pragma clang diagnostic pop
}

void CORArena::init(size_t size_bytes) {
  page_size = getpagesize();
  CORArena::size_bytes = size_bytes;
  shm_fd = memfd_create("CORArena", 0);
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

  size_t N = size_bytes / sizeof(int);
  size_t num_of_pages = size_bytes / page_size;

  dirty_segs = (num_of_pages + 63) / 64;
  dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
  new_dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
  find_at = (int8_t **)calloc(num_of_pages, sizeof(void *));

  // Mark MAP_SHARED so that we can map the same memory multiple times
  olap_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                           MAP_SHARED, shm_fd, 0);
  olap_start = (int8_t *)olap_arena;
  if (olap_arena == MAP_FAILED) {
    auto msg = std::string{"mmap failed"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  // Create snapshot

  // 0. Install handler to lazily steal and save data to-be-overwritten by
  // OLTP
  struct sigaction act {};
  sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  act.sa_sigaction = CORArena::handler;
#pragma clang diagnostic pop
  act.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigaction(SIGSEGV, &act, nullptr);
}

int *CORArena::oltp() { return oltp_arena; }
int *CORArena::olap() { return olap_arena; }

void CORArena::create_snapshot(void *place_at) {
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
  oltp_arena = (int *)mmap(place_at, size_bytes, PROT_READ,
                           MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
  assert(oltp_arena != MAP_FAILED);
  start = (int8_t *)oltp_arena;
  for (size_t i = 0; i < dirty_segs; ++i) {
    uint64_t d = dirty[i];
    while (d) {
      uint64_t bit = d & -d;
      d ^= bit;
      size_t bit_index = __builtin_popcountll(bit - 1);
      size_t page_offset = i * 64 + bit_index;
      size_t offset = page_offset * page_size;
      assert(mprotect(((int8_t *)oltp_arena) + offset, page_size,
                      // PROT_WRITE would be sufficient and prefered, but on
                      // some architectures in also implies PROT_READ!
                      PROT_NONE) >= 0);
    }
  }

  // return {olap_arena, oltp_arena};
}

void CORArena::destroy_snapshot() {
  // time_block t{"T2: "};
  // TODO the eager version: instead of the code below, apply the changes
  assert(mprotect(oltp_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
  for (size_t i = 0; i < dirty_segs; ++i) {
    dirty[i] |= new_dirty[i];
    uint64_t d = dirty[i];
    while (d) {
      uint64_t bit = d & -d;
      d ^= bit;
      size_t bit_index = __builtin_popcountll(bit - 1);
      size_t page_offset = i * 64 + bit_index;
      size_t offset = page_offset * page_size;
      if (new_dirty[i] & bit) {
        find_at[page_offset] = ((int8_t *)oltp_arena) + offset;
      }
      assert(mprotect(((int8_t *)olap_arena) + offset, page_size,
                      // PROT_WRITE would be sufficient and prefered, but on
                      // some architectures in also implies PROT_READ!
                      PROT_NONE) >= 0);
    }
    new_dirty[i] = 0;
  }
  // If we do not mremap oltp_arena, then the next oltp_arena will be in a
  // different virtual address

  // Throw away snapshot munmap(oltp_arena, size_bytes);
}

uint64_t *CORArena::dirty;
uint64_t *CORArena::new_dirty;
int8_t **CORArena::find_at;
int8_t *CORArena::olap_start;
size_t CORArena::size_bytes;
size_t CORArena::page_size;

int *CORArena::olap_arena;
int CORArena::shm_fd;
size_t CORArena::dirty_segs;
int8_t *CORArena::start;

}  // namespace snapshot
}  // namespace aeolus
