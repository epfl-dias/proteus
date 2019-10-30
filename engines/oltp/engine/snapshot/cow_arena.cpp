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

#include "cow_arena.hpp"

#include <glog/logging.h>
#include <sys/mman.h>

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace aeolus {
namespace snapshot {

void COWProvider::handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  //   printf("\nAttempt to access memory at address %p\n", siginfo->si_addr);
  // #ifdef LINUX_64BIT
  //   printf("Instruction pointer: %p\n",
  //          (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
  // #elif defined(LINUX_32BIT)
  //   printf("Instruction pointer: %p\n",
  //          (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
  // #endif
  void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

  const auto &instance = getInstance(addr);
  auto offset = ((int8_t *)addr) - instance->start;

  // auto f = mremap(src2 + offset, page_size, page_size,
  //                 MREMAP_FIXED | MREMAP_MAYMOVE, addr);
  // assert(f != MAP_FAILED);
  // assert(f == addr);
  std::memcpy(instance->save_to + offset, addr, page_size);

  mprotect(addr, page_size, PROT_READ | PROT_WRITE);
  // signal(sig, SIG_DFL);
  // raise(sig);
#pragma clang diagnostic pop
}

COWArena::COWArena(size_t size, guard) : size_bytes(size) {
  assert(size_bytes && "Requested empty arena");
  shm_fd = memfd_create("cow", 0);
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

  // preallocate some consecutive memory so that we can overwrite it
  void *prealloc = mmap(nullptr, 2 * size_bytes, PROT_WRITE | PROT_READ,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  // Mark MAP_SHARED so that we can map the same memory multiple times
  oltp_arena = (int *)mmap(prealloc, size_bytes, PROT_WRITE | PROT_READ,
                           MAP_SHARED | MAP_FIXED, shm_fd, 0);
  assert(oltp_arena != MAP_FAILED);
  start = (int8_t *)oltp_arena;
}

COWArena::~COWArena() { COWProvider::remove(oltp_arena); }

void COWProvider::init() {
  page_size = getpagesize();
  // Create snapshot

  // 0. Install handler to lazily steal and save data to-be-overwritten by
  // OLTP
  struct sigaction act {};
  sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
  act.sa_sigaction = COWProvider::handler;
#pragma clang diagnostic pop
  act.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigaction(SIGSEGV, &act, nullptr);
}

void COWArena::create_snapshot_() {
  // Should we mprotect before or after?
  // ANSWER:
  //    If we mprotect after, we may loose some updates!
  //    If we mprotect before, our handler may not know where to copy some
  //    updates!

  // 1. Mark OLTP side as RO
  assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

  // 2. Create a second mapping of the OLTP data for the OLAP side
  // 2.i   The mapping can not be MAP_SHARED as we do not want to create
  // changes private to the OLTP side.
  // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
  // side, we have to catch OLTP writes, still the underlying data and
  // replicate the, to the OLAP side.
  olap_arena =
      (int *)mmap(start + size_bytes, size_bytes, PROT_WRITE | PROT_READ,
                  MAP_PRIVATE | MAP_FIXED, shm_fd, 0);
  assert(olap_arena != MAP_FAILED);
  save_to = (int8_t *)olap_arena;
  // return olap_arena;
}

void COWArena::destroy_snapshot_() {
  // time_block t{"T2: "};
  // Throw away snapshot
  munmap(olap_arena, size_bytes);
}

size_t COWProvider::page_size;
std::map<void *, COWArena *, std::greater<>> COWProvider::instances;

}  // namespace snapshot
}  // namespace aeolus
