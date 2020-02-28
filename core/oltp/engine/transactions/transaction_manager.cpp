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

#include "transactions/transaction_manager.hpp"

#include "utils/utils.hpp"

namespace txn {

/*
  static inline uint64_t __attribute__((always_inline)) read_tsc(uint8_t wid) {
    uint32_t a, d;
    __asm __volatile("rdtsc" : "=a"(a), "=d"(d));

    return ((((uint64_t)a) | (((uint64_t)d) << 32)) & 0x00FFFFFFFFFFFFFF) |
           (((uint64_t)wid) << 56);

    // return (((uint64_t)((d & 0x00FFFFFF) | (((uint32_t)wid) << 24))) << 32) |
    //       ((uint64_t)a);
  }

inline uint64_t __attribute__((always_inline)) get_next_xid(uint8_t wid) {
  // Global Atomic
  // return ++g_xid;

  // WorkerID + timestamp
  // thread_local std::chrono::time_point<std::chrono::system_clock,
  //                                     std::chrono::nanoseconds>
  //    curr;

  // curr = std::chrono::system_clock::now().time_since_epoch().count();

  // uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                   std::chrono::system_clock::now().time_since_epoch())
  //                   .count();
  // uint64_t cc = ((now << 8) >> 8) + (((uint64_t)wid) << 56);
  // uint64_t cc = (now & 0x00FFFFFFFFFFFFFF) + (((uint64_t)wid) << 56);

  uint64_t now = read_tsc(wid);
  // uint64_t cc = now + (((uint64_t)wid) << 56);

  // std::cout << "NOW:" << now << "|cc:" << cc << std::endl;
  return now;

  // return 0;
}

uint64_t rdtscl() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}*/

}  // namespace txn
