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

#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

namespace txn {

uint64_t rdtscl() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// void TransactionManager::init() {
//   std::cout << "Initializing Txn Manager..." << std::endl;

//   // template <class Duration>
//   // using sys_time = std::chrono::time_point<std::chrono::system_clock,
//   // Duration>; using sys_nanoseconds = sys_time<std::chrono::nanoseconds>;

//   // timed_func::interval_runner([this] { this->switch_master(); },
//   //                            global_conf::time_master_switch_ms);

//   // timed_func::interval_runner(switch_master, )
// }

// void TransactionManager::switch_master() {
//   // TODO: Master as the most significant short of the TXN ID.

//   // ushort curr_master = g_xid.load() >> 48;
//   // uint64_t new_master = (curr_master + 1) %
//   global_conf::num_master_versions;
//   // ushort neww = g_xid.fetch_and(new_master << 48) >> 48;
//   // ushort bc = g_xid.load() >> 48;
//   // std::cout << "Old Master: " << curr_master << "| New Master: " <<
//   // new_master
//   //           << "| Actual New: " << bc << std::endl;
// }
// void TransactionManager::gc() {}

}  // namespace txn
