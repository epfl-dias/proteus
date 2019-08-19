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

#ifndef CC_HPP_
#define CC_HPP_

#include <iostream>
#include <mutex>
#include <vector>

#include "transactions/txn_utils.hpp"
#include "utils/spinlock.h"

/*
GC Notes:

Either maintain minimum active txn_id or maybe a mask of fewer significant bits
so you will end up updting the number less frequently. idk at the moment what
and how to do it.

*/

namespace txn {

class CC_MV2PL;
class CC_GlobalLock;

class CC_MV2PL {
 public:
  struct PRIMARY_INDEX_VAL {
    uint64_t t_min;  // transaction id that inserted the record
    uint64_t t_max;  // transaction id that deleted the row
    uint64_t VID;    // VID of the record in memory
    ushort last_master_ver;
    ushort delta_id;
    lock::Spinlock latch;
    std::atomic<bool> write_lck;

    PRIMARY_INDEX_VAL();
    PRIMARY_INDEX_VAL(uint64_t tid, uint64_t vid, ushort master_ver)
        : t_min(tid), t_max(0), VID(vid), last_master_ver(master_ver) {
      write_lck.store(0);
    }
  } __attribute__((aligned(64)));

  CC_MV2PL() {
    std::cout << "CC Protocol: MV2PL" << std::endl;
    // modified_vids.clear();
  }
  bool execute_txn(void *stmts, uint64_t xid, ushort curr_master,
                   ushort delta_ver);

  // TODO: this needs to be modified as we changed the format of TIDs
  static inline bool __attribute__((always_inline))
  is_readable(uint64_t tmin, uint64_t tmax, uint64_t tid) {
    // FIXME: the following is wrong as we have encoded the worker_id in the
    // txn_id. the comparision should be of the xid only and if same then idk
    // because two threads can read_tsc at the same time. it doesnt mean thread
    // with lesser ID comes first.

    // TXN ID= ((txn_id << 8) >> 8) ?? cant we just AND to clear top bits?
    // WORKER_ID = (txn_id >> 56)

    uint64_t w_tid = tid & 0x00FFFFFFFFFFFFFF;
    uint64_t w_tmin = tmin & 0x00FFFFFFFFFFFFFF;
    uint64_t w_tmax = tmax & 0x00FFFFFFFFFFFFFF;

    // if (w_tmax != 0 && w_tid > w_tmax){
    //   return -1;
    // } else if (w_tid >= w_tmin) && (w_tmax == 0 || w_tid < w_tmax){
    //   return 1;
    // } else {
    //   return 0;
    // }
    assert(w_tmin != w_tid);

    if ((w_tid >= w_tmin) && (w_tmax == 0 || w_tid < w_tmax)) {
      return true;
    } else {
      return false;
    }

    // if ((tid >= tmin) && (tmax == 0 || tid < tmax)) {
    //   return true;
    // } else {
    //   return false;
    // }
  }

  static bool is_mv() { return true; }
  static inline void __attribute__((always_inline)) release_locks(
      std::vector<CC_MV2PL::PRIMARY_INDEX_VAL *> &hash_ptrs_lock_acquired) {
    for (auto c : hash_ptrs_lock_acquired) c->write_lck = false;
  }

 private:
};

struct VERSION {
  uint64_t t_min;
  uint64_t t_max;
  void *data;
  VERSION *next;
  VERSION(uint64_t t_min, uint64_t t_max, void *data)
      : t_min(t_min), t_max(t_max), data(data), next(nullptr) {}
};

struct VERSION_LIST {
  VERSION *head;
  short master_version;

  VERSION_LIST() { head = nullptr; }

  void insert(VERSION *val) {
    {
      val->next = head;
      head = val;
    }
  }

  void *get_readable_ver(uint64_t tid_self) {
    VERSION *tmp = nullptr;
    {
      tmp = head;
      // C++ standard says that (x == NULL) <=> (x==nullptr)
      while (tmp != nullptr) {
        if (CC_MV2PL::is_readable(tmp->t_min, tmp->t_max, tid_self)) {
          return tmp->data;
        } else {
          tmp = tmp->next;
        }
      }
    }
    return nullptr;
  }

  // void print_list(uint64_t print) {
  //   VERSION *tmp = head;
  //   while (tmp != nullptr) {
  //     std::cout << "[" << print << "] xmin:" << tmp->t_min << std::endl;
  //     tmp = tmp->next;
  //   }
  // }
} __attribute__((aligned(64)));

class CC_GlobalLock {
 public:
  struct PRIMARY_INDEX_VAL {
    uint64_t VID;
    short last_master_ver;
    lock::Spinlock latch;
    PRIMARY_INDEX_VAL(uint64_t vid) : VID(vid), last_master_ver(0) {}
    PRIMARY_INDEX_VAL(uint64_t vid, short master_ver)
        : VID(vid), last_master_ver(master_ver) {}
    PRIMARY_INDEX_VAL(uint64_t tid, uint64_t vid, short master_ver)
        : VID(vid), last_master_ver(master_ver) {}
  } __attribute__((aligned(64)));

  bool execute_txn(void *stmts, uint64_t xid);

  CC_GlobalLock() {
    std::cout << "CC Protocol: GlobalLock" << std::endl;
    curr_master = 0;
  }
  static bool is_mv() { return false; }

 private:
  std::mutex global_lock;
  volatile short curr_master;
};

}  // namespace txn

#endif /* CC_HPP_ */
