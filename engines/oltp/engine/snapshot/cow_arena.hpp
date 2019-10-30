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

#ifndef AEOLUS_SNAPSHOT_COW_ARENA_HPP_
#define AEOLUS_SNAPSHOT_COW_ARENA_HPP_

#include <signal.h>

#include <cstdint>
#include <map>
#include <memory>

#include "arena.hpp"

namespace aeolus {
namespace snapshot {

class COWArena : public Arena<COWArena> {
 protected:
  int shm_fd;
  size_t size_bytes;
  int8_t *save_to;
  int8_t *start;

  int *oltp_arena;
  int *olap_arena;

 protected:
  class guard {
   private:
    guard(int) {}

    friend class COWProvider;
  };

 public:
  COWArena(size_t size, guard g);
  ~COWArena();

  int *oltp() const { return oltp_arena; }
  int *olap() const { return olap_arena; }

 protected:
  void create_snapshot_();
  void destroy_snapshot_();

  friend class Arena<COWArena>;
  friend class COWProvider;
};

class COWProvider {
 public:
  static size_t page_size;
  static std::map<void *, COWArena *, std::greater<>> instances;

 private:
  static COWArena *getInstance(void *addr) {
    return instances.lower_bound(addr)->second;
  }

 private:
  static void remove(void *oltp) { instances.erase(oltp); }

 public:
  static void handler(int sig, siginfo_t *siginfo, void *uap);

 public:
  static void init();
  static void deinit();

  static std::unique_ptr<COWArena> create(size_t size) {
    size = ((size + page_size - 1) / page_size) * page_size;
    auto ptr = std::make_unique<COWArena>(size, COWArena::guard{5});
    instances.emplace(ptr->oltp(), ptr.get());
    return ptr;
  }

  friend class COWArena;
};

}  // namespace snapshot
}  // namespace aeolus

#endif /* AEOLUS_SNAPSHOT_COW_ARENA_HPP_ */
