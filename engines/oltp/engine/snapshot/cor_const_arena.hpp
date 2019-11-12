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

#ifndef AEOLUS_SNAPSHOT_COR_CONST_ARENA_HPP_
#define AEOLUS_SNAPSHOT_COR_CONST_ARENA_HPP_

#include <signal.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>

#include "engines/oltp/engine/include/snapshot/arena.hpp"

namespace aeolus {
namespace snapshot {

class CORConstArena : public Arena<CORConstArena> {
 protected:
  size_t size_bytes;

  int *const olap_arena;
  int *const oltp_arena;
  int shm_fd;

  int8_t *olap_start;
  int8_t *start;
  uint64_t *dirty;
  uint64_t *new_dirty;
  size_t dirty_segs;

 protected:
  class guard {
   private:
    guard(int) {}

    friend class CORConstProvider;
  };

 public:
  CORConstArena(size_t size, guard g);
  ~CORConstArena();

  int *oltp() const { return oltp_arena; }
  int *olap() const { return olap_arena; }

 protected:
  void create_snapshot_();
  void destroy_snapshot_() {}

  friend class Arena<CORConstArena>;
  friend class CORConstProvider;
};

class CORConstProvider {
 public:
  static size_t page_size;
  static std::map<void *, CORConstArena *, std::greater<>> instances;

 private:
  static CORConstArena *getInstance(void *addr) {
    return instances.lower_bound(addr)->second;
  }

 private:
  static void remove(void *olap) { instances.erase(olap); }

 public:
  static void handler(int sig, siginfo_t *siginfo, void *uap);

 public:
  static void init();
  static void deinit();

  static std::unique_ptr<CORConstArena> create(size_t size) {
    size = ((size + page_size - 1) / page_size) * page_size;
    auto ptr = std::make_unique<CORConstArena>(size, CORConstArena::guard{5});
    instances.emplace(ptr->olap(), ptr.get());
    return ptr;
  }

  friend class CORConstArena;
};

}  // namespace snapshot
}  // namespace aeolus

#endif /* AEOLUS_SNAPSHOT_COR_CONST_ARENA_HPP_ */
