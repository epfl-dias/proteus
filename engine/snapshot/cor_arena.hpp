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

#ifndef AEOLUS_SNAPSHOT_COR_ARENA_HPP_
#define AEOLUS_SNAPSHOT_COR_ARENA_HPP_

#include <signal.h>

#include <cstdint>

#include "arena.hpp"

namespace aeolus {
namespace snapshot {

class CORArena : public Arena<CORArena> {
 private:
  static uint64_t *dirty;
  static uint64_t *new_dirty;
  static int8_t **find_at;
  static int8_t *olap_start;
  static size_t size_bytes;
  static size_t page_size;

  int *oltp_arena;
  static int *olap_arena;
  static int shm_fd;
  static size_t dirty_segs;
  static int8_t *start;

 public:
  inline static void fix_olap(int8_t *addr);

  static void handler2(int sig, siginfo_t *siginfo, void *uap);

  static void handler(int sig, siginfo_t *siginfo, void *uap);

 public:
  static void init(size_t size_bytes);
  int *oltp();
  int *olap();
  void create_snapshot(void *place_at = nullptr);

  void destroy_snapshot();
};

}  // namespace snapshot
}  // namespace aeolus

#endif /* AEOLUS_SNAPSHOT_COR_ARENA_HPP_ */
