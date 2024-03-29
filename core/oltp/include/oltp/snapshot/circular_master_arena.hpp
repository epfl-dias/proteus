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

#ifndef AEOLUS_SNAPSHOT_CIRCULAR_MASTER_ARENA_HPP_
#define AEOLUS_SNAPSHOT_CIRCULAR_MASTER_ARENA_HPP_

#include <signal.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>

#include "oltp/snapshot/arena.hpp"

namespace aeolus::snapshot {

class CircularMasterArenaV2 : public ArenaV2 {
 public:
  CircularMasterArenaV2() : ArenaV2() {}

  CircularMasterArenaV2(const CircularMasterArenaV2&) = delete;
  CircularMasterArenaV2(CircularMasterArenaV2&&) = delete;
  CircularMasterArenaV2& operator=(const CircularMasterArenaV2&) = delete;
  CircularMasterArenaV2& operator=(CircularMasterArenaV2&&) = delete;

 public:
  void create_snapshot(metadata save) override {
    this->duringSnapshot = std::move(save);
  }

  void setUpdated() override { duringSnapshot.upd_since_last_snapshot = true; }

  const metadata& getMetadata() override { return duringSnapshot; }

  void destroy_snapshot() override {}

  void init(size_t size) override {}
  void deinit() override {}

  [[nodiscard]] void* oltp() const override { return nullptr; }
  [[nodiscard]] void* olap() const override { return nullptr; }

  ~CircularMasterArenaV2() override = default;
};

class CircularMasterArena : public Arena<CircularMasterArena> {
 protected:
  class guard {
   private:
    guard(int) {}

    friend class CircularMasterProvider;
  };

 public:
  CircularMasterArena(size_t size, guard g) {}
  ~CircularMasterArena() {}

  int* oltp() const { return nullptr; }
  int* olap() const { return nullptr; }

 protected:
  void create_snapshot_() {}
  void destroy_snapshot_() {}

  friend class Arena<CircularMasterArena>;
  friend class CircularMasterProvider;
};

class CircularMasterProvider {
 public:
  // static std::map<void *, CircularMasterArena *, std::greater<>> instances;

 private:
  // static CircularMasterArena *getInstance(void *addr) {
  //   return instances.lower_bound(addr)->second;
  // }

 private:
  // static void remove(void *olap) { instances.erase(olap); }

 public:
  // static void handler(int sig, siginfo_t *siginfo, void *uap);

 public:
  static void init() {}
  static void deinit() {}

  static std::unique_ptr<CircularMasterArena> create(size_t size) {
    auto ptr = std::make_unique<CircularMasterArena>(
        size, CircularMasterArena::guard{5});
    // instances.emplace(ptr->olap(), ptr.get());
    return ptr;
  }

  friend class CircularMasterArena;
};

}  // namespace aeolus::snapshot

#endif /* AEOLUS_SNAPSHOT_CIRCULAR_MASTER_ARENA_HPP_ */
