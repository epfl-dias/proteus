/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_LAZY_MASTER_ARENA_HPP
#define PROTEUS_LAZY_MASTER_ARENA_HPP

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>

#include "oltp/snapshot/arena.hpp"

namespace aeolus::snapshot {

class LazyMasterArena : public ArenaV2 {
 public:
  LazyMasterArena() : ArenaV2() {}

  LazyMasterArena(const LazyMasterArena&) = delete;
  LazyMasterArena(LazyMasterArena&&) = delete;
  LazyMasterArena& operator=(const LazyMasterArena&) = delete;
  LazyMasterArena& operator=(LazyMasterArena&&) = delete;

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

  ~LazyMasterArena() override = default;
};

// class LazyMasterProvider {
// public:
//  // static std::map<void *, CircularMasterArena *, std::greater<>> instances;
//
// private:
//  // static CircularMasterArena *getInstance(void *addr) {
//  //   return instances.lower_bound(addr)->second;
//  // }
//
// private:
//  // static void remove(void *olap) { instances.erase(olap); }
//
// public:
//  // static void handler(int sig, siginfo_t *siginfo, void *uap);
//
// public:
//  static void init() {}
//  static void deinit() {}
//
//  static std::unique_ptr<CircularMasterArena> create(size_t size) {
//    auto ptr = std::make_unique<CircularMasterArena>(
//        size, CircularMasterArena::guard{5});
//    // instances.emplace(ptr->olap(), ptr.get());
//    return ptr;
//  }
//
//  friend class CircularMasterArena;
//};
}  // namespace aeolus::snapshot

#endif  // PROTEUS_LAZY_MASTER_ARENA_HPP
