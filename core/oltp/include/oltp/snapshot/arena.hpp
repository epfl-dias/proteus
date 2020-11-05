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

#ifndef AEOLUS_SNAPSHOT_ARENA_HPP_
#define AEOLUS_SNAPSHOT_ARENA_HPP_

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "oltp/common/common.hpp"

namespace aeolus::snapshot {

class ArenaV2 {
 public:
  struct metadata {
    rowid_t numOfRecords;
    // rowid_t prev_numOfRecords;
    xid_t epoch_id;
    // rowid_t num_updated_records;
    master_version_t master_ver;
    partition_id_t partition_id;
    bool upd_since_last_snapshot;
  };

 protected:
  metadata duringSnapshot{};
  ArenaV2() = default;

 public:
  ArenaV2(const ArenaV2&) = delete;
  ArenaV2(ArenaV2&&) = delete;
  ArenaV2& operator=(const ArenaV2&) = delete;
  ArenaV2& operator=(ArenaV2&&) = delete;

  virtual ~ArenaV2() = default;

  virtual void create_snapshot(metadata save) = 0;
  virtual void destroy_snapshot() = 0;

  virtual void setUpdated() = 0;
  virtual const metadata& getMetadata() = 0;

  [[nodiscard]] virtual void* oltp() const = 0;
  [[nodiscard]] virtual void* olap() const = 0;

  virtual void init(size_t size) = 0;
  virtual void deinit() = 0;
};

template <typename T>
class Arena {
  struct metadata {
    uint64_t numOfRecords;
    // uint64_t prev_numOfRecords;
    uint64_t epoch_id;
    // uint64_t num_updated_records;
    uint8_t master_ver;
    uint8_t partition_id;
    bool upd_since_last_snapshot;
  };

 protected:
  metadata duringSnapshot;

  Arena() = default;

  Arena(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena& operator=(Arena&&) = delete;

 public:
  void create_snapshot(metadata save) {
    duringSnapshot = std::move(save);
    static_cast<T&>(*this).create_snapshot_();
  }

  void setUpdated() { duringSnapshot.upd_since_last_snapshot = true; }

  const metadata& getMetadata() { return duringSnapshot; }

  void destroy_snapshot() { static_cast<T&>(*this).destroy_snapshot_(); }

  void* oltp() const { return T::oltp(); }
  void* olap() const { return T::olap(); }
};

}  // namespace aeolus::snapshot

#endif /* AEOLUS_SNAPSHOT_ARENA_HPP_ */
