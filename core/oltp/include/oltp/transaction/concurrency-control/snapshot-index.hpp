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

#ifndef PROTEUS_SNAPSHOT_INDEX_HPP
#define PROTEUS_SNAPSHOT_INDEX_HPP

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/IndexedMap.h>
#include <llvm/ADT/IntervalMap.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "oltp/common/atomic_bit_set.hpp"
#include "oltp/common/common.hpp"

// something to look at: llvm/ADT/IntEqClasses.h¶

namespace txn {

static auto max_columns = std::pow(2, sizeof(column_id_t) * 8) - 1;

class SnapshotIndex_I {
 public:
  virtual ~SnapshotIndex_I();
  virtual snapshot_version_t getSnapshotIdx(column_id_t col_id) = 0;
  virtual void clean() {}
  virtual void setSnapshotIdx(column_id_t col_id, snapshot_version_t ver) = 0;
};

class SnapIdx_naive : public SnapshotIndex_I {
  SnapIdx_naive() : col_snap_mapping(max_columns, -1) {
    assert(UINT8_MAX > max_columns);
  }

  SnapIdx_naive(column_id_t n_columns) : col_snap_mapping(n_columns, -1) {
    // maximum numbers of columns (column_id_t max value)
    assert(UINT8_MAX > n_columns);
  }

  inline snapshot_version_t getSnapshotIdx(column_id_t col_id) override {
    return col_snap_mapping[col_id];
  }

  inline void setSnapshotIdx(column_id_t col_id,
                             snapshot_version_t ver) override {
    col_snap_mapping[col_id] = ver;
  }

 private:
  std::vector<snapshot_version_t> col_snap_mapping;
};

class SnapIdx_Map : public SnapshotIndex_I {
  SnapIdx_Map() { assert(UINT8_MAX > max_columns); }

  SnapIdx_Map(column_id_t n_columns) {
    // maximum numbers of columns (column_id_t max value)
    assert(UINT8_MAX > n_columns);
  }

  inline snapshot_version_t getSnapshotIdx(column_id_t col_id) override {
    auto search = snapMap.find(col_id);
    if (search != snapMap.end()) {
      return search->second;
    } else {
      return -1;
    }
  }

  inline void setSnapshotIdx(column_id_t col_id,
                             snapshot_version_t ver) override {
    snapMap.insert_or_assign(col_id, ver);
  }

  inline void clean() override {
    std::erase_if(snapMap, [](const auto& item) {
      auto const& [key, value] = item;
      return key == -1;
    });
  }

 private:
  std::map<column_id_t, snapshot_version_t, std::less<>> snapMap;
};

class SnapIdx_bitset : public SnapshotIndex_I {
  SnapIdx_bitset() {}
  snapshot_version_t getSnapshotIdx(column_id_t col_id) { return 0; }
  void setSnapshotIdx(column_id_t col_id, snapshot_version_t ver) {}

  utils::AtomicBitSet<512> dirtyMask;
};

using SnapshotIndex = SnapIdx_Map;

}  // namespace txn

#endif  // PROTEUS_SNAPSHOT_INDEX_HPP
