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

#ifndef AEOLUS_SNAPSHOT_ARENA_HPP_
#define AEOLUS_SNAPSHOT_ARENA_HPP_

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace aeolus {
namespace snapshot {

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
  // int64_t* ptr_to_plugin_m1 = nullptr;
  // int64_t* ptr_to_plugin_m0 = nullptr;

  Arena() = default;

  Arena(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena& operator=(Arena&&) = delete;

 public:
  void create_snapshot(metadata save) {
    duringSnapshot = std::move(save);

    // std::cout << "ARENA: NUM RECORDS:" << num_records << std::endl;

    // if (duringSnapshot.master_ver == 0) {
    //   if (ptr_to_plugin_m0 != nullptr) *ptr_to_plugin_m0 = num_records;
    // } else {
    //   if (ptr_to_plugin_m1 != nullptr) *ptr_to_plugin_m1 = num_records;
    // }

    static_cast<T&>(*this).create_snapshot_();
  }

  void setUpdated() { duringSnapshot.upd_since_last_snapshot = true; }

  const metadata& getMetadata() {
    // if (duringSnapshot.master_ver == 0) {
    //   ptr_to_plugin_m0 = upd;

    // } else {
    //   ptr_to_plugin_m1 = upd;
    // }

    return duringSnapshot;
  }

  void destroy_snapshot() { static_cast<T&>(*this).destroy_snapshot_(); }

  void* oltp() const { return T::oltp(); }
  void* olap() const { return T::olap(); }

  static void init(size_t size) { T::init(size); }
  static void deinit() { T::deinit(); }
};

}  // namespace snapshot
}  // namespace aeolus

#endif /* AEOLUS_SNAPSHOT_ARENA_HPP_ */
