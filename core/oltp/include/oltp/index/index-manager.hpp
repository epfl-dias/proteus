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

#ifndef PROTEUS_INDEX_MANAGER_HPP
#define PROTEUS_INDEX_MANAGER_HPP

#include <iostream>
#include <libcuckoo/cuckoohash_map.hh>
#include <platform/util/erase-constructor-idioms.hpp>
#include <vector>

#include "oltp/common/common.hpp"
#include "oltp/index/index.hpp"

namespace indexes {

class IndexManager : proteus::utils::remove_copy_move {
 public:
  // Singleton
  static inline IndexManager &getInstance() {
    static IndexManager instance;
    return instance;
  }

  std::shared_ptr<IndexAny> getPrimaryIndex(table_id_t table_id) const {
    if (primary_index_map.contains(table_id)) {
      return primary_index_map.find(table_id);
    } else {
      return nullptr;
    }
  }

  void registerPrimaryIndex(table_id_t table_id,
                            std::shared_ptr<IndexAny> index) {
    primary_index_map.insert_or_assign(table_id, index);
  }

  void removePrimaryIndex(table_id_t table_id) {
    primary_index_map.erase(table_id);
  }

  auto hasPrimaryIndex(table_id_t table_id) {
    return primary_index_map.contains(table_id);
  }

  //  index_t getIndex(size_t table_id,
  //                   const std::vector<size_t> &column_ids) const;
  //  index_t getIndexByID(size_t index_id) const;
  //  size_t getIndexID(size_t table_id, const std::vector<size_t> &column_ids);

  // create(..)..

 private:
  //  map < index_id, index_t>
  //  map < table_id , index_id>
  //  map <(size_t table_id, std::vector<size_t> column_ids), index_id>

  libcuckoo::cuckoohash_map<table_id_t, std::shared_ptr<IndexAny>>
      primary_index_map;

  IndexManager() = default;
};

}  // namespace indexes

#endif  // PROTEUS_INDEX_MANAGER_HPP
