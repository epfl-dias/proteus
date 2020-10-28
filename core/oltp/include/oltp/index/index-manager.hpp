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
#include <vector>

//#include "../indexes/index.hpp"

namespace indexes {

class IndexAny { /* [...] */
};

template <typename K = uint64_t, class V = void *>
class Index : public IndexAny {
  Index() = default;
  Index(std::string name) : name(std::move(name)) {}
  Index(std::string name, uint64_t initial_capacity) : name(std::move(name)) {}
  virtual V find(K key) = 0;
  virtual bool insert(K key, V &value) = 0;
  virtual ~Index() {}

  // some other any. proteus index any.
  proteus_any findAny(K key) { return find(key); }

  const std::string name;
  const size_t index_id;
};

class index_t {
 private:
  std::shared_ptr<IndexAny> index;

 public:
  template <typename K, typename V>
  V probe(const K &key) const {  // or even better, proteus_any instead of T

    const auto &in = reinterpret_cast<const Index<K, V> &>(*(index.get()));
    return in.find(key);
  }

  proteus_any probe(proteus_any key) {}

  VID probe(proteus_any key) {}
};

class IndexManager {
 public:
  // Singleton
  static inline IndexManager &getInstance() {
    static IndexManager instance;
    return instance;
  }
  IndexManager(IndexManager const &) = delete;             // Don't Implement
  IndexManager(IndexManager &&) = delete;                  // Don't Implement
  IndexManager &operator=(IndexManager const &) = delete;  // Don't implement
  IndexManager &operator=(IndexManager &&) = delete;       // Don't implement

  index_t getIndex(size_t table_id,
                   const std::vector<size_t> &column_ids) const;
  index_t getPrimaryIndex(size_t table_id) const;

  index_t getIndexByID(size_t index_id) const;
  size_t getIndexID(size_t table_id, const std::vector<size_t> &column_ids);

  // create(..)..

 private:
  //  map <index_id, index_t>
  //
  //  map..table_id , index_id
  //
  //  map.. (size_t table_id, std::vector<size_t> column_ids), index_id

  IndexManager() {}
};

}  // namespace indexes

#endif  // PROTEUS_INDEX_MANAGER_HPP
