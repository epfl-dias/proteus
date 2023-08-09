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

#ifndef INDEXES_INDEX_HPP_
#define INDEXES_INDEX_HPP_

#include <atomic>
#include <iostream>

#include "oltp/common/common.hpp"

namespace indexes {

enum INDEX_TYPE { HASH_INDEX, RANGE_INDEX };

class IndexAny {
 public:
  explicit IndexAny(std::string name, INDEX_TYPE indexType_)
      : name(std::move(name)),
        index_id(index_id_gen.fetch_add(1)),
        indexType(indexType_) {}
  virtual ~IndexAny() = default;

 public:
  const std::string name;
  const index_id_t index_id;
  const INDEX_TYPE indexType;

 private:
  static std::atomic<index_id_t> index_id_gen;
};

template <class K = uint64_t, class V = void *>
class Index : public IndexAny {
 public:
  explicit Index(std::string name, INDEX_TYPE indexType)
      : IndexAny(name, indexType) {}
  ~Index() override = default;

 public:
  virtual V find(K key) = 0;
  virtual bool find(K key, V &value) = 0;
  virtual bool insert(K key, V &value) = 0;
  virtual bool update(const K &key, V &value) = 0;

 public:
};

template <class K = uint64_t, class V = void *>
class HashIndex : public Index<K, V> {
 public:
  explicit HashIndex(std::string name)
      : Index<K, V>(std::move(name), INDEX_TYPE::HASH_INDEX) {}
  ~HashIndex() override = default;

  // Add additional methods specific to hash indexes
};

template <class K = uint64_t, class V = void *>
class RangeIndex : public Index<K, V> {
 public:
  explicit RangeIndex(std::string name)
      : Index<K, V>(std::move(name), INDEX_TYPE::RANGE_INDEX) {}
  ~RangeIndex() override = default;

  // Add additional methods specific to range indexes
};

};  // namespace indexes

#endif /* INDEXES_INDEX_HPP_ */
