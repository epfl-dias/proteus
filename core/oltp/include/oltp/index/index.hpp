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

template <class K = uint64_t, class V = void *>
class Index {
 public:
  Index() = default;
  explicit Index(std::string name) : name(std::move(name)) {}
  Index(std::string name, rowid_t initial_capacity) : name(std::move(name)) {}

  virtual ~Index() = default;

 public:
  const std::string name;

  // public:
  //  virtual V find(K key) = 0;
  //  virtual bool insert(K key, V &value) = 0;
  //  auto size(){
  //    return _size.load();
  //  }
  //
  // private:
  //  std::atomic<size_t> _size;
};

};  // namespace indexes

#endif /* INDEXES_INDEX_HPP_ */
