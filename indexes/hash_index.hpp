/*
                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef HASH_INDEX_HPP_
#define HASH_INDEX_HPP_

#include <iostream>
#include "lib/libcuckoo/cuckoohash_map.hh"

namespace indexes {

// typedef cuckoohash_map<std::string, std::string> HashIndex;

template <class key, class hash_val>
using HashIndex = cuckoohash_map<key, hash_val>;

/*
template <class key, class >
class HashIndex {
 public:
  HashIndex();

  static void test();
};
*/

};  // namespace indexes

#endif /* HASH_INDEX_HPP_ */
