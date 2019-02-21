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

#ifndef GLO_HPP_
#define GLO_HPP_

#include "indexes/hash_index.hpp"
#include "transactions/cc.hpp"

// typedef cuckoohash_map<std::string, std::string> HashIndex;

// template <class hash_val, class key = uint64_t>
// using HashIndex = cuckoohash_map<key, hash_val>;

namespace global_conf {

using ConcurrencyControl = txn::CC_GlobalLock;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;

template <class T_KEY>
using PrimaryIndex = indexes::HashIndex<T_KEY, IndexVal>;

}  // namespace global_conf

#endif /* GLO_HPP_ */
