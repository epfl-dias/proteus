/*
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

#ifndef GLO_HPP_
#define GLO_HPP_

#include "indexes/hash_index.hpp"
#include "transactions/cc.hpp"

// typedef cuckoohash_map<std::string, std::string> HashIndex;

// template <class hash_val, class key = uint64_t>
// using HashIndex = cuckoohash_map<key, hash_val>;

namespace global_conf {

using ConcurrencyControl = txn::CC_MV2PL;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;

bool cc_multiversion = ConcurrencyControl::is_mv();
using mv_version_list = ConcurrencyControl::VERSION_LIST;
using mv_version = ConcurrencyControl::VERSION;

template <class T_KEY>
using PrimaryIndex = indexes::HashIndex<T_KEY, IndexVal>;

/* # of Snapshots*/
const short num_master_versions = 2;

}  // namespace global_conf

#endif /* GLO_HPP_ */
