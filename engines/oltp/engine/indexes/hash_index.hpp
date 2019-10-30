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

#ifndef INDEXES_HASH_INDEX_HPP_
#define INDEXES_HASH_INDEX_HPP_

#include <iostream>
#include <libcuckoo/cuckoohash_map.hh>

namespace indexes {

// typedef cuckoohash_map<std::string, std::string> HashIndex;

// template <class key, class hash_val>
// using HashIndex = cuckoohash_map<key, hash_val>;

template <class K, class V = void*>
class HashIndex : public cuckoohash_map<K, V> {};

// template <class K>
// class HashIndex : public cuckoohash_map<K, void*> {
// p_index->find(op.key, val

// void* find(const K &key){

//}

/*template <key, hash_val>
bool delete_fn(const K &key, F fn) {
  const hash_value hv = hashed_key(key);
  const auto b = snapshot_and_lock_two<normal_mode>(hv);
  const table_position pos = cuckoo_find(key, hv.partial, b.i1, b.i2);
  if (pos.status == ok) {
    fn(buckets_[pos.index].mapped(pos.slot));
    return true;
  } else {
    return false;
  }*/
//};

};  // namespace indexes

#endif /* INDEXES_HASH_INDEX_HPP_ */
