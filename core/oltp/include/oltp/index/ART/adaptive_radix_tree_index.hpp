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

#ifndef PROTEUS_ADPATIVE_RADIX_TREE_INDEX_HPP
#define PROTEUS_ADPATIVE_RADIX_TREE_INDEX_HPP
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "../index.hpp"
#include "adaptive_radix_tree_nodes.hpp"
#include "binary_comparable.hpp"

/* TODO: should be lock-free or not?
 * if lock-free, then will need GC and stuff.
 * if not, then simple latching would effect perf.
 * */

namespace indexes {

template <class K, class V>
class AdaptiveRadixTreeIndex;

template <class K, class V>
std::ostream &operator<<(std::ostream &out,
                         const AdaptiveRadixTreeIndex<K, V> &r) {
  // r._root->serialize(out);
  out << *(r._root.get());
  return out;
}

template <class K, class V>
class AdaptiveRadixTreeIndex : public Index<K, V> {
  static_assert(std::numeric_limits<K>::is_integer,
                "AdaptiveRadixTreeIndex: Only integer keys supported.");

 public:
  AdaptiveRadixTreeIndex() = default;
  AdaptiveRadixTreeIndex(std::string name, uint64_t initial_capacity)
      : Index<K, V>(name) {}
  V find(K key);
  bool find(BinaryComparableKey<K> key, V &val);
  void insert(K key, V &value);

 private:
  // Properties
  std::shared_ptr<ARTNode<K, V>> _root;
  static constexpr size_t key_length = sizeof(K) / 8;

  static inline uint getKeyLen(K &key) { return log2(key) + 1; }
  static inline uint getKeyLenInBytes(K &key) { return (log2(key) / 8) + 1; }

  inline void replace(std::shared_ptr<ARTNode<K, V>> parent,
                      std::shared_ptr<ARTNode<K, V>> old_node,
                      std::shared_ptr<ARTNode<K, V>> new_node);

  friend std::ostream &operator<< <>(std::ostream &out,
                                     const AdaptiveRadixTreeIndex<K, V> &r);
};

// template <class K = uint64_t, class V = void *>
// class AdaptiveRadixTreeIndex {
//   static_assert(std::numeric_limits<K>::is_integer::value, "Only numeric keys
//   supported.");
//
// public:
//   AdaptiveRadixTreeIndex(std::string name, uint64_t
//   initial_capacity):name(name){
//     this->reserve(initial_capacity);
//   }
//
//  //class ARTIterator : public std::iterator<std::forward_iterator_tag, V> {};
//
//  // Point-queries
// // bool find(const K &key, V &val) const;
//  //std::shared_ptr<V> find(const K &key) const;
//   V find(K key);
////  bool contains(const K &key) const;
////  K min();
////  K min(V &val);
////  K max();
////  K max(V &val);
////
////  // Range Queries
////  ARTIterator from(K &key) const ;
////  ARTIterator to(K &key) const ;
////  ARTIterator range(K &key_begin, K &key_end) const ;
////  ARTIterator begin() const ;
////  ARTIterator end() const ;
////
////  // Predicate Filtering (scan-all, return passed only)
////  ARTIterator predicate_key(std::function<bool(V)>) const;
////  ARTIterator predicate_value(std::function<bool(V)>) const;
////  ARTIterator predicate(std::function<bool(K)>, std::function<bool(V)>)
/// const;
//
//  // Edits
//  bool insert(K key, V &value);
//  //bool erase(K key, V &value);
//
//  // -------
//
//  /**
//   * Reserves memory for given number of entries.
//   */
//  void reserve(size_t size){}
//
//  /**
//   * Clears all the entries in the index.
//   */
//  //void clear();
//
//  /**
//   * Returns true if the index doesnt contain any entry.
//   */
//  //bool empty() const;
//
//  /**
//   * Returns the number of entries in the index
//   */
//  //size_t size() const;
//
//  /**
//   * Returns the memory consumption of this Index in bytes
//   */
//  //size_t memory_consumption() const;
//
//  /* idea from HyRise (https://github.com/hyrise/hyrise/)
//   * Predicts the memory consumption in bytes of creating this index.
//   * */
//  //static size_t estimate_memory_consumption();
//
// private:
//  // tree root
//  std::shared_ptr<ARTNode<K,V>> _root;
//  static constexpr size_t key_length = sizeof(K) / 8;
//  std::string name;
//
//  static inline uint8_t getKeyLen(K &key){
//    return log2(key)+1;
//  }
//   static inline uint8_t getKeyLenInBytes(K &key){
//     return (log2(key)/8)+1;
//   }
//
//
//  // helper functions
//   //std::shared_ptr<V> find_recursive(std::shared_ptr<ARTNode<K,V>> node,
//   const K &key, size_t depth);
//
//  // store some back-reference to the columns which this index is built upon?
//
//  /* Reporting metrics and statistics of index.
//   * */
////  friend std::ostream &operator<<(
////      std::ostream &out, const indexes::AdaptiveRadixTreeIndex<K, V> &r);
//};
//
//// template <class K, class V = void *>
//// std::ostream &operator<<(std::ostream &out,
////                         const indexes::AdaptiveRadixTreeIndex<K, V> &r);

}  // namespace indexes

#endif  // PROTEUS_ADPATIVE_RADIX_TREE_INDEX_HPP
