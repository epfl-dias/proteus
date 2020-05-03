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

#include <functional>
#include <iostream>
#include <iterator>

/* TODO: should be lock-free or not?
 * if lock-free, then will need GC and stuff.
 * if not, then simple latching would effect perf.
 * */

namespace indexes {

template <class K, class V = void *>
class AdaptiveRadixTreeIndex {
 public:
  class ARTIterator : public std::iterator<std::forward_iterator_tag, V> {};

  // Point-queries
  bool find(const K &key, V &val) const;
  V find(const K &key) const;
  bool contains(const K &key) const;
  K min();
  K min(V &val);
  K max();
  K max(V &val);

  // Range Queries
  ARTIterator from(K &key) const final;
  ARTIterator to(K &key) const final;
  ARTIterator range(K &key_begin, K &key_end) const final;
  ARTIterator begin() const final;
  ARTIterator end() const final;

  // Predicate Filtering (scan-all, return passed only)
  ARTIterator predicate_key(std::function<bool(V)>) const;
  ARTIterator predicate_value(std::function<bool(V)>) const;
  ARTIterator predicate(std::function<bool(K)>, std::function<bool(V)>) const;

  // Edits
  bool insert(K key, V &value);
  bool erase(K key, V &value);

  // -------

  /**
   * Reserves memory for given number of entries.
   */
  void reserve(size_t size);

  /**
   * Clears all the entries in the index.
   */
  void clear();

  /**
   * Returns true if the index doesnt contain any entry.
   */
  bool empty() const;

  /**
   * Returns the number of entries in the index
   */
  size_t size() const;

  /**
   * Returns the memory consumption of this Index in bytes
   */
  size_t memory_consumption() const;

  /* idea from HyRise (https://github.com/hyrise/hyrise/)
   * Predicts the memory consumption in bytes of creating this index.
   * */
  static size_t estimate_memory_consumption();

 private:
  // tree root
  // std::shared_ptr<ARTNode> _root;

  // store some back-refernce to the columns which this index is built upon?

  /* Reporting metrics and statistics of index.
   * */
  friend std::ostream &operator<<(
      std::ostream &out, const indexes::AdaptiveRadixTreeIndex<K, V> &r);
};

template <class K, class V = void *>
std::ostream &operator<<(std::ostream &out,
                         const indexes::AdaptiveRadixTreeIndex<K, V> &r);

}  // namespace indexes

#endif  // PROTEUS_ADPATIVE_RADIX_TREE_INDEX_HPP
