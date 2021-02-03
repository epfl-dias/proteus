/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_INTERVAL_MAP_HPP
#define PROTEUS_INTERVAL_MAP_HPP

#include <cassert>
#include <deque>
#include <map>
#include <platform/memory/allocator.hpp>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace utils {

// TODO: look at LLVM IntervalMap
// first-look, LLVM-IntervalMap, you cannot insert a interval which
// is in between the larger inteval. given the requirement, we can change
// the mapped value in between and then it will essentially break the interval
// resulting in three intervals.

// custom interval-map based on std::map
// which stores only the starting-point
// and assumes until the next starting point
// the value is same.

template <typename V = size_t>
class IntervalValue {
 public:
  explicit IntervalValue(V value) : value(value) {}

  V value;
};

// FIXME: not concurrent at the moment.

// LOGIC: map stores the starting position and value associated.
// end is based on the either next value or infinite.

template <typename K = size_t, typename V = size_t>
class IntervalMap {
 public:
  typedef IntervalValue<V> value_t;
  typedef K key_t;

  IntervalMap(K start, V val) : upper_bound(start) { map.emplace(start, val); }
  IntervalMap() : upper_bound(std::numeric_limits<K>::max()) {
    map.emplace(0, 0);
  }

  V getValue(K key) {
    std::shared_lock(this->map_latch);

    if (key >= upper_bound) {
      return map.rend()->second.value;
    } else {
      auto bound_iter = this->map.lower_bound(key);
      if (bound_iter->first == key) {
        return bound_iter->second.value;
      } else {
        // FIXME: check if it wasnt already at the first one.
        bound_iter--;
        return bound_iter->second.value;
      }
    }
  }

  void reset(V value) {
    std::unique_lock(this->map_latch);
    map.clear();
    map.emplace(0, value);
  }

  void upsert(K key, V value) {
    std::shared_lock(this->map_latch);

    // actually directly check if key exists in the map or not.
    auto bound_iter = this->map.lower_bound(key);

    if (bound_iter == map.end()) {
      // key is greater than any in the map.
      assert(key > upper_bound);
      if (map.rbegin()->second.value == value) {
        // case: last value: 10, snap: 0
        // insert 11, snap: 0
        // in this case, the value is already contained, no need to insert.
        return;
      } else {
        map.emplace_hint(map.end(), key + 1, map.rbegin()->second.value);
        map.emplace_hint(map.end(), key, value);
      }
    } else {
      // greater or equal key exist in the map.

      if (bound_iter->first == key) {
        // same key exists
        if (bound_iter->second.value == value) {
          return;
        } else {
          map.try_emplace(bound_iter, (key + 1), bound_iter->second.value);
          bound_iter->second.value = value;
        }
      } else {
        // greater than key exists.
        // how much greater? by doing minus one do the job?
        bound_iter--;
        assert(bound_iter->first > key);

        if (bound_iter->second.value == value) {
          return;
        } else {
          map.try_emplace(bound_iter, (key + 1), bound_iter->second.value);
          map.emplace_hint(bound_iter, key, value);
        }
      }
    }
  }

 private:
  //  std::map<
  //      key_t, value_t,
  //      proteus::memory::PinnedMemoryAllocator<std::pair<const key_t,
  //      value_t>>> map{};
  std::map<key_t, value_t> map{};
  K upper_bound;  // max starting point.
  // K lower_bound; // min starting point

  std::shared_mutex map_latch;
};

}  // namespace utils
#endif  // PROTEUS_INTERVAL_MAP_HPP
