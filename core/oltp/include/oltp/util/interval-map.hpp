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
#include <platform/util/timing.hpp>
#include <shared_mutex>
#include <type_traits>
#include <vector>

#include "oltp/util/monitored-lock.hpp"

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

// FIXME: currently, entire map is either shared_lock or unique_lock
//  for gurading concurrent accesses, in general, if only value is being
//  inserted or removed, then map should be locked, else the value should be
//  latched.
//  need more thoughts in making it better concurrent (similar to b+tree in
//  literature for locking and concurrent accesses.

template <typename V = size_t>
class IntervalValue {
 public:
  explicit IntervalValue(V value) : value(value) {}

  V value;
};

// LOGIC: map stores the starting position and value associated.
// end is based on the either next value or infinite.

template <typename K = size_t, typename V = size_t>
class IntervalMap {
 public:
  typedef IntervalValue<V> value_t;
  typedef K key_t;

  IntervalMap(K start, V val) : upper_bound(start) { map.emplace(start, val); }
  IntervalMap() : upper_bound(0) { map.emplace(0, 0); }

  V getValue(K key) {
    std::shared_lock(this->map_latch);

    if (key >= upper_bound) {
      return map.rbegin()->second.value;
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
    std::unique_lock(this->map_latch);

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
        upper_bound = key + 1;
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
        // LOG(INFO) << "BoundIter: " << bound_iter->first;
        bound_iter--;
        // LOG(INFO) << "BoundIter: " << bound_iter->first;
        assert(bound_iter->first < key);

        if (bound_iter->second.value == value) {
          return;
        } else {
          map.try_emplace(bound_iter, (key + 1), bound_iter->second.value);
          map.emplace_hint(bound_iter, key, value);
        }
      }
    }
  }

  void updateByValue(V oldval, V newval) {
    std::unique_lock<std::shared_mutex>(this->map_latch);

    auto map_it = map.begin();
    for (; map_it != map.end(); map_it++) {
      if (map_it->second.value == oldval) {
        map_it->second.value = newval;
      }
    }
  }

  size_t updateByValue_withCount(V oldval, V newval) {
    std::unique_lock(this->map_latch);
    size_t ctr = 0;
    auto map_it = map.begin();
    for (; map_it != map.end(); map_it++) {
      if (map_it->second.value == oldval) {
        map_it->second.value = newval;
        ctr++;
      }
    }
    return ctr;
  }

  void consolidate() {
    std::unique_lock(this->map_latch);
    time_block t("T_mapConsolidate");
    LOG(INFO) << "MapSize: " << map.size();
    auto map_it = map.begin();
    auto nx = std::next(map_it, 1);

    while (nx != map.end() && map_it != map.end()) {
      if (map_it->second.value == nx->second.value) {
        map.erase(nx);
        nx = std::next(map_it, 1);
        continue;
      }

      map_it++;
      nx = std::next(map_it, 1);
    }
    upper_bound = map.rbegin()->first;
    LOG(INFO) << "MapSizeAfterCleaning: " << map.size();
  }

  size_t updateAndConsolidate(V oldval, V newval) {
    // TODO: by calling two sub-functions, this makes the complexity of
    //  map traversal by twice. implement and utilize both functions together.
    size_t ctr = updateByValue_withCount(oldval, newval);
    consolidate();
    return ctr;
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

  friend std::ostream &operator<<(std::ostream &out,
                                  const IntervalMap<K, V> &r) {
    for (const auto &[key, val] : r.map) {
      out << "[" << key << ", )\t:\t" << val.value << "\n";
    }
    return out;
  }
};

}  // namespace utils
#endif  // PROTEUS_INTERVAL_MAP_HPP
