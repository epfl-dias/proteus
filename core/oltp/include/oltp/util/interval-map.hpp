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

#include "oltp/common/common.hpp"
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

// template <typename V = size_t>
// class IntervalValue {
// public:
//  explicit IntervalValue(V value) : value(value) {}
//
//  V value;
//};

// LOGIC: map stores the starting position and value associated.
// end is based on the either next value or infinite.

template <typename K = size_t, typename V = size_t>
class IntervalMap {
 public:
  // typedef IntervalValue<V> value_t;
  typedef V value_t;
  typedef K key_t;

  IntervalMap(K start, V val) : upper_bound(start), touched(false) {
    map.emplace(start, val);
  }
  IntervalMap() : upper_bound(0), touched(false) { map.emplace(0, 0); }

  V getValue(K key) {
    // std::shared_lock<std::shared_mutex> slk(this->map_latch);
    if (__likely(!touched || key >= upper_bound)) {
      // intuition is that snapshot always start at zero, and during insert,
      //  always a +1 is inserted with default value, meaning the upper-bound
      //  will be always zero.
      return 0;
    }
    /*else if (key == upper_bound) {
      return map.rbegin()->second;
    } */
    else {
      std::shared_lock<std::shared_mutex> slk(this->map_latch);
      auto bound_iter = this->map.lower_bound(key);
      if (bound_iter->first == key) {
        return bound_iter->second;
      } else {
        // FIXME: check if it wasnt already at the first one.
        bound_iter--;
        return bound_iter->second;
      }
    }
  }

  void reset(V value) {
    std::unique_lock<std::shared_mutex> ulk(this->map_latch);
    map.clear();
    map.emplace(0, value);
  }

  void upsert(K key, V value) {
    {
      std::shared_lock<std::shared_mutex> slk(this->map_latch, std::defer_lock);

      slk.lock();

      // actually directly check if key exists in the map or not.
      auto bound_iter = this->map.lower_bound(key);

      if (__likely(bound_iter == map.end())) {
        // key is greater than any in the map.
        // assert(key > upper_bound);
        if (__unlikely(map.rbegin()->second == value)) {
          // case: last value: 10, snap: 0
          // insert 11, snap: 0
          // in this case, the value is already contained, no need to insert.
          return;
        } else {
          {
            slk.unlock();
            std::unique_lock<std::shared_mutex> ulk(this->map_latch);
            upper_bound = key + 1;
            map.emplace_hint(map.end(), key + 1, map.rbegin()->second);
            map.emplace_hint(map.end(), key, value);

            if (!touched) touched = true;
          }
        }
      } else {
        // greater or equal key exist in the map.

        if (__likely(bound_iter->first == key)) {
          // same key exists
          if (bound_iter->second == value) {
            return;
          } else {
            {
              slk.unlock();
              std::unique_lock<std::shared_mutex> ulk(this->map_latch);
              map.try_emplace(bound_iter, (key + 1), bound_iter->second);
              bound_iter->second = value;
              if (!touched) touched = true;
            }
          }
        } else {
          // greater than key exists.
          // how much greater? by doing minus one do the job?
          // LOG(INFO) << "BoundIter: " << bound_iter->first;
          bound_iter--;
          // LOG(INFO) << "BoundIter: " << bound_iter->first;
          assert(bound_iter->first < key);

          if (bound_iter->second == value) {
            return;
          } else {
            {
              slk.unlock();
              std::unique_lock<std::shared_mutex> ulk(this->map_latch);
              map.try_emplace(bound_iter, (key + 1), bound_iter->second);
              map.emplace_hint(bound_iter, key, value);
              if (!touched) touched = true;
            }
          }
        }
      }
    }
  }

  void updateByValue(V oldval, V newval) {
    {
      std::unique_lock<std::shared_mutex>(this->map_latch);
      auto map_it = map.begin();
      for (; map_it != map.end(); map_it++) {
        if (map_it->second == oldval) {
          map_it->second = newval;
        }
      }
    }

    if (!touched) touched = true;
  }

  size_t updateByValue_withCount(V oldval, V newval) {
    size_t ctr = 0;
    {
      std::unique_lock<std::shared_mutex> slk(this->map_latch);
      auto map_it = map.begin();
      for (; map_it != map.end(); map_it++) {
        if (map_it->second == oldval) {
          map_it->second = newval;
          ctr++;
        }
      }
    }
    if (!touched) touched = true;
    return ctr;
  }

  void consolidate() {
    if (!touched) return;
    std::unique_lock<std::shared_mutex> slk(this->map_latch);
    time_block t("T_mapConsolidate");
    LOG(INFO) << "MapSize: " << map.size();
    auto map_it = map.begin();
    auto nx = std::next(map_it, 1);

    while (nx != map.end() && map_it != map.end()) {
      if (map_it->second == nx->second) {
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
  std::map<
      key_t, value_t, std::less<key_t>,
      proteus::memory::PinnedMemoryAllocator<std::pair<const key_t, value_t>>>
      map{};
  // std::map<key_t, value_t> map{};
  std::atomic<K> upper_bound;  // max starting point.
  // K lower_bound; // min starting point

  std::shared_mutex map_latch;
  volatile bool touched;

 public:
  friend std::ostream &operator<<(std::ostream &out,
                                  const IntervalMap<K, V> &r) {
    for (const auto &[key, val] : r.map) {
      out << "[" << key << ", )\t:\t" << val << "\n";
    }
    return out;
  }
};

}  // namespace utils
#endif  // PROTEUS_INTERVAL_MAP_HPP
