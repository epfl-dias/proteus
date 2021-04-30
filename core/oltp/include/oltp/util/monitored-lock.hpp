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

#ifndef PROTEUS_MONITORED_LOCK_HPP
#define PROTEUS_MONITORED_LOCK_HPP

#include <mutex>

namespace lock {

template <class T>
class monitor {
 private:
  mutable T t;
  mutable std::mutex m;

 public:
  monitor(T t_) : t(t_) {}
  template <typename F>
  auto operator()(F f) const -> decltype(f(t)) {
    std::lock_guard<std::mutex> hold{m};
    return f(t);
  }
};

/*
 elem = index->find(item)
monitor{elem}([](unlocked_elem){
  elem2 = index->find(item2);
  monitor{elem2}([](unlocked_elem2){
    unlocked_elem2 = unlocked_elem;
    unlocked_elem = 5;
  })
})
 * */

}  // namespace lock

#endif  // PROTEUS_MONITORED_LOCK_HPP
