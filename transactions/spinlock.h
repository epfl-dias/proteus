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

#ifndef SPIN_LOCK_HPP_
#define SPIN_LOCK_HPP_

namespace lock {

struct Spinlock {
  Spinlock() : value(0) {}

  int acquire() {
    for (int tries = 0; true; ++tries) {
      if (__sync_bool_compare_and_swap(&value, 0, 1)) return 0;
      if (tries == 100) {
        tries = 0;
        sched_yield();
      }
    }
  }

  int release() {
    __sync_lock_release(&value);
    return 0;
  }

  volatile int value;
};

}  // namespace lock

#endif /* SPIN_LOCK_HPP_ */
