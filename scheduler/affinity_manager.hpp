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

#ifndef AFFINITY_MANAGER_HPP_
#define AFFINITY_MANAGER_HPP_

#include "scheduler/topology.hpp"

namespace scheduler {
// TODO: EVERYTHING
class AffinityManager {
 protected:
 public:
  // Singleton
  static inline AffinityManager &getInstance() {
    static AffinityManager instance;
    return instance;
  }
  AffinityManager(AffinityManager const &) = delete;  // Don't Implement
  void operator=(AffinityManager const &) = delete;   // Don't implement

  void set(core *core_id) {}
  void set(cpunumanode *core_id) {}
  void get() {}

 private:
  AffinityManager() {}
};
}  // namespace scheduler

#endif /* AFFINITY_MANAGER_HPP_ */
