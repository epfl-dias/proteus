/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef COMM_MANAGER_HPP_
#define COMM_MANAGER_HPP_

#include <mqueue.h>

#include <cstdint>

namespace communication {

enum communicatin_msg_type {

  SNAPSHOT_REQUEST = 10,
  SNAPSHOT_RESPONSE_POSITIVE = 11,
  SNAPSHOT_RESPONSE_NEGATIVE = 12,

  READ_TXN_REQUEST = 20,

  /* old trireme ones*/

  // CORE_REQUEST_TO_RM = 10,
  // CORE_GRANT_FROM_RM = 11,
  // CORE_REQUEST_FROM_RM = 12,
  // MEMORY_REQUEST_TO_RM = 20,
  // MEMORY_GRANT_FROM_RM = 21,
  // MEMORY_REQUEST_FROM_RM = 22,
  // READ_TXN_REQUEST = 30,
  // READ_TXN_RESPONSE = 31,
  // SHUTDOWN_NOTIFICATION_TO_RM = 90,
};

class CommManager {
 protected:
 public:
  // Singleton
  static CommManager &getInstance() {
    static CommManager instance;
    return instance;
  }

  // Prevent copies
  CommManager(const CommManager &) = delete;
  void operator=(const CommManager &) = delete;

  CommManager(CommManager &&) = delete;
  CommManager &operator=(CommManager &&) = delete;

  bool reqeust_snapshot(unsigned short &master_ver, uint64_t &epoch_num);

 private:
  mqd_t recv_mq;

  CommManager();
  ~CommManager() = default;
};

}  // namespace communication

#endif /* COMM_MANAGER_HPP_ */
