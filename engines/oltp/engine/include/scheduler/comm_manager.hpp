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

#ifndef SCHEDULER_COMM_MANAGER_HPP_
#define SCHEDULER_COMM_MANAGER_HPP_

#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <map>
#include <vector>

#define INCOMING_QUEUE_AEOLUS_NAME "/rm_to_aeolus"
#define OUTGOING_QUEUE_AEOLUS_NAME "/htap_mqueue"

#define QUEUE_PERMISSIONS 0600
#define MAX_MESSAGES 10
#define MAX_MSG_SIZE 500
#define MSG_BUFFER_SIZE MAX_MSG_SIZE + 1

namespace scheduler {

enum communicatin_msg_type {
  SUCCESS = 01,
  FAILURE = 02,
  REGISTER_OLTP = 11,
  REGISTER_OLAP = 12,
  MEMORY_REQUEST = 21,

  SNAPSHOT_REQUEST = 30,
  SNAPSHOT_NUM_RECORD = 31,

  READ_TXN_REQUEST = 40,

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

  void init();
  void shutdown();

  // memory allocations
  bool request_memory_alloc(const std::string &key, const size_t size_bytes,
                            const size_t unit_size);
  bool request_memory_free(const std::string &key);

  // elasticity
  [[noreturn]] void scale_up();
  [[noreturn]] void scale_down();

  // snapshot
  [[noreturn]] void snaphsot();

 private:
  mqd_t recv_mq;

  static void send_msg(const char *response_msg);
  static void process_msg(union sigval sv);

  ssize_t get_response(const std::string &queue_name,
                       char (&array)[MSG_BUFFER_SIZE]);
  CommManager() {}
  ~CommManager();
};

}  // namespace scheduler

#endif /* SCHEDULER_COMM_MANAGER_HPP_ */
