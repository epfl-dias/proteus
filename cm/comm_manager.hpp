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

#ifndef COMM_MANAGER_HPP_
#define COMM_MANAGER_HPP_

#include <errno.h>
#include <fcntl.h>
#include <map>
#include <mqueue.h>
#include <pthread.h>
#include <signal.h>
//#include <stdio.h>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace CM {

#define QUEUE_PERMISSIONS 0600
#define MAX_MESSAGES 10
#define MAX_MSG_SIZE 500
#define MSG_BUFFER_SIZE MAX_MSG_SIZE + 1

#define INCOMING_QUEUE "/htap_mqueue"

// #define INCOMING_QUEUE_OLTP "/aeolus_to_rm"
// #define OUTGOING_QUEUE_OLTP "/rm_to_aeolus"

// #define INCOMING_QUEUE_OLAP "/proteus_to_rm"
// #define OUTGOING_QUEUE_OLAP "/rm_to_proteus"

enum communicatin_msg_type {
  SUCCESS = 01,
  FAILURE = 02,
  REGISTER_OLTP = 11,
  REGISTER_OLAP = 12,
  MEMORY_REQUEST = 21,

  SNAPSHOT_REQUEST = 30,

  TXN_BARRIER = 31,

};

enum client_type { OLAP, OLTP };

struct client {
  uint id;
  enum client_type type;
  std::string queue_name;
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

  void process(char in_buffer[MAX_MSG_SIZE]);

private:
  mqd_t recv_mq;
  struct client oltp_client;
  bool oltp_connected;

  std::vector<client> olap_clients;
  std::atomic<uint> connected_olap_clients;

  static void respond(const char *msg, size_t msg_len,
                      const std::string &queue_name);
  static void process_msg(union sigval sv);

  CommManager() { oltp_connected = false; }
  ~CommManager();
};

} // namespace CM

#endif /* COMM_MANAGER_HPP_ */
