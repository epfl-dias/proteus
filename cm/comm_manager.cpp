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

#include "comm_manager.hpp"

#include <mqueue.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "sm/storage_manager.hpp"

namespace CM {

void CommManager::shutdown() { this->~CommManager(); }

CommManager::~CommManager() {
  if (mq_close(recv_mq) == -1) {
    std::cerr
        << "[HTAP][CommManager][shutdown] Error in closing incoming queue: "
        << strerror(errno) << std::endl;
  }
}

void CommManager::process(char inn[MAX_MSG_SIZE]) {
  const communicatin_msg_type resp = SUCCESS;
  const communicatin_msg_type resp_fail = FAILURE;
  communicatin_msg_type *msg = (communicatin_msg_type *)inn;
  ushort id;
  enum client_type type;
  std::string queue_name(inn + sizeof(communicatin_msg_type));

  switch (*msg) {
    case REGISTER_OLTP: {
      assert(oltp_connected == false);
      std::cout << "OLTP Queue Name:" << queue_name << std::endl;
      oltp_client = {connected_olap_clients.fetch_add(1), OLTP, queue_name};
      CommManager::respond((char *)&resp, sizeof(communicatin_msg_type),
                           queue_name);
      oltp_connected = true;
      break;
    }
    case REGISTER_OLAP: {
      struct client olap_client {
        connected_olap_clients.fetch_add(1), OLTP, queue_name
      };
      olap_clients.push_back(olap_client);
      CommManager::respond((char *)&resp, sizeof(communicatin_msg_type),
                           queue_name);
      break;
    }
    case MEMORY_REQUEST: {
      // Message: |CODE|REPLY_QUEUE_NAME|KEY::BYTES::UNIT_SIZE|

      std::string msg(inn + sizeof(communicatin_msg_type));

      std::size_t one = msg.find("::");
      assert(one != std::string::npos);
      queue_name = msg.substr(0, one);

      std::size_t two = msg.find("::", one + 2);
      assert(two != std::string::npos);
      // std::cout << "one:" << one << std::endl;
      // std::cout << "two:" << two << std::endl;
      std::string shm_key = msg.substr(one + 2, two - one - 2);

      size_t *size_bytes =
          (size_t *)(inn + sizeof(communicatin_msg_type) + (two + 2));

      size_t *unit_size =
          size_bytes + 1;  //(size_t *)(inn + sizeof(communicatin_msg_type) +
                           //           (two + 2) + sizeof(size_t));

      std::cout << "queue_name: " << queue_name << std::endl;
      std::cout << "shm_key: " << shm_key << std::endl;
      std::cout << "size_bytes: " << *size_bytes << std::endl;
      std::cout << "unit_bytes: " << *(size_bytes + 1) << std::endl;

      if (storage::StorageManager::getInstance().alloc_shm(shm_key, *size_bytes,
                                                           *unit_size)) {
        // success
        CommManager::respond((char *)&resp, sizeof(communicatin_msg_type),
                             queue_name);
      } else {
        // failure
        CommManager::respond((char *)&resp_fail, sizeof(communicatin_msg_type),
                             queue_name);
      }
    }
    case SNAPSHOT_REQUEST: {
      // Proteus is gonna ask for snapshot
      // htap layer will do the things

      // Ask OLTP to stop.
      // Ask OLTP for # of records for all columns, epoch #
      // set. then fork.

      // Parent: ask OLTP to resume it's shit.
      // Parent: set snapshot/epoch_id to zero

      // Child: communicate with proteus..
      // Child: return proteus with queue handler for the child, epoch_id.

      // Message: |CODE|REPLY_QUEUE_NAME|
      // Response:

      std::string msg(inn + sizeof(communicatin_msg_type));

      std::size_t one = msg.find("::");
      assert(one != std::string::npos);
      queue_name = msg.substr(0, one);

      const communicatin_msg_type txn_barrier = TXN_BARRIER;
      std::string oltp_queue_name = "aaa";

      // char *reg_msg =
      //     new char[sizeof(communicatin_msg_type) + queue_name.length() +
      //              key.length() + sizeof(size_t)];
      // size_t offset = 0;

      // memcpy(reg_msg, &reg, sizeof(communicatin_msg_type));
      // offset += sizeof(communicatin_msg_type);

      //    strcpy(reg_msg + offset, oltp_queue_name.c_str());

      CommManager::respond((char *)&txn_barrier, sizeof(communicatin_msg_type),
                           queue_name);
    }
    default:
      std::cout << "[HTAP][DEFAULT] Recieved msg type: " << *msg << std::endl;
  }

  // PROTOCOL

  /*
    if a message starts with 0, means its a register_request. register that
    client. else if msg starts with something else, that is the client id.


    Q: why not hardcode it for now, hardcode for OLTP and OLAP maybe for now.

  */
}

// Possible BUG: first process messages and then set notifier. and i think it is
// not thread safe.
void CommManager::process_msg(union sigval sv) {
  static CommManager *ref = &CM::CommManager::getInstance();

  // struct timeval tp;
  // gettimeofday(&tp, NULL);
  // long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
  // fprintf(stdout, "Message recieved at: %lu\n", ms);

  char in_buffer[MAX_MSG_SIZE];

  // Re-register for new messages on Q
  struct sigevent sev;
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = &process_msg;
  sev.sigev_notify_attributes = NULL;
  // sev.sigev_value.sival_ptr = sv.sival_ptr;

  if (mq_notify(ref->recv_mq, &sev) < 0) {
    std::cerr
        << "[HTAP][CommManager][process_msg] Error during re-registering the "
           "message queue: "
        << strerror(errno) << std::endl;
    // exit(EXIT_FAILURE);
  }

  while (mq_receive(ref->recv_mq, in_buffer, MAX_MSG_SIZE, NULL) > 0) {
    ref->process(in_buffer);
  }

  if (errno != EAGAIN) {
    std::cerr << "[HTAP][CommManager][process_msg] Error in mq_recieve: "
              << strerror(errno) << std::endl;
  }

  // if (mq_receive(ref->recv_mq, in_buffer, MAX_MSG_SIZE, NULL) < 0 &&
  //     errno != EAGAIN) {
  //   std::cerr << "[CommManager][process_msg] Error in mq_recieve: "
  //             << strerror(errno) << std::endl;
  // } else {
  //   // in_buffer contains the message, now parse and process accordingly.
  //   std::cout << "Recieved MSG" << std::endl;
  //   ref->process(in_buffer);
  // }
}

// FIXME: not thread-safe
void CommManager::respond(const char *msg, size_t msg_len,
                          const std::string &queue_name) {
  // Assume response_msg is `char response_msg[MSG_BUFFER_SIZE];`
  /*
  send a core request to RM,
  wait for the RM to respond with either
  -1 : request rejected
  0/+ve integer: number being core_id has been assigned
  start transactions on the the recieved core.
  */

  mqd_t mq;
  // char out_buffer[MSG_BUFFER_SIZE];
  struct mq_attr attr;

  attr.mq_flags = 0;
  attr.mq_maxmsg = MAX_MESSAGES;
  attr.mq_msgsize = MAX_MSG_SIZE;
  attr.mq_curmsgs = 0;

  // Open server queue
  if ((mq = mq_open(queue_name.c_str(),
                    O_WRONLY /*, QUEUE_PERMISSIONS, &attr*/)) == -1) {
    std::cerr << "[HTAP][CommManager] Error opening outgoing queue: "
              << strerror(errno) << std::endl;
  }

  if (mq_send(mq, msg, msg_len, 0) == -1) {
    std::cerr << "[HTAP][CommManager] Cannot send message to server: "
              << strerror(errno) << std::endl;
  }

  if (mq_close(mq) == -1) {
    std::cerr << "[HTAP][CommManager] Error closing outgoing queue: "
              << strerror(errno) << std::endl;
  }
}

void CommManager::init() {
  struct sigevent sev;
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = &process_msg;
  sev.sigev_notify_attributes = NULL;
  // sev.sigev_value.sival_ptr = &recv_mq;

  struct mq_attr attr;
  struct mq_attr old_attr;
  attr.mq_flags = 0;
  attr.mq_maxmsg = MAX_MESSAGES;
  attr.mq_msgsize = MAX_MSG_SIZE;
  attr.mq_curmsgs = 0;

  // open queue
  if ((recv_mq = mq_open(INCOMING_QUEUE, O_CREAT, QUEUE_PERMISSIONS, &attr)) ==
      -1) {
    std::cerr << "[HTAP][CommManager] Error opening incoming queue: "
              << strerror(errno) << std::endl;
  }

  // Clear the msg queue for pending messages
  mq_getattr(recv_mq, &attr);

  std::cout << "[HTAP][CommManager] " << attr.mq_curmsgs
            << " already present in the message queue." << std::endl;

  if (attr.mq_curmsgs != 0) {
    // There are some messages on this queue....eat em

    // First set the queue to not block any calls
    attr.mq_flags = O_NONBLOCK;
    mq_setattr(recv_mq, &attr, &old_attr);

    // Now eat all of the messages
    char temp_buffer[MSG_BUFFER_SIZE];
    while (mq_receive(recv_mq, temp_buffer, MAX_MSG_SIZE, NULL) != -1)
      ;

    // Now restore the attributes
    mq_setattr(recv_mq, &old_attr, 0);
  }

  if (mq_notify(recv_mq, &sev) == -1) {
    std::cerr << "[HTAP][CommManager] Error setting up mqueue listener: "
              << strerror(errno) << std::endl;
  }

  std::cout << "[HTAP][CommManager] setup completed." << std::endl;
}
}  // namespace CM
