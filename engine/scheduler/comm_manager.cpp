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

#include "scheduler/comm_manager.hpp"

#include <mqueue.h>

#include <cstdlib>
#include <iostream>

#include "transactions/transaction_manager.hpp"

namespace scheduler {

static std::string incoming_queue_name = INCOMING_QUEUE_AEOLUS_NAME;

ssize_t CommManager::get_response(const std::string& queue_name,
                                  char (&array)[MSG_BUFFER_SIZE]) {
  mqd_t mem_recv_queue;
  struct mq_attr attr;

  attr.mq_flags = 0;
  attr.mq_maxmsg = MAX_MESSAGES;
  attr.mq_msgsize = MAX_MSG_SIZE;
  attr.mq_curmsgs = 0;

  if ((mem_recv_queue = mq_open(queue_name.c_str(), O_RDONLY | O_CREAT,
                                QUEUE_PERMISSIONS, &attr)) == -1) {
    std::cerr
        << "[CommManager][request_memory_alloc] Error opening mem_recv_queue: "
        << strerror(errno) << std::endl;
  }

  char temp_buffer[MSG_BUFFER_SIZE];
  ssize_t recv_sz =
      mq_receive(mem_recv_queue, temp_buffer, MAX_MSG_SIZE, nullptr);

  if (recv_sz < 0) {
    std::cerr << "[CommManager][request_memory_alloc] Error in mq_recieve: "
              << strerror(errno) << std::endl;
  } else {
    temp_buffer[recv_sz] = '\0';
    std::cout << "recieved: " << temp_buffer << std::endl;

    mq_unlink(queue_name.c_str());
  }

  return recv_sz;
}

bool CommManager::request_memory_alloc(const std::string& key,
                                       const size_t size_bytes,
                                       const size_t unit_size) {
  const communicatin_msg_type reg = MEMORY_REQUEST;
  const std::string queue_name = incoming_queue_name + "_memRequest";
  // queue_name += "_memRequest";

  // Message: |CODE|REPLY_QUEUE_NAME|KEY::BYTES|

  // SEND MESSAGE
  char* reg_msg = new char[sizeof(communicatin_msg_type) + queue_name.length() +
                           key.length() + sizeof(size_t)];
  size_t offset = 0;

  memcpy(reg_msg, &reg, sizeof(communicatin_msg_type));
  offset += sizeof(communicatin_msg_type);

  strcpy(reg_msg + offset, queue_name.c_str());
  offset += queue_name.length();

  strcpy(reg_msg + offset, "::");
  offset += 2;

  strcpy(reg_msg + offset, key.c_str());
  offset += key.length();

  strcpy(reg_msg + offset, "::");
  offset += 2;

  memcpy(reg_msg + offset, &size_bytes, sizeof(size_t));
  offset += sizeof(size_t);

  memcpy(reg_msg + offset, &unit_size, sizeof(size_t));

  std::cout << "Requesting size: " << size_bytes << std::endl;
  std::cout << "Requesting unit size: " << unit_size << std::endl;
  this->send_msg(reg_msg);
  char temp_buffer[MSG_BUFFER_SIZE];

  // WAIT FOR RESPONSE?
  // mqd_t mem_recv_queue;
  // struct mq_attr attr;

  // attr.mq_flags = 0;
  // attr.mq_maxmsg = MAX_MESSAGES;
  // attr.mq_msgsize = MAX_MSG_SIZE;
  // attr.mq_curmsgs = 0;

  // if ((mem_recv_queue = mq_open(queue_name.c_str(), O_RDONLY | O_CREAT,
  //                               QUEUE_PERMISSIONS, &attr)) == -1) {
  //   std::cerr
  //       << "[CommManager][request_memory_alloc] Error opening mem_recv_queue:
  //       "
  //       << strerror(errno) << std::endl;
  // }

  // char temp_buffer[MSG_BUFFER_SIZE];
  // ssize_t recv_sz = mq_receive(mem_recv_queue, temp_buffer, MAX_MSG_SIZE,
  // nullptr);

  // if (recv_sz < 0) {
  //   std::cerr << "[CommManager][request_memory_alloc] Error in mq_recieve: "
  //             << strerror(errno) << std::endl;
  // } else {
  //   temp_buffer[recv_sz] = '\0';
  //   std::cout << "recieved: " << temp_buffer << std::endl;

  //   mq_unlink(queue_name.c_str());
  // }

  ssize_t sz = get_response(queue_name, temp_buffer);

  communicatin_msg_type* msg = (communicatin_msg_type*)temp_buffer;
  if (sz > 0 && *msg == SUCCESS)
    return true;
  else
    return false;
}

bool CommManager::request_memory_free(const std::string& key) { return true; }

// elasticity
[[noreturn]] void CommManager::scale_up() {
  assert(false && "Not Implemeneted");
}
[[noreturn]] void CommManager::scale_down() {
  assert(false && "Not Implemeneted");
}

// snapshot
[[noreturn]] void CommManager::snaphsot() {
  assert(false && "Not Implemeneted");
}

void CommManager::shutdown() { this->~CommManager(); }

CommManager::~CommManager() {
  // if (mq_close(recv_mq) == -1) {

  if (mq_unlink(INCOMING_QUEUE_AEOLUS_NAME) != 0) {
    std::cerr << "[CommManager][shutdown] Error in closing incoming queue: "
              << strerror(errno) << std::endl;
  }
}

/*

  FIXME: how would OLAP would know how much memory to scan? size of shm maybe?
  but we dont zero out the memory so that might be a problem too.

*/

void CommManager::process_msg(union sigval sv) {
  static CommManager* ref = &scheduler::CommManager::getInstance();

  struct timeval tp;
  gettimeofday(&tp, nullptr);
  long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

  // fprintf(stdout, "Message recieved at: %lu\n", ms);

  char in_buffer[MAX_MSG_SIZE];

  // printf ("Received sig .\n");

  // Re-register for new messages on Q
  struct sigevent sev;
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = &process_msg;
  sev.sigev_notify_attributes = nullptr;
  // sev.sigev_value.sival_ptr = sv.sival_ptr;
  if (mq_notify(ref->recv_mq, &sev) < 0) {
    std::cerr << "[CommManager][process_msg] Error during re-registering the "
                 "message queue: "
              << strerror(errno) << std::endl;
    // exit(EXIT_FAILURE);
  }

  if (mq_receive(ref->recv_mq, in_buffer, MAX_MSG_SIZE, nullptr) < 0 &&
      errno != EAGAIN) {
    std::cerr << "[CommManager][process_msg] Error in mq_recieve: "
              << strerror(errno) << std::endl;
  } else {
    // in_buffer contains the message, now parse and process accordingly.

    /* Possible message options:

        - OLAP wants to do something,

          switch master and OLAP will copy/snapshot.
          switch master and OLAP will remote scan.

          OLAP might want to ask the last update timestamp or current epoch
       number.

          Elasticity included in the request or not?

    */

    if (in_buffer[0] == '1' && in_buffer[1] == '0') {
      // Snapshot request
      uint8_t last_master =
          txn::TransactionManager::getInstance().switch_master();
      uint64_t last_epoch = 0;  // BROKEN

      char response_msg[MSG_BUFFER_SIZE];

      // respond with success code + last_epoch_time + master_version!

      // Success code 11
      response_msg[0] = '1';
      response_msg[1] = '1';

      // last epoch time (13 digit epoch number)
      snprintf((response_msg + 2), 14, "%013lu", last_epoch);

      // std::string epoch_str = std::to_string(last_epoch);
      // std::cout << "SANITY TESTING: master_ver: "<< curr_master <<
      // std::endl; std::cout << "SANITY TESTING : sizeof uint64_t converted
      // to stirng: "
      // << epoch_str.length() << std::endl;

      // strncpy((response_msg+2), epoch_str.c_str(), epoch_str.length());

      // master_version
      // std::cout << "sanity test--" << std::to_string(curr_master)[0] <<
      // "---"
      // << std::endl; response_msg[ 2 + epoch_str.length() ] =
      // std::to_string(curr_master)[0];

      response_msg[2 + 13] = std::to_string(last_master)[0];
      response_msg[2 + 14] = '\0';

      std::cout << "[CommManager] Response: " << response_msg << std::endl;

      send_msg(response_msg);

    } else if (in_buffer[0] == '1' && in_buffer[1] == '3') {
      // num_rec_query
    }

    // int resp = atoi(in_buffer);
    // if (resp/CORE_MEM_POSITION == CORE_COMM){
    //   process_response_core(resp);
    // }
    // else if (resp/CORE_MEM_POSITION == MEM_COMM)
    // {
    //   process_response_memory(resp);
    // }
    // else if (resp/CORE_MEM_POSITION == READ_TXN)
    // {
    //   read_txn = 1;
    //   read_txn_range_end = (resp%100) * 1000;
    // }
    // else if (resp == -1)
    // {
    //   printf("Request rejected :(\n");
    // }
    // else
    // {
    //   printf("Invalid comm message recieved from container: %d\n", resp);
    // }
  }
}

// FIXME: not thread-safe
void CommManager::send_msg(const char* response_msg) {
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
  if ((mq = mq_open(OUTGOING_QUEUE_AEOLUS_NAME,
                    O_WRONLY /*, QUEUE_PERMISSIONS, &attr*/)) == -1) {
    std::cerr << "[CommManager] Error opening outgoing queue: "
              << strerror(errno) << std::endl;
  }

  if (mq_send(mq, response_msg, MAX_MSG_SIZE, 0) == -1) {
    std::cerr << "[CommManager] Cannot send message to server: "
              << strerror(errno) << std::endl;
  }

  if (mq_close(mq) == -1) {
    std::cerr << "[CommManager] Error closing outgoing queue: "
              << strerror(errno) << std::endl;
  }
}

void CommManager::init() {
  struct sigevent sev;
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = &process_msg;
  sev.sigev_notify_attributes = nullptr;
  // sev.sigev_value.sival_ptr = &recv_mq;

  struct mq_attr attr;
  struct mq_attr old_attr;
  attr.mq_flags = 0;
  attr.mq_maxmsg = MAX_MESSAGES;
  attr.mq_msgsize = MAX_MSG_SIZE;
  attr.mq_curmsgs = 0;

  if ((recv_mq = mq_open(INCOMING_QUEUE_AEOLUS_NAME, O_CREAT, QUEUE_PERMISSIONS,
                         &attr)) == -1) {
    std::cerr << "[CommManager] Error opening recv queue: " << strerror(errno)
              << std::endl;
  }

  // Clear the msg queue for pending messages
  mq_getattr(recv_mq, &attr);

  if (attr.mq_curmsgs != 0) {
    attr.mq_flags = O_NONBLOCK;
    mq_setattr(recv_mq, &attr, &old_attr);
    char temp_buffer[MSG_BUFFER_SIZE];
    while (mq_receive(recv_mq, temp_buffer, MAX_MSG_SIZE, nullptr) != -1)
      ;

    mq_setattr(recv_mq, &old_attr, nullptr);
  }

  if (mq_notify(recv_mq, &sev) == -1) {
    std::cerr << "[CommManager] Error setting up mqueue listener: "
              << strerror(errno) << std::endl;
  }

  // TODO: Register with HTAP management layer
  const communicatin_msg_type reg = REGISTER_OLTP;

  char* reg_msg = new char[sizeof(communicatin_msg_type) +
                           sizeof(INCOMING_QUEUE_AEOLUS_NAME)];
  // std::string queue_name = INCOMING_QUEUE_AEOLUS_NAME;

  memcpy(reg_msg, &reg, sizeof(communicatin_msg_type));
  memcpy(reg_msg + sizeof(communicatin_msg_type), incoming_queue_name.c_str(),
         sizeof(INCOMING_QUEUE_AEOLUS_NAME));

  this->send_msg(reg_msg);

  std::cout << "[CommManager] setup completed." << std::endl;
}
}  // namespace scheduler
