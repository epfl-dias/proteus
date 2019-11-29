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

#include "communication/comm-manager.hpp"

#include <glog/logging.h>
#include <mqueue.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "common/error-handling.hpp"

#define INCOMING_QUEUE_AEOLUS_NAME "/rm_to_aeolus"
#define OUTGOING_QUEUE_AEOLUS_NAME "/aeolus_to_rm"

// #define QUEUE_PERMISSIONS 0600
#define MAX_MESSAGES 10
#define MAX_MSG_SIZE 500
#define MSG_BUFFER_SIZE MAX_MSG_SIZE + 1

namespace communication {

// FIXME: not thread-safe
bool CommManager::reqeust_snapshot(ushort &master_ver, uint64_t &epoch_num) {
  mq_attr attr;
  attr.mq_flags = 0;
  attr.mq_maxmsg = MAX_MESSAGES;
  attr.mq_msgsize = MAX_MSG_SIZE;
  attr.mq_curmsgs = 0;

  // Open server queue
  mqd_t send_mq = linux_run(mq_open(INCOMING_QUEUE_AEOLUS_NAME,
                                    O_WRONLY /*, QUEUE_PERMISSIONS, &attr*/));

  mqd_t recv_mq =
      linux_run(mq_open(OUTGOING_QUEUE_AEOLUS_NAME, O_CREAT, 0600, &attr));

  // Form the message
  // TODO: we know this message would be same then why snprintf
  // CORE_REQUEST_TO_RM);
  // printf("TRIREME: Sending message %d\n",request_code);
  // fflush(stdout);

  // Request Core
  // printf("CM requesting : %d\n",request_code);
  // fflush(stdout);
  // snprintf(out_buffer, sizeof(out_buffer), "%d", request_code);
  char request_buffer[MSG_BUFFER_SIZE]{0};
  char response_buffer[MSG_BUFFER_SIZE]{0};

  bool ret = false;

  request_buffer[0] = '1';
  request_buffer[1] = '0';

  linux_run(mq_send(send_mq, request_buffer, MAX_MSG_SIZE, 0));

  auto bytes_read = mq_receive(recv_mq, response_buffer, MAX_MSG_SIZE, nullptr);
  if (bytes_read >= 0) {
    LOG(INFO) << "[CommManager: ] Received message: " << response_buffer;

    std::string inn(response_buffer);
    auto response_code = atoi(inn.substr(0, 2).c_str());

    if (response_code == 11) {
      ret = true;
      epoch_num = atoi(inn.substr(2, 13).c_str());
      master_ver = atoi(inn.substr(15, 1).c_str());
    }

  } else {
    LOG(WARNING) << "recv: None";
  }

  linux_run(mq_close(send_mq));

  linux_run(mq_close(recv_mq));

  return ret;
}

CommManager::CommManager() { LOG(INFO) << "[CommManager: ] setup completed"; }
}  // namespace communication
