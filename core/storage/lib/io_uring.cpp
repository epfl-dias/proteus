/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include <storage/io_uring.hpp>

using namespace proteus::storage;

IoUringThreadUnsafe::IoUringThreadUnsafe(size_t max_inflight_requests)
    : m_IoInfo_free_set(),
      m_count_pending_submissions(0),
      m_max_inflight_requests(max_inflight_requests) {
  io_uring probe_ring;
  int probe_success = io_uring_queue_init(1, &probe_ring, 0);
  CHECK(probe_success == 0)
      << "failed to init probe io_uring: " << strerror(-probe_success);

  // first argument is the SQ size. This has no impact on the total number of
  // inflight requests, just the number of requests you can submit at once.
  int init_success = io_uring_queue_init(
      max_inflight_requests, &m_ring,
      IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN);
  // TODO IOPOLL results in no completions ever on dias49 even though it should
  // and we are specifically calling io_uring_submit_and_get_events
  // https://github.com/axboe/liburing/issues/385
  //  IORING_SETUP_IOPOLL |
  CHECK(init_success == 0) << "failed to init ring: "
                           << strerror(-init_success);

  // Minor optimization for io_uring when onl a single thread is accessing it
  if (probe_ring.features & IORING_FEAT_REG_REG_RING) {
    int register_rfd_success = io_uring_register_ring_fd(&m_ring);
    CHECK(register_rfd_success == 1)
        << "failed to register ring fd: " << strerror(-register_rfd_success);
  }
  io_uring_queue_exit(&probe_ring);

  // this is how we actually control inflight requests
  for (size_t i = 0; i < max_inflight_requests; i++) {
    IoInfo *io_info = new IoInfo;
    m_IoInfo_free_set.emplace(io_info);
  }
}

void IoUringThreadUnsafe::read(int fd, void *buf, size_t size, off_t start,
                               CompletionCallBackSuccess cb) {
  poll_until_requests_can_be_made();

  IoInfo *user_info = m_IoInfo_free_set.pop();
  user_info->call_back_success = cb;
  user_info->iov.iov_base = buf;
  user_info->iov.iov_len = size;
  user_info->call_back_failure = [size, start](io_uring_cqe *res_cqe) {
    CHECK(res_cqe->res >= 0)
        << "io_uring request failed with: " << strerror(-res_cqe->res)
        << " start_offset: " << start << "size: " << size;
  };  // forcibly destruct the callback

  struct io_uring_sqe *sqe = io_uring_get_sqe(&m_ring);
  DCHECK_NE(sqe, nullptr)
      << "This should never happen because we restrict inflight IO";
  DCHECK_EQ(start % 512, 0) << "we only support O_DIRECT";
  DCHECK_EQ(size % 512, 0) << "we only support O_DIRECT";
  DCHECK_EQ(reinterpret_cast<uintptr_t>(buf) % 512, 0);
  DCHECK_GE(start, 0);
  io_uring_prep_readv(sqe, fd, &user_info->iov, 1, start);
  io_uring_sqe_set_data(sqe, user_info);
  m_count_pending_submissions += 1;
}

void IoUringThreadUnsafe::readv(int fd, const iovec *iov, int iovcnt,
                                off_t start, CompletionCallBackSuccess cb) {
  poll_until_requests_can_be_made();

  IoInfo *user_info = m_IoInfo_free_set.pop();
  user_info->call_back_success = cb;
  struct io_uring_sqe *sqe = io_uring_get_sqe(&m_ring);
  io_uring_prep_readv(sqe, fd, iov, iovcnt, start);
  io_uring_sqe_set_data(sqe, user_info);
  m_count_pending_submissions += 1;
}

void IoUringThreadUnsafe::submit() {
  int submitted = io_uring_submit_and_get_events(&m_ring);
  CHECK(submitted == m_count_pending_submissions)
      << "Tried to submit: " << m_count_pending_submissions
      << " and failed with: " << strerror(-submitted);
  m_count_pending_submissions = 0;
}

void IoUringThreadUnsafe::poll() {
  nvtxRangePushA("IoUringThreadUnsafe-poll");
  struct io_uring_cqe *cqe;
  unsigned head;
  unsigned i = 0;

  // process completed events
  io_uring_for_each_cqe(&m_ring, head, cqe) {
    IoInfo *io_info = static_cast<IoInfo *>(io_uring_cqe_get_data(cqe));
    if (cqe->res < 0) {
      io_info->call_back_failure(cqe);
    } else {
      io_info->call_back_success();
    }
    m_IoInfo_free_set.emplace(io_info);
    i++;
  }
  io_uring_cq_advance(&m_ring, i);

  // enter the ring to actually poll for events
  auto get_events_status = io_uring_get_events(&m_ring);
  CHECK(get_events_status == 0)
      << "Failed to get events with: " << strerror(-get_events_status);
  nvtxRangePop();
}

void IoUringThreadUnsafe::poll_until_requests_can_be_made() {
  while (m_IoInfo_free_set.empty_unsafe()) {
    submit();
    poll();
  }
}

IoUringThreadUnsafe::~IoUringThreadUnsafe() {
  submit();
  while (m_IoInfo_free_set.size_unsafe() < m_max_inflight_requests) {
    poll();
  }

  for (size_t i = 0; i < m_max_inflight_requests; i++) {
    auto *io_info_ptr = m_IoInfo_free_set.pop();
    delete io_info_ptr;
  }
  io_uring_queue_exit(&m_ring);
}
void IoUringThreadUnsafe::flush() {
  submit();
  while (m_IoInfo_free_set.size_unsafe() < m_max_inflight_requests) {
    poll();
  }
}
