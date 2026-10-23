/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#ifndef PROTEUS_IOURING_HPP
#define PROTEUS_IOURING_HPP

#include <liburing.h>

#include <array>
#include <platform/memory/allocator.hpp>
#include <platform/util/datastructures/threadsafe-set.hpp>

namespace proteus::storage {

class IoUringThreadUnsafe {
 public:
  using CompletionCallBackSuccess = std::function<void()>;
  using CompletionCallBackFailure = std::function<void(io_uring_cqe*)>;
  /**
   * @param max_inflight_requests the maximum number of requests that can be in
   * progress at any time
   */
  IoUringThreadUnsafe(size_t max_inflight_requests);

  /**
   * prepare an async read request.
   *
   * This request may not be submitted until submit is called. poll must be
   * called to poll for completions
   * @param cb a call back that will be called on completion of this request.
   * @see submit
   */
  void read(int fd, void* buf, size_t size, off_t start,
            CompletionCallBackSuccess cb);
  /**
   *
   * @param fd valid file descriptor
   * @param iov pointer to array of iovec. Must remain valid until `submit` is
   * called
   * @param iovcnt length of iov array
   * @param offset starting offset in file in bytes
   * @param cb callback for completion
   * @note code path not tested/validated
   */
  void readv(int fd, const iovec* iov, int iovcnt, off_t offset,
             CompletionCallBackSuccess cb);

  /**
   * Poll for completions and process callbacks. IoUringThreadUnsafe itself does
   * not have a thread or poll by itself. It is the responsibility of the user
   * to call poll
   */
  void poll();

  /**
   * Guarantees that any outstanding requests made since the last call to submit
   * have been submitted
   */
  void submit();

  /**
   * Guarantees that any outstanding requests made since the last call to submit
   * have been submitted and that all completions have been processed
   */
  void flush();

  ~IoUringThreadUnsafe();

 protected:
  /**
   * If we have m_max_inflight_requests currently inflight we need to submit
   * and poll until we can construct and submit further requests
   */
  void poll_until_requests_can_be_made();
  struct IoInfo {
    CompletionCallBackSuccess call_back_success;
    CompletionCallBackFailure call_back_failure;
    struct iovec iov;
  };
  struct io_uring m_ring;
  threadsafe_set<IoInfo*> m_IoInfo_free_set;  /// threadsafe_set is overkill
  int m_count_pending_submissions;
  const size_t m_max_inflight_requests;
};

}  // namespace proteus::storage

#endif  // PROTEUS_IOURING_HPP
