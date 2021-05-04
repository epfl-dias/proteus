/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_SLACK_LIMITER_HPP
#define PROTEUS_SLACK_LIMITER_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace proteus {

class SlackLimiter {
  std::atomic<int32_t> available_slack;

  std::mutex m;
  std::condition_variable cv;

 public:
  explicit SlackLimiter(size_t slack);

  void addNewPending();
  bool addNewPending(bool polling);
  void notifyComplete();
  [[nodiscard]] int32_t getAvailableSlack() const;
};

}  // namespace proteus

#endif /* PROTEUS_SLACK_LIMITER_HPP */
