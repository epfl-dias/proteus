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

#include "slack-limiter.hpp"

#include <platform/util/glog.hpp>

namespace proteus {

SlackLimiter::SlackLimiter(size_t slack) : available_slack(slack + 1) {}

void SlackLimiter::addNewPending() {
  if (--available_slack <= 0) {
    std::unique_lock<std::mutex> lock{m};
    cv.wait(lock, [&]() { return available_slack > 0; });
  }
}

bool SlackLimiter::addNewPending(bool polling) {
  //  if (!polling) {
  //    addNewPending();
  //    return true;
  //  }
  if (--available_slack <= 0) {
    notifyComplete();
    return false;
  }
  return true;
}

void SlackLimiter::notifyComplete() {
  ++available_slack;
  cv.notify_one();
}

[[nodiscard]] int32_t SlackLimiter::getAvailableSlack() const {
  return available_slack;
}

}  // namespace proteus
