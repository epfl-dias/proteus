/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Labaratory (DIAS)
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

/*
 * raw-timing.hpp
 *
 *  Created on: Apr 20, 2015
 *      Author: manolee
 */

#ifndef RAW_TIMING_HPP_
#define RAW_TIMING_HPP_

#include "common/common.hpp"

/**
 *  @brief a timer (singleton) object.
 *
 *  XXX getInstance() does returns an unitialized object!
 *  => Use:
 *  Step 1. reset();
 *  Step 2. time_ms();
 */
class stopwatch_t {
 private:
  struct timeval tv;
  long long mark;

  stopwatch_t() {}
  ~stopwatch_t() {}

  // Not implementing; stopwatch_t is a singleton
  stopwatch_t(stopwatch_t const &);     // Don't Implement.
  void operator=(stopwatch_t const &);  // Don't implement.
 public:
  static stopwatch_t &getInstance() {
    static stopwatch_t instance;
    // instance.reset();
    return instance;
  }

  //    stopwatch_t() {
  //        reset();
  //    }
  long long time_us() {
    long long old_mark = mark;
    reset();
    return mark - old_mark;
  }
  double time_ms() { return ((double)(time_us() * 1e-3)); }
  double time() { return ((double)(time_us() * 1e-6)); }
  long long now() {
    gettimeofday(&tv, NULL);
    return tv.tv_usec + tv.tv_sec * 1000000ll;
  }
  void reset() { mark = now(); }
};

#endif /* RAW_TIMING_HPP_ */
