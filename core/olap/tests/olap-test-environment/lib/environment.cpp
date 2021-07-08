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

#include <glog/logging.h>

#include <olap/test/environment.hpp>
#include <platform/memory/memory-manager.hpp>

void OLAPTestEnvironment::SetUp() {
  assert(!has_already_been_setup);

  setbuf(stdout, nullptr);

  google::InstallFailureSignalHandler();

  // FIXME: reenable tracing as soon as we find the issue with libunwind
  set_trace_allocations(false, true);

  olap = std::make_unique<proteus::olap>();

  has_already_been_setup = true;
}

void OLAPTestEnvironment::TearDown() {
  if (!is_noop) {
    olap.reset();
    has_already_been_setup = false;
  }
}

bool OLAPTestEnvironment::has_already_been_setup = false;
