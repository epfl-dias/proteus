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

#include "test-utils.hpp"

#include "codegen/plan/prepared-statement.hpp"
#include "engines/olap/util/functions.hpp"
#include "engines/olap/util/parallel-context.hpp"
#include "memory/memory-manager.hpp"
#include "plan/plan-parser.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/schema.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/jit/pipeline.hpp"

void TestEnvironment::SetUp() {
  if (has_already_been_setup) {
    is_noop = true;
    return;
  }

  setbuf(stdout, nullptr);

  google::InstallFailureSignalHandler();

  set_trace_allocations(true, true);

  proteus::init();

  has_already_been_setup = true;
}

void TestEnvironment::TearDown() {
  if (!is_noop) {
    MemoryManager::destroy();
    has_already_been_setup = false;
  }
}

bool TestEnvironment::has_already_been_setup = false;
