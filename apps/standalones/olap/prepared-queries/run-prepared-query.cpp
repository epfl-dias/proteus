/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include <cli-flags.hpp>
#include <common/olap-common.hpp>
#include <memory/memory-manager.hpp>
#include <plugins/binary-block-plugin.hpp>
#include <ssb100/query.hpp>
#include <ssb1000/query.hpp>
#include <storage/storage-manager.hpp>
#include <topology/topology.hpp>

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  srand(time(nullptr));

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;  // FIXME: the command line flags/defs seem to fail...

  google::InstallFailureSignalHandler();

  if (FLAGS_query_topology) {
    topology::init();
    std::cout << topology::getInstance() << std::endl;
    return 0;
  }

  set_trace_allocations(FLAGS_trace_allocations);

  proteus::olap::init(FLAGS_gpu_buffers, FLAGS_cpu_buffers,
                      FLAGS_log_buffer_usage);

  LOG(INFO) << "Finished initialization";

  LOG(INFO) << "Preparing queries...";

  {
    std::vector<PreparedStatement> statements;
    //    for (const auto &memmv : {true, false}) {
    //      auto v = ssb100::Query{}.prepareAll(memmv);
    //      statements.insert(statements.end(),
    //      std::make_move_iterator(v.begin()),
    //                        std::make_move_iterator(v.end()));
    //    }
    for (const auto &memmv : {false}) {
      auto v = ssb100::Query{}.prepareAll(memmv);
      statements.insert(statements.end(), std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
    }

    for (size_t i = 0; i < 5; ++i) {
      for (auto &statement : statements) {
        statement.execute();
      }
    }
  }

  LOG(INFO) << "Shutting down...";

  LOG(INFO) << "Unloading files...";
  StorageManager::unloadAll();

  LOG(INFO) << "Shuting down memory manager...";
  MemoryManager::destroy();

  LOG(INFO) << "Shut down finished";
  return 0;
}
