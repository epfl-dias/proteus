/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <olap/test/environment.hpp>
#include <oltp.hpp>
#include <tpcc/tpcc_64.hpp>

constexpr size_t runtime_sec = 10;

::testing::Environment* const pools_env =
    ::testing::AddGlobalTestEnvironment(new OLAPTestEnvironment);

// TEST(TPCC, tpcc_vanilla_one_partition) {
//  EXPECT_NO_THROW({
//    auto oltp_num_workers =
//        topology::getInstance().getCpuNumaNodes()[0].local_cores.size();
//    auto npartitions = 1;
//    LOG(INFO) << "N_Partitions: " << npartitions;
//
//    g_num_partitions = npartitions;
//    OLTP oltp_engine{};
//    bench::Benchmark* bench =
//        new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers, true);
//
//    oltp_engine.init(bench, oltp_num_workers, npartitions);
//    LOG(INFO) << "[OLTP] Initialization completed.";
//
//    oltp_engine.print_storage_stats();
//    oltp_engine.run();
//
//    usleep((runtime_sec)*1000000);
//    oltp_engine.shutdown(true);
//
//    bench->deinit();
//    delete bench;
//  }
//
//  );
//}

TEST(TPCC, tpcc_vanilla_all_partition) {
  EXPECT_NO_THROW({
    auto oltp_num_workers = topology::getInstance().getCoreCount();
    auto npartitions = topology::getInstance().getCpuNumaNodes().size();
    LOG(INFO) << "N_Partitions: " << npartitions;

    g_num_partitions = npartitions;
    OLTP oltp_engine{};
    bench::Benchmark* bench =
        new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers, true);

    oltp_engine.init(bench, oltp_num_workers, npartitions);
    LOG(INFO) << "[OLTP] Initialization completed.";

    oltp_engine.print_storage_stats();
    oltp_engine.run();

    usleep((runtime_sec)*1000000);
    oltp_engine.shutdown(true);

    bench->deinit();
    delete bench;
  }

  );
}
