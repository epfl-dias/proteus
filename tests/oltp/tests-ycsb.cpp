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
#include <ycsb.hpp>

constexpr size_t runtime_sec = 30;

::testing::Environment* const pools_env =
    ::testing::AddGlobalTestEnvironment(new OLAPTestEnvironment);

TEST(YCSB, ycsb_all_socket_50RW_50Zipf) {
  size_t num_records = 1000000;
  double theta = 0.5;
  double rw_ratio = 0.5;
  size_t num_cols = 10;
  size_t ops_per_txn = 10;

  EXPECT_NO_THROW({
    auto oltp_num_workers = topology::getInstance().getCoreCount();
    auto n_partitions = topology::getInstance().getCpuNumaNodes().size();
    LOG(INFO) << "N_Partitions: " << n_partitions;

    num_records = num_records * oltp_num_workers;

    g_num_partitions = n_partitions;
    OLTP oltp_engine{};
    bench::Benchmark* bench =
        new bench::YCSB("YCSB", num_cols, num_records, theta, 0, ops_per_txn,
                        rw_ratio, oltp_num_workers, oltp_num_workers,
                        n_partitions, true, num_cols, num_cols, 0);

    oltp_engine.init(bench, oltp_num_workers, n_partitions);
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

// TEST(YCSB, ycsb_all_socket_physical_50RW_50Zipf) {
//   size_t num_records = 1000000;
//   double theta = 0.5;
//   double rw_ratio = 0.5;
//   size_t num_cols = 10;
//   size_t ops_per_txn = 10;
//
//   EXPECT_NO_THROW({
//     auto ht_size = topology::getInstance().getCores()[0].ht_pairs_id.size();
//     auto oltp_num_workers = topology::getInstance().getCoreCount() / ht_size;
//     auto n_partitions = topology::getInstance().getCpuNumaNodes().size();
//     LOG(INFO) << "N_Partitions: " << n_partitions;
//
//     num_records = num_records * oltp_num_workers;
//
//     g_num_partitions = n_partitions;
//     OLTP oltp_engine{};
//     bench::Benchmark* bench =
//         new bench::YCSB("YCSB", num_cols, num_records, theta, 0, ops_per_txn,
//                         rw_ratio, oltp_num_workers, oltp_num_workers,
//                         n_partitions, true, num_cols, num_cols, 0);
//
//     oltp_engine.init(bench, oltp_num_workers, n_partitions);
//     LOG(INFO) << "[OLTP] Initialization completed.";
//
//     oltp_engine.print_storage_stats();
//     oltp_engine.run();
//
//     usleep((runtime_sec)*1000000);
//     oltp_engine.shutdown(true);
//
//     bench->deinit();
//     delete bench;
//   }
//
//   );
// }
//
// TEST(YCSB, ycsb_all_socket_physical_WriteOnly_50Zipf) {
//   size_t num_records = 1000000;
//   double theta = 0.5;
//   double rw_ratio = 1.0;
//   size_t num_cols = 10;
//   size_t ops_per_txn = 10;
//
//   EXPECT_NO_THROW({
//     auto ht_size = topology::getInstance().getCores()[0].ht_pairs_id.size();
//     auto oltp_num_workers = topology::getInstance().getCoreCount() / ht_size;
//     auto n_partitions = topology::getInstance().getCpuNumaNodes().size();
//     LOG(INFO) << "N_Partitions: " << n_partitions;
//
//     num_records = num_records * oltp_num_workers;
//
//     g_num_partitions = n_partitions;
//     OLTP oltp_engine{};
//     bench::Benchmark* bench =
//         new bench::YCSB("YCSB", num_cols, num_records, theta, 0, ops_per_txn,
//                         rw_ratio, oltp_num_workers, oltp_num_workers,
//                         n_partitions, true, num_cols, num_cols, 0);
//
//     oltp_engine.init(bench, oltp_num_workers, n_partitions);
//     LOG(INFO) << "[OLTP] Initialization completed.";
//
//     oltp_engine.print_storage_stats();
//     oltp_engine.run();
//
//     usleep((runtime_sec)*1000000);
//     oltp_engine.shutdown(true);
//
//     bench->deinit();
//     delete bench;
//   }
//
//   );
// }
