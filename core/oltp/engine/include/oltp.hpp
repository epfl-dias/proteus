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

#ifndef OLTP_HPP_
#define OLTP_HPP_

#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/table.hpp"
#include "tpcc_64.hpp"
#include "ycsb.hpp"

//
#include "util/timing.hpp"

// TODO: Currently this is made for HTAP. make this standard entry point for
// OLTP, so that even OLTP specific benchmarks can use this file.
// __attribute__((always_inline))

class OLTP {
 public:
  // Core
  inline void init(bench::Benchmark *bench = nullptr, uint num_txn_workers = 1,
                   uint num_data_partitions = 1, uint ch_scale_factor = 0) {
    time_block t("G_OLTP_INIT ");

    g_num_partitions = num_data_partitions;

    /*
    TPCC(std::string name = "TPCC", int num_warehouses = 1,
       int active_warehouse = 1, bool layout_column_store = true,
       uint tpch_scale_factor = 0, int g_dist_threshold = 0,
       std::string csv_path = "", bool is_ch_benchmark = false);
    */
    // bench::Benchmark *bench =
    //     new bench::TPCC("TPCC", num_txn_workers, num_txn_workers, true,
    //                     ch_scale_factor, 0, "", true);

    /*
    void WorkerPool::init(bench::Benchmark* txn_bench, uint num_workers,
                        uint num_partitions, uint worker_sched_mode,
                        int num_iter_per_worker, bool elastic_workload)
    */

    scheduler::WorkerPool::getInstance().init(
        bench, num_txn_workers, num_data_partitions,
        global_conf::reverse_partition_numa_mapping ? 3 : 0);

    // save references
    this->worker_pool = &scheduler::WorkerPool::getInstance();
    this->db = &storage::Schema::getInstance();
    this->txn_manager = &txn::TransactionManager::getInstance();
  }
  inline void run() { this->worker_pool->start_workers(); }

  inline void shutdown(bool print_stats = false) {
    time_block t("G_OLTP_SHUTDOWN ");
    this->worker_pool->shutdown(print_stats);
    this->db->teardown();
  }

  // Stats
  inline void print_global_stats() { this->worker_pool->print_worker_stats(); }
  inline void print_differential_stats() {
    this->worker_pool->print_worker_stats_diff();
  }

  inline std::pair<double, double> get_differential_stats(bool print = false) {
    return this->worker_pool->get_worker_stats_diff(print);
  }

  inline void print_storage_stats() { this->db->report(); }

  // Snapshot
  inline void snapshot() {
    time_block t("G_OLTP_SNAPSHOT ");
    this->txn_manager->snapshot();
  }
  inline void etl(const uint &exec_location_numa_idx) {
    time_block t("G_OLTP_ETL ");
    this->db->ETL(exec_location_numa_idx);
  }

  // Resource Management

  inline void scale_back() { this->worker_pool->scale_back(); }

  inline void scale_up(const uint &num_workers) {
    this->worker_pool->scale_back();
  }
  inline void scale_down(const uint &num_workers) {
    this->worker_pool->scale_down(num_workers);
  }

  inline void migrate_worker(const uint &num_workers) {
    static bool workers_in_home = true;
    for (uint i = 0; i < num_workers; i++) {
      this->worker_pool->migrate_worker();
    }

    workers_in_home = !workers_in_home;
  }

  void getFreshnessRatio() {
    // diff b/w oltp arena and olap arena somehow
    // std::vector<Table *> getAllTables()
    for (auto tbl : db->getAllTables()) {
      _getFreshnessRatioRelation(tbl);
    }
  }

  inline void getFreshnessRatioRelation(std::string table_name) {
    _getFreshnessRatioRelation(db->getTable(table_name));
  }

  inline void getFreshnessRationRelation(int table_idx) {
    _getFreshnessRatioRelation(db->getTable(table_idx));
  }

 private:
  storage::Schema *db;
  scheduler::WorkerPool *worker_pool;
  txn::TransactionManager *txn_manager;

  inline std::pair<size_t, size_t> _getFreshnessRelation(storage::Table *rel) {
    int64_t *oltp_num_records =
        (rel(storage::ColumnStore *))
            ->snapshot_get_number_tuples(false, false);  // oltp snapshot
    int64_t *olap_num_records =
        (rel(storage::ColumnStore *))
            ->snapshot_get_number_tuples(true, false);  // olap snapshot
    size_t oltp_sum = 0;
    size_t olap_sum = 0;

    for (int i = 0; i < g_num_partitions; i++) {
      oltp_sum += oltp_num_records[i];
      olap_sum += olap_num_records[i];
    }

    return std::make_pair(olap_sum, oltp_sum);
  }

  inline double _getFreshnessRatioRelation(storage::Table *rel) {
    auto tmp = _getFreshnessRelation(rel);
    // first - olap snapshot
    // second - oltp snapshot

    return ((double)tmp.first) / ((double)tmp.second);
  }
};

#endif /* OLTP_HPP_ */
