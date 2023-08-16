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

#include <cassert>
#include <platform/util/timing.hpp>
#include <utility>

#include "oltp/common/constants.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/interface/bench.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction_manager.hpp"

// TODO: Currently this is made for HTAP. make this standard entry point for
//  OLTP, so that even OLTP specific benchmarks can use this file.

class OLTP {
 public:
  // Core
  inline void init(bench::Benchmark *bench = nullptr, uint num_txn_workers = 1,
                   uint num_data_partitions = 1, uint ch_scale_factor = 0,
                   bool collocated_schedule = false) {
    time_block t("G_OLTP_INIT ");

    g_num_partitions = num_data_partitions;

    uint worker_sched_mode =
        global_conf::reverse_partition_numa_mapping ? 3 : 0;

    if (collocated_schedule) {
      worker_sched_mode = 5;
    }

    scheduler::WorkerPool::getInstance().init(
        bench, num_txn_workers, num_data_partitions, worker_sched_mode);

    // save references
    this->worker_pool = &scheduler::WorkerPool::getInstance();
    this->db = &storage::Schema::getInstance();
    this->txn_manager = &txn::TransactionManager::getInstance();
  }

  //  inline void enqueue_query(
  //      std::function<bool(uint64_t, ushort, ushort, ushort)> query) {
  //    //this->worker_pool->enqueueTask(std::move(query));
  //  }

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

  inline void scale_up(const worker_id_t &num_workers) {
    this->worker_pool->scale_back();
  }
  inline void scale_down(const worker_id_t &num_workers) {
    this->worker_pool->scale_down(num_workers);
  }

  inline void migrate_worker(const worker_id_t &num_workers) {
    static bool workers_in_home = false;
    for (auto i = 0; i < (num_workers / 2); i++) {
      this->worker_pool->migrate_worker(workers_in_home);
    }

    workers_in_home = !workers_in_home;
  }

  inline std::pair<size_t, size_t> getFreshness() {
    // diff b/w oltp arena and olap arena

    size_t total_olap = 0;
    size_t total_oltp = 0;

    for (auto &tbl : db->getTables()) {
      auto tmp = _getFreshnessRelation(tbl.second);
      total_olap += tmp.first;
      total_oltp += tmp.second;
    }
    return std::make_pair(total_olap, total_oltp);
  }

  inline double getFreshnessRatio() {
    auto tmp = getFreshness();
    return ((double)tmp.first) / ((double)tmp.second);
  }

  inline std::pair<size_t, size_t> getFreshnessRelation(
      const std::vector<std::string> &tables) {
    size_t total_olap = 0;
    size_t total_oltp = 0;
    for (auto &tb : tables) {
      auto tmp = _getFreshnessRelation(db->getTable(tb));
      total_olap += tmp.first;
      total_oltp += tmp.second;
    }
    return std::make_pair(total_olap, total_oltp);
  }

  inline double getFreshnessRatioRelation(std::vector<std::string> &tables) {
    size_t total_olap = 0;
    size_t total_oltp = 0;
    for (auto &tb : tables) {
      auto tmp = _getFreshnessRelation(db->getTable(tb));
      total_olap += tmp.first;
      total_oltp += tmp.second;
    }
    return ((double)total_olap) / ((double)total_oltp);
  }

  inline double getFreshnessRatioRelation(std::string table_name) {
    auto tmp = _getFreshnessRelation(db->getTable(table_name));
    return ((double)tmp.first) / ((double)tmp.second);
  }

  inline double getFreshnessRationRelation(int table_idx) {
    auto tmp = _getFreshnessRelation(db->getTable(table_idx));
    return ((double)tmp.first) / ((double)tmp.second);
  }

 private:
  storage::Schema *db;
  scheduler::WorkerPool *worker_pool;
  txn::TransactionManager *txn_manager;

  inline std::pair<size_t, size_t> _getFreshnessRelation(
      std::shared_ptr<storage::Table> rel) {
    assert(rel->storage_layout == storage::COLUMN_STORE);

    int64_t *oltp_num_records =
        ((storage::ColumnStore *)rel.get())
            ->snapshot_get_number_tuples(false, false);  // oltp snapshot
    int64_t *olap_num_records =
        ((storage::ColumnStore *)rel.get())
            ->snapshot_get_number_tuples(true, false);  // olap snapshot
    size_t oltp_sum = 0;
    size_t olap_sum = 0;

    for (int i = 0; i < g_num_partitions; i++) {
      oltp_sum += oltp_num_records[i];
      olap_sum += olap_num_records[i];
    }

    return std::make_pair(olap_sum, oltp_sum);
  }
};

#endif /* OLTP_HPP_ */
