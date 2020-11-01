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

#include "oltp/storage/table.hpp"

#include <cassert>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <olap/plan/catalog-parser.hpp>
#include <olap/values/expressionTypes.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/timing.hpp>
#include <string>

#include "oltp/common/constants.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage {

void Schema::ETL(uint numa_node_idx) {
  std::vector<std::thread> workers;

  for (const auto& tbl : tables) {
    workers.emplace_back([tbl, numa_node_idx]() { tbl->ETL(numa_node_idx); });
  }

  for (auto& th : workers) {
    th.join();
  }
}

void Schema::snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
  // check if prev sync is in progress, if yes, wait for that too complete.
  LOG(INFO) << "------------------------";
  LOG(INFO) << "snap_epoch: " << epoch;

  if (snapshot_sync_in_progress.load()) {
    LOG(INFO) << "Already in progress: " << epoch;
    // snapshot_sync.get();
    while (snapshot_sync_in_progress.load())
      ;
    LOG(INFO) << "Done - snap_mver: " << (uint)snapshot_master_ver;
  }

  for (auto& tbl : tables) {
    tbl->snapshot(epoch, snapshot_master_ver);
  }

  if (global_conf::num_master_versions > 1) {
    // start an async task (threadpool) to sync master..
    this->snapshot_sync = ThreadPool::getInstance().enqueue(
        &Schema::sync_master_ver_schema, this, snapshot_master_ver);
    // this->sync_master_ver_schema(snapshot_master_ver);
  }
}

bool Schema::sync_master_ver_schema(const uint8_t snapshot_master_ver) {
  // add arg: const scheduler::cpunumanode &exec_numanode
  snapshot_sync_in_progress.store(true);
  time_block t("[Schema]syncMasterVersions_: ");
  std::vector<std::future<bool>> sync_tasks;  // pre-allocation
  sync_tasks.reserve(this->num_tables);

  // start
  for (auto& tbl : tables) {
    sync_tasks.emplace_back(ThreadPool::getInstance().enqueue(
        &Schema::sync_master_ver_tbl, this, tbl, snapshot_master_ver));
  }

  // wait for finish
  for (auto& task : sync_tasks) {
    task.get();
  }

  this->snapshot_sync_in_progress.store(false);
  return true;
}

bool Schema::sync_master_ver_tbl(const storage::Table* tbl,
                                 const uint8_t snapshot_master_ver) {
  // TODO: set exec location
  // add arg: const scheduler::cpunumanode &exec_numanode
  // sync
  assert(tbl->storage_layout == COLUMN_STORE);
  ((storage::ColumnStore*)tbl)->sync_master_snapshots(snapshot_master_ver);

  return true;
}

void Schema::memoryReport() const {
  LOG(INFO) << "Total Memory Reserved for Tables: "
            << (double)this->total_mem_reserved / (1024 * 1024 * 1024) << " GB"
            << std::endl;
  LOG(INFO) << "Total Memory Reserved for Deltas: "
            << (double)this->total_delta_mem_reserved / (1024 * 1024 * 1024)
            << " GB" << std::endl;
}

void Schema::report() {
  //  for (int i = 0; i < global_conf::num_delta_storages; i++) {
  //    deltaStore[i]->print_info();
  //  }

  for (auto& tbl : tables) {
    // tbl->p_index->report();
    tbl->reportUsage();
  }
}

void Schema::add_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id) {
  deltaStore[ver]->increment_reader(epoch, worker_id);
}

void Schema::remove_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id) {
  deltaStore[ver]->decrement_reader(epoch, worker_id);
}

void Schema::switch_delta(ushort prev, ushort curr, uint64_t epoch,
                          uint8_t worker_id) {
  deltaStore[prev]->decrement_reader(epoch, worker_id);
  // either add a barrier here or inside delta storage.

  deltaStore[curr]->increment_reader(epoch, worker_id);
}

void Schema::teardown(const std::string& cdf_out_path) {
  // if(!cdf_out_path.empty()){
  save_cdf("");
  //}

  for (auto& dt : deltaStore) {
    dt->~DeltaStore();
  }

  for (auto& tbl : tables) {
    tbl->~Table();
    MemoryManager::freePinned(tbl);
  }
}

std::vector<Table*> Schema::getAllTables() { return tables; }

Table* Schema::getTable(const int idx) { return tables.at(idx); }

Table* Schema::getTable(const std::string& name) {
  // TODO: a better way would be to store table-idx mapping in a hashmap from
  // STL.

  for (const auto& t : tables) {
    if (name == t->name) return t;
  }
  return nullptr;
}

/* returns pointer to the table */
Table* Schema::create_table(std::string name, layout_type layout,
                            ColumnDef columns, uint64_t initial_num_records,
                            bool indexed, bool partitioned, int numa_idx) {
  Table* tbl = nullptr;

  if (numa_idx == -1) {
    numa_idx =
        storage::NUMAPartitionPolicy::getInstance().getDefaultPartition();
  }

  if (layout == COLUMN_STORE) {
    void* obj_ptr =
        MemoryManager::mallocPinnedOnNode(sizeof(ColumnStore), numa_idx);

    tbl = new (obj_ptr)
        ColumnStore((this->num_tables + 1), name, columns, initial_num_records,
                    indexed, partitioned, numa_idx);

  } else if (layout == ROW_STORE) {
    // void* obj_ptr =
    //     MemoryManager::alloc(sizeof(RowStore),
    //     storage::NUMAPartitionPolicy::getInstance().getDefaultPartition());

    // tbl = new RowStore((this->num_tables + 1), name, columns,
    //                    initial_num_records, indexed, partitioned, numa_idx);
    assert(false);
  } else {
    throw new std::runtime_error("Unknown layout type");
  }
  tables.push_back(tbl);
  this->num_tables++;
  this->total_mem_reserved += tbl->total_mem_reserved;

  return tbl;
}

void Schema::drop_table(Table* table) {
  this->total_mem_reserved -= table->total_mem_reserved;
  this->num_tables--;

  table->~Table();
  MemoryManager::freePinned(table);
}

void Schema::drop_table(const std::string& name) {
  int index = -1;
  for (const auto& t : tables) {
    if (name == t->name) {
      drop_table(t);
    }
  }
}

void Schema::save_cdf(const std::string& out_path) {
  proteus::utils::PercentileRegistry::for_each(
      [](std::string key, proteus::utils::Percentile* p, void* args) {
        LOG(INFO) << "\t\tP50\tP90\tP99: " << p->nth(50) << "\t" << p->nth(90)
                  << "\t" << p->nth(99);

        auto* path = (std::string*)args;
        if (!(path->empty())) {
          p->save_cdf(*path + key + ".cdf");
        }
      },
      (void*)&out_path);
}

void Table::reportUsage() {
  std::cout << "Table: " << this->name << std::endl;
  for (int i = 0; i < g_num_partitions; i++) {
    auto curr = vid[i].load();
    double percent =
        ((double)curr / ((double)(initial_num_recs / g_num_partitions))) * 100;

    std::cout << "P" << i << ": " << curr << " / "
              << (initial_num_recs / g_num_partitions) << " | " << percent
              << "%" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& out,
                         const NUMAPartitionPolicy::TablePartition& r) {
  out << "\tPID: " << r.pid << "|\t NUMA Index: " << r.numa_idx << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream& out, const NUMAPartitionPolicy& r) {
  out << "---------------------------" << std::endl;
  out << "NUMA Partition Policy" << std::endl;
  out << "\treverse_numa_mapping: "
      << (global_conf::reverse_partition_numa_mapping ? "True" : "False")
      << std::endl;
  out << "---------------------------" << std::endl;
  out << "Default Partition:" << std::endl;
  out << r.PartitionVector[0];
  out << "---------------------------" << std::endl;
  for (const auto& p : r.PartitionVector) {
    out << p;
  }
  out << "---------------------------" << std::endl;
  return out;
}

ExpressionType* getProteusType(
    const std::tuple<std::string, data_type, size_t, void*>& col) {
  switch (std::get<1>(col)) {
    case INTEGER: {
      switch (std::get<2>(col)) {
        case 4:
          return new IntType();
        case 8:
          return new Int64Type();
        default: {
          auto msg = std::string{"Unknown integer type of size: "} +
                     std::to_string(std::get<2>(col));
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case FLOAT: {
      switch (std::get<2>(col)) {
        case 8:
          return new FloatType();
        default: {
          auto msg = std::string{"Unknown float type of size: "} +
                     std::to_string(std::get<2>(col));
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case VARCHAR:
    case STRING: {
      return new DStringType(new std::map<int, std::string>());
    }
    case DSTRING: {
      if (std::get<3>(col) == nullptr) {
        auto msg = std::string{"Column[" + std::get<0>(col) +
                               "] with type DSTRING with no dictionary."};
        LOG(ERROR) << msg;
        throw std::runtime_error(msg);
      }
      // std::map<int, std::string> *d = new std::map<int, std::string>;
      return new DStringType(std::get<3>(col));
    }
    case DATE: {
      switch (std::get<2>(col)) {
        case 8:
          return new DateType();
        default: {
          auto msg = std::string{"Unknown date type of size: "} +
                     std::to_string(std::get<2>(col));
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case MV:
    case META: {
      auto msg = std::string{"Illegal  type"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    default: {
      auto msg = std::string{"Unknown type"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
  }
}

static std::mutex m_catalog;

Table::Table(std::string name, uint8_t table_id, layout_type storage_layout,
             ColumnDef columns)
    : name(name),
      table_id(table_id),
      total_mem_reserved(0),
      storage_layout(storage_layout) {
  for (int i = 0; i < topology::getInstance().getCpuNumaNodeCount(); i++) {
    vid.emplace_back();
    vid[i].store(0);
  }

  std::vector<RecordAttribute*> attrs;
  attrs.reserve(columns.size());
  for (const auto& t : columns.getColumns()) {
    attrs.emplace_back(
        new RecordAttribute(name, std::get<0>(t), getProteusType(t)));
  }

  auto exprType = new BagType(
      *(new RecordType(attrs)));  // new and derefernce is needed due to the
                                  // BagType getting a reference

  std::lock_guard<std::mutex> lock{m_catalog};
  // FIXME: the table should not require knowledge of all the plugin types
  //  one possible solution is to register the tables "variants" when asking
  //  for the plugin, just before the plan creation, where either way the
  //  scheduler knows about the available plugins
  for (const auto& s :
       {"block-local", "block-remote", "block-elastic", "block-elastic-ni"}) {
    auto tName = name + "<" + s + ">";
    LOG(INFO) << "Registering table " << tName << " to OLAP";
    CatalogParser::getInstance().registerInput(tName, exprType);
  }
}

Table::~Table() {}

};  // namespace storage
