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

#include "oltp/storage/schema.hpp"

#include <platform/threadpool/threadpool.hpp>
#include <utility>

#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/storage/table.hpp"

namespace storage {

std::shared_ptr<Table> Schema::getTable(table_id_t tableId) {
  std::unique_lock<std::mutex> lk(schema_lock);
  if (tables.size() > tableId) {
    return tables.at(tableId);
  }
  return {};
}

std::shared_ptr<Table> Schema::getTable(const std::string& name) {
  std::unique_lock<std::mutex> lk(schema_lock);

  if (this->table_name_map.contains(name)) {
    return tables.at(table_name_map[name]);
  }
  return {};
}

std::shared_ptr<Table> Schema::create_table(
    const std::string& name, layout_type layout, const TableDef& columns,
    size_t initial_num_records, bool indexed, bool partitioned, int numa_idx) {
  std::unique_lock<std::mutex> lk(schema_lock);
  // auto proteusAllocator =
  // proteus::memory::PinnedMemoryAllocator<ColumnStore>();

  LOG(INFO) << "Creating table: " << name
            << " | Capacity: " << initial_num_records;

  if (numa_idx < 0)
    numa_idx =
        storage::NUMAPartitionPolicy::getInstance().getDefaultPartition();

  switch (layout) {
    case COLUMN_STORE: {
      auto tblRef = tables.emplace(
          this->num_tables,
          std::allocate_shared<ColumnStore>(
              proteus::memory::PinnedMemoryAllocator<ColumnStore>(),
              (this->num_tables), name, columns, indexed, partitioned,
              initial_num_records, numa_idx));

      table_name_map.emplace(name, this->num_tables);
      this->num_tables++;
      this->total_mem_reserved += tblRef.first->second->total_memory_reserved;
      return tblRef.first->second;
    }
    default:
      throw std::runtime_error("Unknown layout type");
  }
}

void Schema::ETL(uint numa_affinity_idx) {
  std::vector<std::thread> workers;

  for (const auto& [tblID, tbl] : this->tables) {
    auto table_id = tblID;
    workers.emplace_back([this, table_id, numa_affinity_idx]() {
      this->tables.at(table_id)->ETL(numa_affinity_idx);
    });
  }

  for (auto& th : workers) {
    th.join();
  }
}

void Schema::twinColumn_snapshot(xid_t epoch,
                                 master_version_t snapshot_master_ver) {
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

  assert(false);
  //  for (const auto& tbl : tables) {
  //    tbl->twinColumn_snapshot(epoch, snapshot_master_ver);
  //  }

  if (global_conf::num_master_versions > 1) {
    // start an async task (threadpool) to sync master..
    this->snapshot_sync = ThreadPool::getInstance().enqueue(
        &Schema::sync_master_ver_schema, this, snapshot_master_ver);
    // this->sync_master_ver_schema(snapshot_master_ver);
  }
}

bool Schema::sync_master_ver_schema(
    const master_version_t snapshot_master_ver) {
  // add arg: const scheduler::cpunumanode &exec_numanode
  snapshot_sync_in_progress.store(true);
  time_block t("[Schema]syncMasterVersions_: ");
  std::vector<std::future<bool>> sync_tasks;  // pre-allocation
  sync_tasks.reserve(this->num_tables);

  // start
  for (auto& tbl : tables) {
    sync_tasks.emplace_back(ThreadPool::getInstance().enqueue(
        &Schema::sync_master_ver_tbl, this, tbl.second, snapshot_master_ver));
  }

  // wait for finish
  for (auto& task : sync_tasks) {
    task.get();
  }

  this->snapshot_sync_in_progress.store(false);
  return true;
}

bool Schema::sync_master_ver_tbl(const std::shared_ptr<Table>& tbl,
                                 master_version_t snapshot_master_ver) {
  tbl->twinColumn_syncMasters(snapshot_master_ver);
  return true;
}

void Schema::snapshot(xid_t epoch,
                      std::vector<column_uuid_t>* snapshot_columns) {
  if (snapshot_columns != nullptr) {
    for (const auto& col_uuid : *snapshot_columns) {
      auto tableId =
          storage::StorageUtils::get_tableId_from_columnUuid(col_uuid);
      auto colId =
          storage::StorageUtils::get_columnId_from_columnUuid(col_uuid);
      getTable(tableId)->snapshot(epoch, colId);
    }
  } else {
    for (const auto& tbl : tables) {
      tbl.second->snapshot(epoch);
    }
  }
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

  for (const auto& tbl : tables) {
    // tbl->p_index->report();
    tbl.second->reportUsage();
  }
}

void Schema::teardown(const std::string& cdf_out_path) {
  std::unique_lock<std::mutex> lk(schema_lock);

  // if(!cdf_out_path.empty()){
  save_cdf("");
  //}

  for (auto dt : deltaStore) {
    delete dt;
  }

  //  for (const auto& tbl : tables) {
  //    tbl->~Table();
  //    MemoryManager::freePinned(tbl);
  //  }
  tables.clear();
  table_name_map.clear();
  this->cleaned = true;
}


void Schema::drop_table(const std::string& name) {
  std::unique_lock<std::mutex> lk(schema_lock);

  if (table_name_map.contains(name)) {
    auto table_id = table_name_map[name];
    table_name_map.erase(name);
    tables.erase(table_id);

    LOG(INFO) << "Table " << name << " (table_id: " << table_id
              << " ) dropped.";
  } else {
    throw std::runtime_error("Table not found");
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

}  // namespace storage
