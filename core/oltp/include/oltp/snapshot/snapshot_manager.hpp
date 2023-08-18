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

#ifndef AEOLUS_SNAPSHOT_SNAPSHOT_MANAGER_HPP_
#define AEOLUS_SNAPSHOT_SNAPSHOT_MANAGER_HPP_

#include <cstdlib>

#include "oltp/common/common.hpp"
#include "oltp/snapshot/arena.hpp"
#include "oltp/snapshot/circular_master_arena.hpp"
#include "oltp/snapshot/lazy_master_arena.hpp"

namespace aeolus::snapshot {

template <typename T>
class SnapshotManager_impl {
 public:
  static void init() { T::init(); }
  static void deinit() { T::deinit(); }

  static auto create(size_t bytes) { return T::create(bytes); }

 private:
};

using SnapshotManager = SnapshotManager_impl<CircularMasterProvider>;

}  // namespace aeolus::snapshot

namespace oltp::snapshot {

// somehow keep track of the users so that once nobody wants to use it,
// we can safely mark it as this can be destroyed (snapshot can be released)
class SnapshotEntity {
  typedef std::vector<std::vector<std::shared_ptr<aeolus::snapshot::ArenaV2>>>
      EntityVector;

 public:
  auto &getPartition(partition_id_t partitionId) {
    assert(entities.size() > partitionId);
    return entities[partitionId];
  }

  // get by vid? -> probe
  // scan? -> should give pointers.. the interface in column..

  SnapshotEntity *create() { return nullptr; }

  SnapshotEntity *destroy() { return nullptr; }

 private:
  // first vector -> partition.
  // second vector -> segment within the partitions.
  const EntityVector entities;
  // const column_uuid_t columnUuid;

 private:
  SnapshotEntity(column_uuid_t columnUuid, EntityVector entity)
      : /*columnUuid(columnUuid), */ entities(entity) {}
};

class SnapshotCtx {
  SnapshotEntity &getEntity(column_uuid_t columnUuid);

 private:
  // maps table_id/col_id to index in entity vector
  std::map<column_uuid_t, uint32_t> entity_idx_map;
  std::deque<SnapshotEntity> snapshot_entities;
};

class SnapshotMaster {
 public:
  static inline SnapshotMaster &getInstance() {
    static SnapshotMaster instance;
    return instance;
  }
  SnapshotMaster(SnapshotMaster const &) = delete;  // Don't Implement
  void operator=(SnapshotMaster const &) = delete;  // Don't implement

  // typedef std::shared_ptr<std::array<aeolus::snapshot::ArenaV2,
  // MAX_PARTITIONS>>&  SnapshotEntity;

  // first vector -> partition.
  // second vector -> segment within the partitions.
  typedef std::vector<std::vector<std::shared_ptr<aeolus::snapshot::ArenaV2>>>
      SnapshotEntity;

  // returns snapshotEntity ID.
  std::tuple<size_t, SnapshotEntity> registerSnapshotEntity(
      SnapshotTypes type, std::string name, partition_id_t n_partitions,
      size_t segments_per_partition = 1);

  //  std::vector<std::pair<void*, size_t>>
  //  snapshot_get_data(size_t scan_idx,
  //                                 std::vector<RecordAttribute*>&
  //                                 wantedFields, bool olap_local, bool
  //                                 elastic_scan);

  [[maybe_unused]] inline SnapshotEntity &getSnapshotArena(size_t entity_id) {
    assert(snapshot_arenas.contains(entity_id));
    return snapshot_arenas[entity_id];
  }
  [[maybe_unused]] inline SnapshotEntity &getSnapshotArena(
      std::string entity_name) {
    assert(entity_id_map.contains(entity_name));
    return snapshot_arenas[entity_id_map[entity_name]];
  }

  [[nodiscard]] inline bool isMechanismActive(SnapshotTypes type) const {
    switch (type) {
      case CircularMaster:
        return isCircularMaster_used;
      case LazyMaster:
        return isLazyMaster_used;
      case None:
      case MVCC:
        return true;
      default:
        return false;
    }
  }

  void snapshot();

 private:
  SnapshotMaster()
      : isCircularMaster_used(false),
        isLazyMaster_used(false),
        entity_id_assignment(0) {}

  void createCircularMasterEntity(size_t entity_id, partition_id_t n_partitions,
                                  size_t segments_per_partition = 1);

 private:
  bool isCircularMaster_used;
  bool isLazyMaster_used;
  std::atomic<size_t> entity_id_assignment;

 private:
  std::map<std::string, size_t> entity_id_map{};
  std::map<size_t, SnapshotEntity> snapshot_arenas{};
};

}  // namespace oltp::snapshot

#endif /* AEOLUS_SNAPSHOT_SNAPSHOT_MANAGER_HPP_ */
