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

#include <tuple>

#include "oltp/snapshot/snapshot_manager.hpp"

namespace oltp::snapshot {

void SnapshotMaster::snapshot() {
  for (const auto &[id, snapArenaPart] : snapshot_arenas) {
    // std::cout << key << " has value " << value << std::endl;

    LOG(INFO) << "Snapshotting entity-id: " << id;
    // IT cant be done here as the knowledge about num-records in the current-
    // snapshot are inside the Column, not here. this can be manager only,
    // but cannot create snapshots.
    //    for(const auto& arP: snapArenaPart){
    //      for(const auto& ar: arP){
    //        ar->create_snapshot({});
    //      }
    //    }
  }
}

std::tuple<size_t, SnapshotMaster::SnapshotEntity>
SnapshotMaster::registerSnapshotEntity(SnapshotTypes type, std::string name,
                                       partition_id_t n_partitions,
                                       size_t segments_per_partition) {
  size_t id = entity_id_assignment.fetch_add(1);

  assert(!entity_id_map.contains(name) &&
         "Snapshot Entity already registered.");
  entity_id_map.emplace(name, id);

  switch (type) {
    case CircularMaster:
      this->isCircularMaster_used = true;
      createCircularMasterEntity(id, n_partitions, segments_per_partition);
      break;
    case LazyMaster:
      this->isLazyMaster_used = true;
      break;
    case MVCC:
    case None:
    default:
      break;
  }

  assert(snapshot_arenas.contains(id));
  return std::make_tuple(id, snapshot_arenas[id]);
}

void SnapshotMaster::createCircularMasterEntity(size_t entity_id,
                                                partition_id_t n_partitions,
                                                size_t segments_per_partition) {
  SnapshotEntity ret;

  for (auto i = 0; i < n_partitions; i++) {
    std::vector<std::shared_ptr<aeolus::snapshot::ArenaV2>> tmp_p;

    for (auto j = 0; j < segments_per_partition; j++) {
      tmp_p.emplace_back(
          std::make_shared<aeolus::snapshot::CircularMasterArenaV2>());
    }

    ret.push_back(tmp_p);
  }

  // initialize-snapshots with zero-value
  partition_id_t pid = 0;
  for (auto &sp : ret) {
    for (auto &s : sp) {
      s->create_snapshot({0, 0, 0, pid, false});
    }
    pid++;
  }

  this->snapshot_arenas.emplace(entity_id, ret);
}
}  // namespace oltp::snapshot
