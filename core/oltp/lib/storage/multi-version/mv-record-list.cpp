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

#include "oltp/storage/multi-version/mv-record-list.hpp"

#include <iostream>
#include <platform/memory/allocator.hpp>

#include "oltp/common/constants.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/transaction_manager.hpp"

#define STEAM_EPO_PERIODIC_FETCH_INTERVAL_MS 5

namespace storage::mv {

template <typename MV_TYPE, typename VERSION_TYPE>
static void steamGC(global_conf::IndexVal* idx_ptr, txn::TxnTs global_min) {
  idx_ptr->latch.acquire();
  if (!idx_ptr->delta_list.isValid()) {
    idx_ptr->latch.release();
    return;
  }

  auto& delta_list =
      reinterpret_cast<TaggedDeltaDataPtr<VERSION_TYPE>&>(idx_ptr->delta_list);

  TaggedDeltaDataPtr<VERSION_TYPE> cleanable;

  if (global_conf::ConcurrencyControl::is_readable(*idx_ptr, global_min)) {
    // as the top-version is visible, the entire delta-chain is cleanable
    cleanable = delta_list.reset();
    idx_ptr->latch.release();
  } else {
    // find the first-version which is cleanable, that is, version after visible

    const auto* versionBase =
        VersionChain<MV_TYPE>::get_readable_ver(global_min, delta_list);
    if (versionBase == nullptr || !versionBase->isValid()) {
      idx_ptr->latch.release();
      return;
    }
    // prev->next actually points to first cleanable.
    // take ownership of prev->next, and make it nullptr
    cleanable = versionBase->typePtr()->next.reset();
    idx_ptr->latch.release();
  }

  while (cleanable.isValid()) {
    auto x = cleanable.typePtr()->next.reset();
    cleanable.release();
    cleanable = std::move(x);
  }
}

void steamConsolidation(global_conf::IndexVal* idx_ptr,
                        const TaggedDeltaDataPtr<VersionSingle>& start) {
#if STEAM_EPO_PERIODIC_FETCH_INTERVAL_MS > 0
  // Retrieve active txn list periodically only as per SteamGC paper:
  static thread_local std::vector<xid_t> activeTxnList =
      txn::TransactionManager::getInstance().get_all_activeTxn();
  static thread_local auto last_fetch = std::chrono::system_clock::now();

  auto now = std::chrono::system_clock::now();
  if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fetch)
          .count() >= STEAM_EPO_PERIODIC_FETCH_INTERVAL_MS) {
    activeTxnList = txn::TransactionManager::getInstance().get_all_activeTxn();
  }

#else
  auto activeTxnList =
      txn::TransactionManager::getInstance().get_all_activeTxn();
#endif

  auto record_t_min = idx_ptr->ts.t_min;
  // assert(idx_ptr->latch.try_acquire() == false);

  const auto* curr = &start;

  size_t n_consolidations = 0;

  for (auto activeTxn : activeTxnList) {
    // if activeTxn visible version is top of head then entire list is cleanable
    if (activeTxn >= record_t_min) {
      break;
    }
    if (curr == nullptr || !(curr->isValid())) {
      break;
    }
    if (global_conf::ConcurrencyControl::is_readable(
            record_t_min, {activeTxn << 27u, activeTxn})) {
      // meaning, global record is visible.

      // micro-opt: then clean the remaining list also?
      //      auto cleanable = curr->typePtr()->next.reset();
      //      while (cleanable.isValid()) {
      //        auto x = cleanable.typePtr()->next.reset();
      //        cleanable.release();
      //        cleanable = std::move(x);
      //      }

      break;
    }

    // retrieve first visible version by activeTxn
    auto* visible = VersionChain<MV_RecordList_Full>::get_readable_ver(
        {activeTxn << 27u, activeTxn}, *curr);
    if (visible == nullptr || !(visible->isValid())) {
      break;
    }
    if (curr == visible) {
      break;
    }

    // prune obsolete in-between versions (curr, visible)
    auto currTypePtr = curr->typePtr();
    while (currTypePtr->next != *visible) {
      // TODO: additional merge step for attribute-levelMV

      assert(currTypePtr->next.typePtr()->next.isValid());
      auto tmp = currTypePtr->next.reset();
      auto tmpTypePtr = tmp.typePtr();
      assert(tmp.isValid());
      assert(tmpTypePtr);
      currTypePtr->next = std::move(tmpTypePtr->next);
      tmp.release();

      n_consolidations++;
    }

    // update current version in iterator
    curr = visible;
  }
  LOG_IF(INFO, n_consolidations > 0)
      << "Consolidation by " << std::this_thread::get_id() << ": "
      << n_consolidations;
}

void MV_RecordList_Full::gc(global_conf::IndexVal* idx_ptr,
                            txn::TxnTs global_min) {
  steamGC<MV_RecordList_Full, version_t>(idx_ptr, global_min);
}

std::bitset<1> MV_RecordList_Full::get_readable_version(
    const global_conf::IndexVal& index_ptr, const txn::TxnTs& txTs,
    char* write_loc, const Table& tableRef, const column_id_t* col_idx,
    const short num_cols) {
  static thread_local std::bitset<1> ret_bitmask("1");

  auto& delta_list_ptr = reinterpret_cast<const TaggedDeltaDataPtr<version_t>&>(
      index_ptr.delta_list);

  const TaggedDeltaDataPtr<version_t>* versionBase = nullptr;
  TaggedDeltaDataPtr<version_t> instanceCrossingCopy;

  if constexpr (GcMechanism == GcTypes::OneShot && OneShot_CONSOLIDATION) {
    // NOTE: with instance consolidation, the head can also be null as it may
    //  get GC'ed given the condition.

    instanceCrossingCopy =
        VersionChain<MV_RecordList_Full>::get_version_with_consolidation(
            txTs, delta_list_ptr,
            storage::StorageUtils::get_row_uuid(tableRef.table_id,
                                                index_ptr.VID));
    versionBase = &instanceCrossingCopy;
  } else {
    LOG_IF(FATAL, !(delta_list_ptr.isValid()))
        << std::this_thread::get_id() << " delta-tag verification failed "
        << txTs.txn_start_time << " | " << index_ptr.ts.t_min << " | "
        << StorageUtils::get_offset(index_ptr.VID);

    versionBase = VersionChain<MV_RecordList_Full>::get_readable_ver(
        txTs, delta_list_ptr);
  }

  assert(versionBase != nullptr);
  auto* version = versionBase->typePtr();
  assert(version != nullptr);
  auto* version_data = reinterpret_cast<char*>(version->data);
  assert(version_data != nullptr);

  if (index_ptr.ts.deleted) {
    // verify the deleted record access check.
    //  -> check if the version was deleted, if yes, throw.
    throw std::runtime_error("Record does not exists");
  }

  if (__unlikely(col_idx == nullptr || num_cols == 0)) {
    for (const auto& col_s_pair : tableRef.column_size_offset_pairs) {
      memcpy(write_loc, version_data + col_s_pair.second, col_s_pair.first);
      write_loc += col_s_pair.first;
    }

  } else {
    // copy the required attr
    for (auto i = 0; i < num_cols; i++) {
      const auto& col_s_pair = tableRef.column_size_offset_pairs[col_idx[i]];
      // assumption: full row is in the version.
      memcpy(write_loc, version_data + col_s_pair.second, col_s_pair.first);
      write_loc += col_s_pair.first;
    }
  }
  return ret_bitmask;
}

std::vector<MV_RecordList_Full::version_t*> MV_RecordList_Full::create_versions(
    xid_t xid, global_conf::IndexVal* idx_ptr, const Table& tableRef,
    storage::DeltaStore& deltaStore, partition_id_t partition_id,
    const column_id_t* col_idx, short num_cols) {
  size_t ver_size = tableRef.record_size;
  size_t total_size = ver_size + sizeof(MV_RecordList_Full::version_t);

  //  total_size += total_size % 64;
  // if (total_size < 64) total_size = 64;

  auto deltaDataPtr =
      deltaStore.allocate(total_size, partition_id, idx_ptr->ts.t_min);

  auto* base_ptr = deltaDataPtr.ptr();
  assert(base_ptr);

  // FIXME: pushing deleted versions here,
  //  for now just passing t_min, but this should check if the version was
  //  deleted, and if yes, then mark it as tmax thing.

  auto* ver = new (base_ptr) MV_RecordList_Full::version_t(
      idx_ptr->ts.t_min, 0,
      base_ptr + alignof(MV_RecordList_Full::version_t) +
          sizeof(MV_RecordList_Full::version_t),
      ver_size);
  assert(ver->size == ver_size);

  // point next version to previous head, regardless it is valid or not.
  if (idx_ptr->delta_list.isValid()) {
    ver->next = std::move(idx_ptr->delta_list);
    if constexpr (GcMechanism == GcTypes::OneShot && OneShot_CONSOLIDATION) {
      if (deltaDataPtr.get_delta_idx() != ver->next.get_delta_idx()) {
        ver->next.saveInstanceCrossingPtr(storage::StorageUtils::get_row_uuid(
            tableRef.table_id, idx_ptr->VID));
      }
    }
    if constexpr (SteamGC_CONSOLIDATION) {
      if (ver->next.isValid()) steamConsolidation(idx_ptr, ver->next);
    }
  }

  // update pointer on delta_list (which is inside index) to point to latest
  // version.
  idx_ptr->delta_list = std::move(deltaDataPtr);
  // assert(idx_ptr->delta_list.isValid());
  // assert(ver != nullptr && ver->data != nullptr);
  // assert(ver->size == ver_size);
  return {ver};
}

void MV_RecordList_Partial::gc(global_conf::IndexVal* idx_ptr,
                               txn::TxnTs gmin) {
  steamGC<MV_RecordList_Partial, version_t>(idx_ptr, gmin);
}

std::vector<MV_RecordList_Partial::version_t*>
MV_RecordList_Partial::create_versions(
    xid_t xid, global_conf::IndexVal* idx_ptr, const Table& tableRef,
    storage::DeltaStore& deltaStore, partition_id_t partition_id,
    const column_id_t* col_idx, short num_cols) {
  std::bitset<64> attr_mask;
  std::vector<uint16_t> ver_offsets{64, 0};

  auto ver_data_size = MV_RecordList_Partial::version_t::get_partial_mask_size(
      tableRef.column_size, ver_offsets, attr_mask, col_idx, num_cols);

  auto offset_arr_sz = attr_mask.count() * sizeof(uint16_t);

  auto deltaDataPtr = deltaStore.allocate(
      ver_data_size + offset_arr_sz + sizeof(MV_RecordList_Partial::version_t),
      partition_id);

  auto* base_ptr = deltaDataPtr.ptr();
  assert(base_ptr);

  auto* ver = new (base_ptr) MV_RecordList_Partial::version_t(
      idx_ptr->ts.t_min, 0, base_ptr + sizeof(MV_RecordList_Partial::version_t),
      ver_data_size);

  // point next version to previous head, regardless it is valid or not.
  if (idx_ptr->delta_list.isValid()) {
    ver->next = std::move(idx_ptr->delta_list);

    if (SteamGC_CONSOLIDATION || OneShot_CONSOLIDATION) {
      LOG(FATAL)
          << "Consolidation not implemented for partial MV_RecordList_Partial";
    }
  }

  // update pointer on delta_list (which is inside index) to point to latest
  // version.
  idx_ptr->delta_list = std::move(deltaDataPtr);

  //  assert(idx_ptr->delta_list.isValid());
  //  assert(ver != nullptr && ver->data != nullptr);

  char* offset_arr = reinterpret_cast<char*>(ver->data) + ver_data_size;

  ver->create_partial_mask((uint16_t*)offset_arr, attr_mask);

  memcpy(offset_arr, ver_offsets.data(), offset_arr_sz);
  return {ver};
}

std::bitset<64> MV_RecordList_Partial::get_readable_version(
    const global_conf::IndexVal& index_ptr, const txn::TxnTs& txTs,
    char* write_loc, const Table& tableRef, const column_id_t* col_idx,
    short num_cols) {
  // FIXME: as with oneshot-instance consolidation, the next ptr might not be
  //  valid, we need to implement getNextVersion functionality, which will deal
  //  with breaking chains also. ALSO: ensure that all the attributes
  //  (undo-images are not GC'ed: STEAM's consolidation/merge check)

  auto& tmpTagPtr = reinterpret_cast<const TaggedDeltaDataPtr<version_t>&>(
      index_ptr.delta_list);

  auto* version_ptr = tmpTagPtr.typePtr();

  LOG_IF(FATAL, version_ptr == nullptr) << "delta-tag verification failed";

  // CREATE containment mask.
  std::bitset<64> done_mask;
  std::bitset<64> required_mask;

  // keep a thread-local allocated vector to remove allocations
  // on the critical path: removed static thread_local for now.
  std::vector<uint16_t, proteus::memory::PinnedMemoryAllocator<uint16_t>>
      return_col_offsets(64, 0);

  if (__likely(num_cols > 0 && col_idx != nullptr)) {
    assert(num_cols <= 64 && "MAX columns supported: 64");
    done_mask.set();
    for (auto i = 0, offset = 0; i < num_cols; i++) {
      done_mask.reset(col_idx[i]);
      // required_mask.set(col_idx[i]);

      offset += tableRef.column_size_offset_pairs[col_idx[i]].first;
      return_col_offsets[i] = offset;
    }
  } else {
    for (auto i = tableRef.column_size_offset_pairs.size();
         i < done_mask.size(); i++) {
      done_mask.reset(i);
    }
    // offsets are not set in this case!
  }
  required_mask = ~done_mask;

  assert(!(done_mask.all()) && "haven't even started and its done?");

  while (version_ptr != nullptr) {
    // The problem with before-images is that you have to traverse the snapshot
    // until the visible part, no matter the required attribute is there or not.

    // before-images copy
    // The following, undo the record image, for the set of required attributes
    // only.
    auto tmp = version_ptr->attribute_mask & (~done_mask);
    if (tmp.any()) {
      // set the bits that what we have it.
      done_mask |= tmp;

      // so this version contains some of the required stuff.
      // TODO: use intrinsic to find first set bit and start i from there.
      for (auto i = 0; i < tmp.size(); i++) {
        if (!(tmp[i])) {
          continue;
        }
        auto& col_s_pair = tableRef.column_size_offset_pairs[i];

        // Offset of some column in the version itself
        auto version_col_offset = version_ptr->get_offset(i);

        // Offset of column in the requested set of columns.
        // if requested all columns, then its columns cumulative offset.
        auto offset_idx_output =
            (num_cols == 0)
                ? col_s_pair.second
                : return_col_offsets
                      [(required_mask >> (required_mask.size() - i)).count()];

        assert(version_ptr->data != nullptr);
        memcpy((write_loc + offset_idx_output),
               static_cast<char*>(version_ptr->data) + version_col_offset,
               col_s_pair.first);
      }
    }

    // END before-images copy
    // NOTE: seems buggy
    if (version_ptr->t_min == txTs.txn_id ||
        version_ptr->t_min < txTs.txn_start_time) {
      break;
    } else {
      version_ptr = version_ptr->next.typePtr();
    }
  }
  return done_mask;
}

}  // namespace storage::mv
