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

#ifndef PROTEUS_STORAGE_UTILS_HPP
#define PROTEUS_STORAGE_UTILS_HPP

#include "oltp/common/common.hpp"

namespace storage {

class StorageUtils {
  static_assert(sizeof(rowid_t) == 8,
                "Size of rowid_t is expected to be 64-bit value.");

  // Circular Master
  // | empty 1-byte | 1-byte master_ver | 1-byte partition_id | 5-byte VID |

  // LazyMaster XX.
  // | 2-byte secondary_ver | 1-byte partition_id | 5-byte VID |

 public:
  static inline rowid_t create_vid(rowid_t vid, partition_id_t partition_id,
                                   master_version_t master_ver) {
    return ((vid & 0x000000FFFFFFFFFFu) |
            ((uint64_t)(partition_id & 0x00FFu) << 40u) |
            ((uint64_t)(master_ver & 0x00FFu) << 48u));
  }

  static inline rowid_t create_vid(rowid_t vid, partition_id_t partition_id) {
    return ((vid & 0x000000FFFFFFFFFFu) |
            ((uint64_t)(partition_id & 0x00FFu) << 40u));
  }

  static inline rowid_t update_mVer(rowid_t vid,
                                    master_version_t master_version) {
    return (vid & 0xFF00FFFFFFFFFFFFu) |
           ((uint64_t)(master_version & 0x00FFu) << 48u);
  }

  static inline auto get_offset(rowid_t vid) {
    return (vid & 0x000000FFFFFFFFFFu);
  }

  static inline auto get_pid(rowid_t vid) {
    return (((vid)&0x0000FF0000000000u) >> 40u);
  }

  static inline auto get_m_version(rowid_t vid) {
    return (((vid)&0x00FF000000000000u) >> 48u);
  }

  static inline column_uuid_t get_column_uuid(table_id_t tableId,
                                              column_id_t columnId) {
    // top-16 bits for table_id
    // bottom-16 bits for column_id
    column_uuid_t ret = columnId;
    ret |= (((column_uuid_t)tableId) << 16u);
    return ret;
  }

  static inline table_id_t get_tableId_from_columnUuid(column_uuid_t columnUuid){
    return ((table_id_t)(columnUuid >> 16u));
  }

  static inline column_id_t get_columnId_from_columnUuid(column_uuid_t columnUuid){
    return ((column_id_t)(columnUuid & 0x0000FFFFu));
  }
};

}  // namespace storage

#endif  // PROTEUS_STORAGE_UTILS_HPP
