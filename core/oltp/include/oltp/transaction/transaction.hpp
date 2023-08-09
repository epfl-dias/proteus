/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_TRANSACTION_HPP
#define PROTEUS_TRANSACTION_HPP

#include <cassert>
#include <functional>
#include <memory>
#include <platform/util/erase-constructor-idioms.hpp>
#include <vector>

#include "oltp/common/common.hpp"

namespace txn {

class Txn;

class TxnTs {
 public:
  xid_t txn_start_time;
  xid_t txn_id;

  TxnTs(xid_t txn_id, xid_t txn_start_time)
      : txn_id(txn_id), txn_start_time(txn_start_time) {}
  explicit TxnTs(std::pair<xid_t, xid_t> txnId_startTime_pair)
      : txn_id(txnId_startTime_pair.first),
        txn_start_time(txnId_startTime_pair.second) {}

  struct TxnTsCmp {
    bool operator()(const TxnTs& a, const TxnTs& b) const {
      return a.txn_start_time < b.txn_start_time;
    }
  };

  friend class Txn;
};

class Txn : proteus::utils::remove_copy {
 public:
  Txn(Txn&& other) = default;
  Txn& operator=(Txn&& other) = default;

  Txn(TxnTs txnTs, worker_id_t workerId, partition_id_t partitionId,
      master_version_t master_version, bool read_only = false);

  Txn(TxnTs txnTs, worker_id_t workerId, partition_id_t partitionId,
      bool read_only = false);

  Txn(worker_id_t workerId, partition_id_t partitionId, bool read_only = false);

 public:
  static Txn getTxn(worker_id_t workerId, partition_id_t partitionId,
                    bool read_only = false);
  static xid_t getTxn(Txn* txnPtr, worker_id_t workerId,
                      partition_id_t partitionId, bool read_only = false);

  static std::unique_ptr<Txn> make_unique(worker_id_t workerId,
                                          partition_id_t partitionId,
                                          bool read_only = false);

  struct [[maybe_unused]] TxnCmp {
    bool operator()(const Txn& a, const Txn& b) const {
      return a.txnTs.txn_start_time < b.txnTs.txn_start_time;
    }
  };

 public:
  TxnTs txnTs;
  bool read_only;
  worker_id_t worker_id;
  partition_id_t partition_id;

  master_version_t master_version;
  delta_id_t delta_version;

  xid_t commit_ts{};

  std::vector<row_uuid_t> undoLogVector;
};

}  // namespace txn

#endif  // PROTEUS_TRANSACTION_HPP
