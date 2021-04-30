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

namespace txn {

class Txn;

class TxnTs {
 public:
  const xid_t txn_id;
  const xid_t txn_start_time;

 private:
  TxnTs(xid_t txn_id, xid_t txn_start_time)
      : txn_id(txn_id), txn_start_time(txn_start_time) {}
  TxnTs(std::pair<xid_t, xid_t> txnId_startTime_pair)
      : txn_id(txnId_startTime_pair.first),
        txn_start_time(txnId_startTime_pair.second) {}

  static TxnTs getTimestamps();

  friend class Txn;
};

class Txn {
 public:
  const TxnTs txnTs;
  const void* stmts;

 public:
  Txn(void* stmts) : txnTs(TxnTs::getTimestamps()), stmts(stmts) {}

  // for bench pre-runners
  Txn(void* stmts, xid_t xid) : txnTs(xid, xid), stmts(stmts) {}

  xid_t getCommitTs();
};

}  // namespace txn

#endif  // PROTEUS_TRANSACTION_HPP
