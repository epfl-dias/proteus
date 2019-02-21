/*
                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "transactions/cc.hpp"

#include "storage/table.hpp"

namespace txn {
bool CC_GlobalLock::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  short n = txn_stmts->n_ops;
  {
    std::unique_lock<std::mutex> lock(global_lock);
    for (short i = 0; i < n; i++) {
      struct TXN_OP op = txn_stmts->ops[i];
      storage::Table *tbl_ptr = (storage::Table *)op.data_table;
      switch (op.op_type) {
        case OPTYPE_LOOKUP:
          /* basically, lookup just touches everything. */
          /* FIXME: there should be someway to recognize the query, project the
           * appropriate columns and then return the result. maybe a class
           * ResultParser and Result which knows the data type of returning
           * query. for now, lets hardcode to touching the columns only.
           * However, TPC-C will need this because it works on basis of values,
           * YCSB only touches stuff in lookup */
          struct PRIMARY_INDEX_VAL val;
          if (tbl_ptr->p_index->find(op.key, val)) {
            tbl_ptr->getRecordByKey(val.VID,
                                    nullptr);  // get all columns basically
          }

          break;

        case OPTYPE_UPDATE:
          break;

        case OPTYPE_INSERT:
          // uint64_t vid = tbl_ptr->p_index->insert(K &&key, Args &&val...);
          break;

        default:
          std::cout << "FUCK IT" << std::endl;
          break;
      }
    }
  }

  return true;
}
}  // namespace txn
