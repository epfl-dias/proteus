/*
                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#include "transactions/cc.hpp"

#include "storage/table.hpp"

namespace txn {

bool CC_GlobalLock::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  // std::cout << "\t\tTXN EXECUTE-START-CC" << std::endl;
  short n = txn_stmts->n_ops;
  {
    std::unique_lock<std::mutex> lock(global_lock);
    // std::cout << "\t\tTXN EXECUTE ACQUIRED LOCK" << std::endl;
    for (short i = 0; i < n; i++) {
      // std::cout << "\t\tTXN EXECUTE LOOP OP-" << i << std::endl;
      struct TXN_OP op = txn_stmts->ops[i];
      // std::cout << "\t\tTXN EXECUTE GOT OP FROM STMTS-" << i << std::endl;
      storage::Table *tbl_ptr = (storage::Table *)op.data_table;
      // std::cout << "\t\tTXN EXECUTE GOT TBL PTR FROM STMTS-" << i <<
      // std::endl;
      switch (op.op_type) {
        case OPTYPE_LOOKUP: {
          std::cout << "\t\tTXN EXECUTE OP TYPE LOOKUP KEY:" << op.key
                    << std::endl;
          /* basically, lookup just touches everything. */
          /* FIXME: there should be someway to recognize the query, project the
           * appropriate columns and then return the result. maybe a class
           * ResultParser and Result which knows the data type of returning
           * query. for now, lets hardcode to touching the columns only.
           * However, TPC-C will need this because it works on basis of values,
           * YCSB only touches stuff in lookup */
          struct PRIMARY_INDEX_VAL val(-1);
          if (tbl_ptr->p_index->find(op.key, val)) {
            std::cout << "\t\tFOUND THE KEY" << std::endl;
            tbl_ptr->getRecordByKey(val.VID,
                                    nullptr);  // get all columns basically
          }

          break;
        }

        case OPTYPE_UPDATE: {
          std::cout << "\t\tTXN EXECUTE OP TYPE  UPD KEY:" << op.key
                    << std::endl;
          break;
        }
        case OPTYPE_INSERT: {
          // uint64_t vid = tbl_ptr->p_index->insert(K &&key, Args &&val...);
          uint64_t vid = tbl_ptr->insertRecord(op.rec);
          struct PRIMARY_INDEX_VAL hashval(vid);
          tbl_ptr->p_index->insert(vid, hashval);
          break;
        }

        default:
          std::cout << "FUCK IT:" << op.op_type << std::endl;
          break;
      }
      std::cout << "end for loop" << std::endl;
    }
  }
  std::cout << "txn-end" << std::endl;
  return true;
}
}  // namespace txn
