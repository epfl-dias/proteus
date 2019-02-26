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
const bool e_false = false;

bool CC_MV2PL::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  short n = txn_stmts->n_ops;
  /* Acquire all the locks for write in the start, if cant, abort. while
     traversing do the lookups too.
       - Acquire write locks and do the lookups also.
       - update the shit
       - release locks
   */

  /* Lookups/ inserts/ & Acquire locks for updates*/
  for (short i = 0; i < n; i++) {
    struct TXN_OP op = txn_stmts->ops[i];
    storage::Table *tbl_ptr = (storage::Table *)op.data_table;
    switch (op.op_type) {
      case OPTYPE_LOOKUP: {
        struct PRIMARY_INDEX_VAL val;
        if (tbl_ptr->p_index->find(op.key, val)) {
          if (CC_MV2PL::is_readable(val.t_min, val.t_max, xid)) {
            tbl_ptr->getRecordByKey(val.VID, val.last_master_ver);
          } else {
            VERSION_LIST vlst;
            if (tbl_ptr->getVersions(val.VID, this->curr_master, vlst)) {
              if (vlst.get_readable_ver(xid) == nullptr) {
                std::cout << "NO SUITABLE VERSION FOUND !!" << std::endl;
              }

            } else {
              std::cout << "FUCKKKK" << std::endl;
            }
          }
        } else {
          std::cout << "REC NOT FOUND: " << op.key << std::endl;
        }
        break;
      }

      case OPTYPE_UPDATE: {
        break;
      }
      case OPTYPE_INSERT: {
        uint64_t vid = tbl_ptr->insertRecord(op.rec, this->curr_master);
        struct PRIMARY_INDEX_VAL hashval(xid, vid, this->curr_master);
        tbl_ptr->p_index->insert(vid, hashval);
        break;
      }
      case OP_TYPE_DELETE:
        std::cout << "[CC_MV2PL] OP: Delete not implemented" << std::endl;
        break;
      default:
        std::cout << "[CC_MV2PL] Unknown OP: " << op.op_type << std::endl;
        break;
    }
  }

  /* perform updates
  for (short i = 0; i < n; i++) {
    struct TXN_OP op = txn_stmts->ops[i];
    storage::Table *tbl_ptr = (storage::Table *)op.data_table;
    switch (op.op_type) {
      case OPTYPE_UPDATE: {
        break;
      }
      case OPTYPE_INSERT:
      case OPTYPE_LOOKUP:
      case OP_TYPE_DELETE:
      default:
        break;
    }
  }*/

  return true;
}  // namespace txn

void acquire_lock(struct PRIMARY_INDEX_VAL &rec) {}

// bool CC_GlobalLock::execute_txn(void *stmts, uint64_t xid) {
//   struct TXN *txn_stmts = (struct TXN *)stmts;
//   short n = txn_stmts->n_ops;
//   {
//     std::unique_lock<std::mutex> lock(global_lock);
//     for (short i = 0; i < n; i++) {
//       struct TXN_OP op = txn_stmts->ops[i];
//       storage::Table *tbl_ptr = (storage::Table *)op.data_table;
//       switch (op.op_type) {
//         case OPTYPE_LOOKUP: {
//           /* basically, lookup just touches everything. */
//           /* FIXME: there should be someway to recognize the query, project
//           the
//            * appropriate columns and then return the result. maybe a class
//            * ResultParser and Result which knows the data type of returning
//            * query. for now, lets hardcode to touching the columns only.
//            * However, TPC-C will need this because it works on basis of
//            values,
//            * YCSB only touches stuff in lookup */
//           struct PRIMARY_INDEX_VAL val(-1);
//           if (tbl_ptr->p_index->find(op.key, val)) {
//             tbl_ptr->getRecordByKey(val.VID,
//                                     nullptr);  // get all columns basically
//           }

//           break;
//         }

//         case OPTYPE_UPDATE: {
//           struct PRIMARY_INDEX_VAL val(-1);
//           if (tbl_ptr->p_index->find(op.key, val)) {
//             tbl_ptr->updateRecord(val.VID, op.rec);
//           }
//           break;
//         }
//         case OPTYPE_INSERT: {
//           // uint64_t vid = tbl_ptr->p_index->insert(K &&key, Args
//           &&val...); uint64_t vid = tbl_ptr->insertRecord(op.rec); struct
//           PRIMARY_INDEX_VAL hashval(vid); tbl_ptr->p_index->insert(vid,
//           hashval); break;
//         }

//         default:
//           std::cout << "FUCK IT:" << op.op_type << std::endl;
//           break;
//       }
//     }
//   }
//   return true;
// }
}  // namespace txn
