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
bool e_false = false;

bool CC_MV2PL::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  short n = txn_stmts->n_ops;
  /* Question: for MV2PL, we acquire locks for all the statements in a TXN OR
   * execute the statements and then rollback if some statement get aborted
   */
  {
    for (short i = 0; i < n; i++) {
      struct TXN_OP op = txn_stmts->ops[i];
      // storage::Table *tbl_ptr = (storage::Table *)op.data_table;
      switch (op.op_type) {
        case OPTYPE_LOOKUP: {
          // get the hash val. if readable, get the actual rec by VID
          // if not readable, traverse the linked-list
          break;
        }

        case OPTYPE_UPDATE: {
          // basically delete the last version and insert new version
          // additionally keep track of updated records for GC?

          // modified_vids.emplace_back(VID);
          break;
        }
        case OPTYPE_INSERT: {
          /* insert a record with t_min as current txn_id, t_max = 0*/
          // uint64_t vid = tbl_ptr->insertRecord(op.rec);
          // struct PRIMARY_INDEX_VAL hashval(xid, vid);
          // tbl_ptr->p_index->insert(vid, hashval);
          break;
        }
        case OP_TYPE_DELETE: {
          /* set tmax of record to curre txn id*/

          /*try {
            // auto &val = tbl_ptr->p_index->find(op.key);

              bool success = tbl_ptr->p_index->
              //update_fn(op.key, [hasval] {});

              if (is_record_visible(&val, xid) &&
                  val.write_lck.compare_exchange_strong(e_false, true)) {
                // write lock acquired.
                // update the hash value actually with released lock

              } else {
                return false;
              }

            } catch (const std::out_of_range &e) {
              return false;
            }*/
        }

        default:
          std::cout << "FUCK IT:" << op.op_type << std::endl;
          break;
      }
    }
  }
  return true;
}

void acquire_lock(struct PRIMARY_INDEX_VAL &rec) {}

inline bool CC_MV2PL::is_record_visible(struct PRIMARY_INDEX_VAL *rec,
                                        uint64_t tid_self) {
  if ((tid_self >= rec->t_min) && tid_self < rec->t_max) return true;
  return false;
}

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
