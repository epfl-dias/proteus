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
bool e_true = false;

void release_locks(
    std::vector<CC_MV2PL::PRIMARY_INDEX_VAL *> &hash_ptrs_lock_acquired) {
  for (auto c : hash_ptrs_lock_acquired) {
    c->write_lck = false;
  }
}

// MV2PL principle : Fail bloody fast :P
bool CC_MV2PL::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  short n = txn_stmts->n_ops;
  short txn_master_ver = this->curr_master;

  /* Acquire locks for updates*/

  std::vector<PRIMARY_INDEX_VAL *> hash_ptrs_lock_acquired;
  for (short i = 0; i < n; i++) {
    struct TXN_OP op = txn_stmts->ops[i];
    storage::Table *tbl_ptr = (storage::Table *)op.data_table;
    switch (op.op_type) {
      case OPTYPE_UPDATE: {
        void *tmp;
        if (!tbl_ptr->p_index->find(op.key, tmp)) {
          std::cout << "BC KEY NOT FOUND" << std::endl;
        }
        PRIMARY_INDEX_VAL *hash_ptr = (PRIMARY_INDEX_VAL *)tmp;
        if (hash_ptr->write_lck.compare_exchange_strong(e_false, true)) {
          hash_ptrs_lock_acquired.emplace_back(hash_ptr);
        } else {
          // std::cout << "ABORT" << std::endl;
          release_locks(hash_ptrs_lock_acquired);
          return false;
        }
        // hash_ptr->write_lck // acquire lock or abort
        break;
      }
      case OPTYPE_LOOKUP:
      case OPTYPE_INSERT:
      default:
        break;
    }
  }

  // perform updates / inserts
  for (short i = 0; i < n; i++) {
    struct TXN_OP op = txn_stmts->ops[i];
    storage::Table *tbl_ptr = (storage::Table *)op.data_table;
    switch (op.op_type) {
      case OPTYPE_LOOKUP: {
        void *tmp;
        if (!tbl_ptr->p_index->find(op.key, tmp)) {
          std::cout << "BLOODY KEY NOT FOUND: " << op.key << std::endl;
          break;
        }
        PRIMARY_INDEX_VAL *hash_ptr = (PRIMARY_INDEX_VAL *)tmp;
        /*std::cout << "XID: " << xid << "\t t_min:" << hash_ptr->t_min
                  << "\tt_max: " << hash_ptr->t_max << std::endl;*/
        if (CC_MV2PL::is_readable(hash_ptr->t_min, hash_ptr->t_max, xid)) {
          tbl_ptr->getRecordByKey(hash_ptr->VID, hash_ptr->last_master_ver);
        } else {
          VERSION_LIST *vlst =
              tbl_ptr->getVersions(hash_ptr->VID, txn_master_ver);
          if (vlst == nullptr || vlst->get_readable_ver(xid) == nullptr) {
            std::cout << "NO SUITABLE VERSION FOUND !!" << std::endl;
          }
        }
        break;
      }

      case OPTYPE_UPDATE: {
        void *tmp;
        if (!tbl_ptr->p_index->find(op.key, tmp)) {
          std::cout << "BC KEY NOT FOUND" << std::endl;
        }

        PRIMARY_INDEX_VAL *hash_ptr = (PRIMARY_INDEX_VAL *)tmp;
        // std::cout << "HASH: " << hash_ptr->VID << std::endl;
        // std::cout << "updating rec: " << op.key << std::endl;
        tbl_ptr->updateRecord(hash_ptr->VID, op.rec, txn_master_ver,
                              hash_ptr->last_master_ver, xid, 0);
        // std::cout << "updating meta: " << op.key << std::endl;
        hash_ptr->t_min = xid;
        hash_ptr->last_master_ver = txn_master_ver;
        // std::cout << "update done: " << op.key << std::endl;
        break;
      }
      case OPTYPE_INSERT: {
        void *hash_ptr = tbl_ptr->insertRecord(op.rec, xid, txn_master_ver);
        tbl_ptr->p_index->insert(op.key, hash_ptr);
        break;
      }
      default:
        break;
    }
  }
  release_locks(hash_ptrs_lock_acquired);

  return true;
}

// Following implementation using global locks is wrong as it doesnt care about
// seriazability
bool CC_GlobalLock::execute_txn(void *stmts, uint64_t xid) {
  struct TXN *txn_stmts = (struct TXN *)stmts;
  short txn_master_ver = this->curr_master;
  short n = txn_stmts->n_ops;
  {
    std::unique_lock<std::mutex> lock(global_lock);
    for (short i = 0; i < n; i++) {
      struct TXN_OP op = txn_stmts->ops[i];
      storage::Table *tbl_ptr = (storage::Table *)op.data_table;
      switch (op.op_type) {
        case OPTYPE_LOOKUP: {
          /* basically, lookup just touches everything. */
          /* FIXME: there should be someway to recognize the query, project
          the
           * appropriate columns and then return the result. maybe a class
           * ResultParser and Result which knows the data type of returning
           * query. for now, lets hardcode to touching the columns only.
           * However, TPC-C will need this because it works on basis of
           values,
           * YCSB only touches stuff in lookup */

          void *tmp;
          if (tbl_ptr->p_index->find(op.key, tmp)) {
            PRIMARY_INDEX_VAL *hash_ptr = (PRIMARY_INDEX_VAL *)tmp;
            tbl_ptr->getRecordByKey(hash_ptr->VID, hash_ptr->last_master_ver);
          }
          break;
        }

        case OPTYPE_UPDATE: {
          void *tmp;
          if (tbl_ptr->p_index->find(op.key, tmp)) {
            PRIMARY_INDEX_VAL *hash_ptr = (PRIMARY_INDEX_VAL *)tmp;
            // tbl_ptr->updateRecord(val.VID, op.rec);
            tbl_ptr->updateRecord(hash_ptr->VID, op.rec, txn_master_ver,
                                  hash_ptr->last_master_ver, xid, curr_master);
          }
          break;
        }
        case OPTYPE_INSERT: {
          void *hash_ptr = tbl_ptr->insertRecord(op.rec, xid, txn_master_ver);
          tbl_ptr->p_index->insert(op.key, hash_ptr);
          break;
        }

        default:
          break;
      }
    }
  }
  return true;
}
}  // namespace txn
