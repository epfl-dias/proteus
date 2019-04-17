/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

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

#ifndef TXN_UTILS_HPP_
#define TXN_UTILS_HPP_

#include <iostream>

namespace txn {

enum OP_TYPE { OPTYPE_LOOKUP, OPTYPE_INSERT, OPTYPE_UPDATE };

enum QUERY_STATE { QUERY_EMPTY, QUERY_COMMITTED, QUERY_ABORTED };

struct TXN_OP {
  uint64_t key;
  OP_TYPE op_type;
  void* data_table;
  void* rec;
} __attribute__((aligned(64)));

struct TXN {
  struct TXN_OP* ops;
  short n_ops;
  enum QUERY_STATE state;

  ~TXN() { delete ops; }
} __attribute__((aligned(64)));
}  // namespace txn

#endif /* TXN_UTILS_HPP_ */
