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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <unistd.h>
#include <functional>
#include <iostream>
#include <tuple>

#include "transactions/transaction_manager.hpp"

std::ostream& operator<<(std::ostream& o, const struct txn::TXN& a) {
  o << "---TXN---\n";
  for (int i = 0; i < a.n_ops; i++) {
    o << "\tTXN[" << i << "]";
    o << "\t\tOP: ";
    switch (a.ops[i].op_type) {
      case txn::OPTYPE_LOOKUP:
        o << " LOOKUP";
        break;
      case txn::OPTYPE_INSERT:
        o << " INSERT";
        break;
      case txn::OPTYPE_UPDATE:
        o << " UPDATE";
        break;
      case txn::OP_TYPE_DELETE:
        o << " DELETE";
        break;
      default:
        o << " UNKNOWN";
        break;
    }
    o << ", key: " << a.ops[i].key << std::endl;
    ;
  }

  o << "---------\n";
  return o;
}

#endif /* UTILS_HPP_ */
