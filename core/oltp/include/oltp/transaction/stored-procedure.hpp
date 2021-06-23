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

#ifndef PROTEUS_STORED_PROCEDURE_HPP
#define PROTEUS_STORED_PROCEDURE_HPP

#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/txn-executor.hpp"

namespace txn {

using txnSignature =
    std::function<bool(TransactionExecutor& executor, Txn& txn, void* params)>;

class StoredProcedure {
 public:
  const txnSignature tx;
  const bool readOnly;
  void* params{};

  explicit StoredProcedure(txnSignature tx, void* params = nullptr,
                           bool readOnly = false)
      : tx(std::move(tx)), readOnly(readOnly), params(params) {}
};

}  // namespace txn

#endif  // PROTEUS_STORED_PROCEDURE_HPP
