/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef HASH_JOIN_CHAINED_MORSEL_HPP_
#define HASH_JOIN_CHAINED_MORSEL_HPP_

#include <mutex>

#include "hash-join-chained.hpp"

class HashJoinChainedMorsel : public HashJoinChained {
 public:
  using HashJoinChained::HashJoinChained;

  void open_probe(Pipeline *pip) override;
  void open_build(Pipeline *pip) override;
  void close_probe(Pipeline *pip) override;
  void close_build(Pipeline *pip) override;

 protected:
  llvm::Value *nextIndex(ParallelContext *context) override;
  llvm::Value *replaceHead(ParallelContext *context, llvm::Value *h_ptr,
                           llvm::Value *index) override;

  std::mutex init_lock;
  size_t workerCnt = 0;
};

#endif /* HASH_JOIN_CHAINED_MORSEL_HPP_ */
