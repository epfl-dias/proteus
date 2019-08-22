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

#ifndef SORT_HPP_
#define SORT_HPP_

#include "operators/operators.hpp"
#include "util/parallel-context.hpp"
#include "util/sort/sort-direction.hpp"

class Sort : public UnaryOperator {
 public:
  Sort(Operator *const child, ParallelContext *const context,
       const vector<expression_t> &orderByFields,
       const vector<direction> &dirs);

  virtual ~Sort() { LOG(INFO) << "Collapsing Sort operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : orderByFields) {
      attrs.emplace_back(new RecordAttribute{attr.getRegisteredAs()});
    }
    return std::vector<RecordAttribute *>{new RecordAttribute{
        attrs[0]->getRelationName(), "__sorted", new RecordType{attrs}}};
  }

 protected:
  // virtual void open (Pipeline * pip);
  // virtual void close(Pipeline * pip);

  virtual void flush_sorted();

  virtual void call_sort(llvm::Value *mem, llvm::Value *N);

  std::vector<expression_t> orderByFields;
  const vector<direction> dirs;

  expressions::RecordConstruction outputExpr;
  std::string relName;

  // size_t                          width       ;

  StateVar cntVar_id;
  StateVar memVar_id;

  llvm::Type *mem_type;

  ParallelContext *const context;
};

#endif /* SORT_HPP_ */
