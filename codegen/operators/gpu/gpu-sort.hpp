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

#ifndef GPU_SORT_HPP_
#define GPU_SORT_HPP_

#include "codegen/util/parallel-context.hpp"
#include "operators/operators.hpp"
#include "operators/sort.hpp"

class GpuSort : public UnaryOperator {
 public:
  GpuSort(Operator *const child, ParallelContext *const context,
          const std::vector<expression_t> &orderByFields,
          const std::vector<direction> &dirs,
          gran_t granularity = gran_t::GRID);

  virtual ~GpuSort() { LOG(INFO) << "Collapsing GpuSort operator"; }

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
    return attrs;
  }

 protected:
  virtual void flush_sorted();

  virtual void call_sort(llvm::Value *mem, llvm::Value *N);

  std::vector<expression_t> orderByFields;
  const std::vector<direction> dirs;

  expressions::RecordConstruction outputExpr;
  std::string relName;

  gran_t granularity;

  size_t cntVar_id;
  size_t memVar_id;

  llvm::Type *mem_type;

  ParallelContext *const context;

  std::string suffix;
};

#endif /* GPU_SORT_HPP_ */
