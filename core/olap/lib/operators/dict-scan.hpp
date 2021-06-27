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
#ifndef DICTSCAN_HPP_
#define DICTSCAN_HPP_

#include <platform/network/infiniband/infiniband-manager.hpp>

#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class DictMatchIter;

class DictScan : public UnaryOperator {
 public:
  DictScan(Context *const context, RecordAttribute attr, std::string rex,
           RecordAttribute regAs)
      : UnaryOperator(nullptr),
        context(dynamic_cast<ParallelContext *const>(context)),
        attr(attr),
        regex(rex),
        regAs(regAs) {
    assert(this->context && "Only ParallelContext supported");
  }
  ~DictScan() override { LOG(INFO) << "Collapsing dictscan operator"; }
  [[nodiscard]] Operator *getChild() const final {
    throw runtime_error(string("Dictscan operator has no children"));
  }

  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override {
    ParallelContext *ctx = dynamic_cast<ParallelContext *>(context);
    if (!ctx) {
      string error_msg =
          "[DictScan: ] Operator only supports code generation "
          "using the ParallelContext";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    consume(ctx, childState);
  }

  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  bool isFiltering() const override { return true; }

  RecordType getRowType() const override {
    return {std::vector{new RecordAttribute{regAs}}};
  }

  [[nodiscard]] DegreeOfParallelism getDOPServers() const override {
    LOG(WARNING) << "Setting arbitrary number for #servers == "
                 << InfiniBandManager::server_count() << " !";
    return DegreeOfParallelism{InfiniBandManager::server_count()};
  }
  [[nodiscard]] bool isPacked() const override { return false; }
  [[nodiscard]] proteus::traits::HomReplication getHomReplication()
      const override {
    return proteus::traits::HomReplication::UNIQUE;
  }

  DeviceType getDeviceType() const override { return DeviceType::CPU; }
  DegreeOfParallelism getDOP() const override { return DegreeOfParallelism{1}; }

  virtual DictMatchIter begin() const;
  virtual DictMatchIter end() const;

  const std::string &getRegex() const { return regex; }

 private:
  const RecordAttribute &getAttr() const { return attr; }

  friend class DictMatchIter;

  ParallelContext *const context;
  const RecordAttribute attr;
  const RecordAttribute regAs;
  const std::string regex;
};

#endif /* DICTSCAN_HPP_ */
