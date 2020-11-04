/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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
#ifndef SCAN_HPP_
#define SCAN_HPP_

#include "operators.hpp"

class Scan : public experimental::UnaryOperator {
 public:
  explicit Scan(Plugin &pg) : UnaryOperator(nullptr), pg(pg) {}
  [[nodiscard]] Operator *getChild() const final {
    throw runtime_error(string("Scan operator has no children"));
  }

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override;

  [[nodiscard]] DeviceType getDeviceType() const override {
    return DeviceType::CPU;
  }
  [[nodiscard]] DegreeOfParallelism getDOP() const override {
    return DegreeOfParallelism{1};
  }
  [[nodiscard]] DegreeOfParallelism getDOPServers() const override {
    LOG(WARNING) << "Setting arbitrary number for #servers == 2 !";
    return DegreeOfParallelism{2};
  }

 private:
  Plugin &pg;
};

#endif /* SCAN_HPP_ */
