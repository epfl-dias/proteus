/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include "olap/operators/relbuilder-factory.hpp"

#include "olap/util/parallel-context.hpp"

class RelBuilderFactory::impl {
 public:
  ParallelContext *ctx;

  explicit impl(std::string name)
      : ctx(new ParallelContext(std::move(name), false)) {}
};

RelBuilderFactory::RelBuilderFactory(std::string name)
    : pimpl(std::make_unique<RelBuilderFactory::impl>(std::move(name))) {}

RelBuilderFactory::~RelBuilderFactory() = default;

RelBuilder RelBuilderFactory::getBuilder() const {
  return RelBuilder{pimpl->ctx};
}
