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

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/plugins/binary-block-plugin.hpp>
#include <olap/routing/degree-of-parallelism.hpp>
#include <ssb100_bloom/query.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
using namespace ssb100_bloom;
#pragma clang diagnostic pop

typedef BinaryBlockPlugin Tplugin;

template <typename Tplugin>
inline static auto getBuilder() {
  static RelBuilderFactory ctx{std::string{query} + typeid(Tplugin).name()};
  return ctx.getBuilder();
}

inline static auto &getCatalog() { return CatalogParser::getInstance(); }

const DegreeOfParallelism dop{2};
const DeviceType dev = DeviceType::GPU;

auto aff_parallel = []() -> std::unique_ptr<Affinitizer> {
  return std::make_unique<GPUAffinitizer>();
};

auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
  return std::make_unique<CpuNumaNodeAffinitizer>();
};
