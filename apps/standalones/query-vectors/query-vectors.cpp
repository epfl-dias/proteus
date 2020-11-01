/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <cli-flags.hpp>
#include <memory>
#include <olap/operators/relbuilder-factory.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("Template", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  std::vector<int32_t> v1{4, 5, 7};
  std::vector<int64_t> v2{6, 7, 8};

  auto builder =
      RelBuilderFactory{__FUNCTION__}
          .getBuilder()
          .scan({{new RecordAttribute{__FUNCTION__, "a", new IntType()},
                  std::make_shared<proteus_any_vector>(std::move(v1))},
                 {new RecordAttribute{__FUNCTION__, "b", new Int64Type()},
                  std::make_shared<proteus_any_vector>(std::move(v2))}})
          .unpack()
          .filter(
              [](const auto &arg) -> expression_t { return lt(arg["a"], 5); })
          .project([](const auto &arg) -> std::vector<expression_t> {
            return {(arg["b"] + int64_t{1}).as("test", "test_b")};
          })
          .print(pg{"pm-csv"});

  LOG(INFO) << builder.prepare().execute();

  return 0;
}
