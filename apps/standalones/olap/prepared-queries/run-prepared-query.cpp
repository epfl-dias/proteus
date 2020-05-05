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
#include <glog/logging.h>

#include <cli-flags.hpp>
#include <common/olap-common.hpp>
#include <memory/memory-manager.hpp>
#include <plugins/binary-block-plugin.hpp>
#include <ssb100/query.hpp>
#include <ssb1000/query.hpp>
#include <storage/storage-manager.hpp>
#include <topology/topology.hpp>
#include <util/timing.hpp>

int main(int argc, char *argv[]) {
  auto olap = proteus::from_cli::olap("Prepared query runner", &argc, &argv);

  LOG(INFO) << "Finished initialization";

  LOG(INFO) << "Preparing queries...";

  /*{
    std::vector<PreparedStatement> statements;
    std::vector<PreparedStatement> cnts;
    //    for (const auto &memmv : {true, false}) {
    //      auto v = ssb100::Query{}.prepareAll(memmv);
    //      statements.insert(statements.end(),
    //      std::make_move_iterator(v.begin()),
    //                        std::make_move_iterator(v.end()));
    //    }
    std::vector<std::chrono::milliseconds> codegen_times;
    for (const auto &memmv : {true, false}) {
      time_block t{[&](const auto &x) { codegen_times.emplace_back(x); }};
      auto v = ssb100::Query{}.prepareAll(memmv);
      statements.insert(statements.end(), std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
    }
    for (auto bloomSize : {
             size_t{32}, size_t{64}, size_t{128}, size_t{256}, size_t{512},
             1_K,        2_K,        4_K,         8_K,         16_K,
             32_K,       64_K,       128_K,       256_K,       512_K,
             1_M,        2_M,        4_M,         8_M,         16_M,
             32_M,       64_M,       128_M,       256_M,       512_M,
             //                           1_G,
             //                           2_G
         }) {
      for (const auto &memmv : {true, false}) {
        //      for (const auto &conf : {  BLOOM_CPUFILTER_PROJECT,
        //                                 BLOOM_CPUFILTER_NOPROJECT,
        //                                 BLOOM_GPUFILTER_NOPROJECT,}) {
        time_block t{[&](const auto &x) { codegen_times.emplace_back(x); }};
        auto v = ssb100_bloom::Query{}.prepareAll(
            memmv, BLOOM_CPUFILTER_NOPROJECT, bloomSize);
        statements.insert(statements.end(), std::make_move_iterator(v.begin()),
                          std::make_move_iterator(v.end()));

        cnts.emplace_back(ssb100_bloom::Query{}.prepare41_b(memmv, bloomSize));
        //      }
      }
    }

    std::vector<std::vector<std::chrono::milliseconds>> times;
    for (size_t i = 0; i < 5; ++i) {
      times.emplace_back();
      for (auto &statement : statements) {
        //        LOG(INFO) <<
        time_block t{[&](const auto &x) { times.back().emplace_back(x); }};
        statement.execute();
      }
    }

    std::vector<std::vector<std::chrono::milliseconds>> probe_times(5);
    std::vector<std::string> vs;
    size_t j = 2;
    for (auto &cnt : cnts) {
      // Fill filter
      statements[j++].execute();

      // probe filter
      std::stringstream str;
      str << cnt.execute();
      vs.emplace_back(str.str());

      for (size_t i = 0; i < 5; ++i) {
        time_block t{[&](const auto &x) { probe_times[i].emplace_back(x); }};
        cnt.execute();
      }
    }

    for (const auto &t : vs) std::cout << t << '\t';
    std::cout << '\n' << '\n';

    for (const auto &t : codegen_times) std::cout << t.count() << '\t';
    std::cout << '\n' << '\n';

    for (const auto &ts : times) {
      for (const auto &t : ts) std::cout << t.count() << '\t';
      std::cout << '\n';
    }

    std::cout << "\n\n";
    for (const auto &ts : probe_times) {
      std::cout << "\t\t";
      for (const auto &t : ts) std::cout << t.count() << '\t';
      std::cout << '\n';
    }
  }*/

  return 0;
}
