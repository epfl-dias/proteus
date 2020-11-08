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

#include <chrono>
#include <cli-flags.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/operators/relbuilder.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/storage/storage-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <query-shaping/input-prefix-query-shaper.hpp>
#include <query-shaping/scale-out-query-shaper.hpp>
#include <ssb100/query.hpp>

int main(int argc, char* argv[]) {
  auto ctx = proteus::from_cli::olap("Multi IB", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  size_t SF = 1000;
  std::map<std::string, std::function<double(proteus::InputPrefixQueryShaper&)>>
      stats{{"sf", [SF](auto&) { return SF; }},
            {"date", [](auto&) { return 2556; }},
            {"customer", [](auto& s) { return s.sf() * 30'000; }},
            {"supplier", [](auto& s) { return s.sf() * 2'000; }},
            {"part",
             [](auto& s) {
               return 200'000 * std::ceil(1 + std::log2((double)s.sf()));
             }},
            {"lineorder", [](auto& s) { return s.sf() * 6'000'000; }}};

  std::vector<std::unique_ptr<proteus::QueryShaper>> shapers;
  //  shapers.emplace_back(std::make_unique<proteus::InputPrefixQueryShaper>(
  //      "inputs/ssbm" + std::to_string(SF) + "/", stats));
  shapers.emplace_back(std::make_unique<proteus::ScaleOutQueryShaper>(
      "inputs/ssbm" + std::to_string(SF) + "/", stats));

  std::vector<std::vector<std::vector<std::chrono::milliseconds>>> times_all;

  assert(FLAGS_port <= std::numeric_limits<uint16_t>::max());
  InfiniBandManager::init(FLAGS_url, static_cast<uint16_t>(FLAGS_port),
                          FLAGS_primary, FLAGS_ipv4);

  for (auto& shaper_ptr : shapers) {
    StorageManager::getInstance().unloadAll();
    auto& shaper = *shaper_ptr;
    std::vector<PreparedStatement> rel{
        ssb100::Query::prepare11(shaper),
        ssb100::Query::prepare12(shaper),
        ssb100::Query::prepare13(shaper),
        //
        ssb100::Query::prepare21(shaper),
        ssb100::Query::prepare22(shaper),
        ssb100::Query::prepare23(shaper),
        //
        ssb100::Query::prepare31(shaper),
        ssb100::Query::prepare32(shaper),
        ssb100::Query::prepare33(shaper),
        ssb100::Query::prepare34(shaper),
        //
        ssb100::Query::prepare41(shaper),
        ssb100::Query::prepare42(shaper),
        ssb100::Query::prepare43(shaper),
    };

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);

    for (auto& st : rel) {
      for (size_t i = 0; i < 1; ++i) {
        LOG(INFO) << st.execute();
      }
    }

    times_all.emplace_back();
    auto& times = times_all.back();

    for (auto& st : rel) {
      times.emplace_back();
      for (size_t i = 0; i < 7; ++i) {
        time_block t{[&](auto tms) { times.back().emplace_back(tms); }};
        st.execute();
      }
    }
  }

  if (!FLAGS_primary) InfiniBandManager::disconnectAll();

  StorageManager::getInstance().unloadAll();
  InfiniBandManager::deinit();

  for (auto& times : times_all) {
    for (auto& st : times) {
      for (const auto ms : st) {
        std::cout << ms.count() << '\t';
      }
      std::cout << '\n';
    }
    std::cout << "\n\n\n";
  }

  return 0;
}
