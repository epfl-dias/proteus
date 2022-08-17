/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <benchmark/benchmark.h>

#include <olap-perf-util/benchmark-aliases.hpp>
#include <olap-perf-util/ssb-aliases.hpp>
#include <olap/util/demangle.hpp>
#include <platform/util/timing.hpp>
#include <query-shaping/experimental-shapers.hpp>
#include <ssb/query.hpp>

void actualRun(PreparedStatement st, benchmark::State &state) {
  std::chrono::milliseconds exeTime{0};
  size_t its = 0;

  for ([[maybe_unused]] auto _ : state) {
    std::stringstream ss;
    {
      time_block t{[&](const auto &ms) {
        if (its++) exeTime += ms;
      }};
      ss << st.execute();  // PreparedStatement::SilentExecution
    }
    std::this_thread::sleep_for(std::chrono::seconds{1});
  }

  state.counters["Exec (ms)"] = (exeTime / (its - 1)).count();
}

PreparedStatement small_scan(proteus::QueryShaper &morph) {
  morph.setQueryName("small_scan");

  return morph
      .parallel(morph.scan("lineorder", {"lo_orderdate"}), {},
                [&](auto probe, auto) {
                  return probe.unpack()
                      .filter([&](const auto &arg) -> expression_t {
                        return expressions::hint(
                            lt(arg["lo_orderdate"], 1),
                            expressions::Selectivity(0.000001));
                      })
                      .reduce(
                          [&](const auto &arg) -> std::vector<expression_t> {
                            return {arg["lo_orderdate"]};
                          },
                          {SUM});
                })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["lo_orderdate"]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}

PreparedStatement sumQuery(proteus::QueryShaper &morph) {
  morph.setQueryName("sum");
  auto rel =
      morph.scan("lineorder", {"lo_custkey", "lo_discount", "lo_extendedprice",
                               "lo_orderdate"
                               /*, "lo_partkey", "lo_quantity",
                               "lo_revenue", "lo_suppkey"*/});

  return morph
      .parallel(rel, {},
                [](RelBuilder probe, const std::vector<RelBuilder> &build) {
                  return probe.unpack().reduce(
                      [&](const auto &arg) -> std::vector<expression_t> {
                        return {arg["lo_custkey"],       arg["lo_discount"],
                                arg["lo_extendedprice"], arg["lo_orderdate"],
                                /*arg["lo_partkey"],       arg["lo_quantity"],
                                arg["lo_revenue"],       arg["lo_suppkey"]*/};
                      },
                      // {SUM, SUM, SUM, SUM, SUM, SUM, SUM, SUM});
                      {SUM, SUM, SUM, SUM});
                })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["lo_custkey"],       arg["lo_discount"],
                    arg["lo_extendedprice"], arg["lo_orderdate"],
                    /* arg["lo_partkey"],       arg["lo_quantity"],
                     arg["lo_revenue"],       arg["lo_suppkey"]*/};
          },
          //          {SUM, SUM, SUM, SUM, SUM, SUM, SUM, SUM})
          {SUM, SUM, SUM, SUM})
      .print(pg{"pm-csv"})
      .prepare();
}

SALIAS(CPUonly, proteus::CPUOnlySingleSever);
SALIAS(GPUonly, proteus::GPUOnlySingleSever);

template <typename Qprep, typename Shaper>
void query_perf(benchmark::State &state) {
  size_t SF = Qprep::SF;
  auto stats = ssb::Query::getStats(SF);

  auto shaper = make_shaper<typename Shaper::shaper_t>(SF, stats);

  actualRun(Qprep{}(*shaper), state);
}

namespace SSB = SSB100;

QALIASSF(Qsum, small_scan, SSB::SF);
QALIASSF(QsumRuSH, sumQuery, SSB::SF);

using QuerySet = std::tuple<
    // Group 1
    SSB::Q1_1, SSB::Q1_2, SSB::Q1_3,
    // Group 2
    SSB::Q2_1, SSB::Q2_2, SSB::Q2_3,
    // Group 3
    SSB::Q3_1, SSB::Q3_2, SSB::Q3_3, SSB::Q3_4,
    // Group 4
    SSB::Q4_1, SSB::Q4_2, SSB::Q4_3,
    // Sum Query
    Qsum>;

using ShaperSet = std::tuple<CPUonly, GPUonly>;

BENCHMARK_NAMED_MACRO(query_perf, query, QuerySet, shaper, ShaperSet);
