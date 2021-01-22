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

#include <olap/util/demangle.hpp>
#include <platform/util/glog.hpp>
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

#define QALIAS(name, prepFunction)                                \
  struct name {                                                   \
    PreparedStatement operator()(proteus::QueryShaper &s) const { \
      return prepFunction(s);                                     \
    }                                                             \
  };
#define SALIAS(name, shaperType) \
  struct name {                  \
    using shaper_t = shaperType; \
  };

namespace SSB {
QALIAS(Q1_1, ssb::Query::prepare11);
QALIAS(Q1_2, ssb::Query::prepare12);
QALIAS(Q1_3, ssb::Query::prepare13);
QALIAS(Q2_1, ssb::Query::prepare21);
QALIAS(Q2_2, ssb::Query::prepare22);
QALIAS(Q2_3, ssb::Query::prepare23);
QALIAS(Q3_1, ssb::Query::prepare31);
QALIAS(Q3_2, ssb::Query::prepare32);
QALIAS(Q3_3, ssb::Query::prepare33);
QALIAS(Q3_4, ssb::Query::prepare34);
QALIAS(Q4_1, ssb::Query::prepare41);
QALIAS(Q4_2, ssb::Query::prepare42);
QALIAS(Q4_3, ssb::Query::prepare43);
}  // namespace SSB

SALIAS(CPUonly, proteus::CPUOnlySingleSever);
SALIAS(GPUonly, proteus::GPUOnlySingleSever);

template <typename Qprep, typename Shaper>
void query_perf(benchmark::State &state) {
  size_t SF = 100;
  auto stats = ssb::Query::getStats(SF);

  auto shaper = make_shaper<typename Shaper::shaper_t>(SF, stats);

  actualRun(Qprep{}(*shaper), state);
}

#define BENCHMARK_NAMED_MACRO(fun, T1name, T1list, T2name, T2list)           \
  static void *BENCHMARK_PRIVATE_NAME(regBenchmarks) [[maybe_unused]] = [] { \
    auto f = [](auto T1, auto T2) {                                          \
      benchmark::internal::RegisterBenchmarkInternal(                        \
          new benchmark::internal::FunctionBenchmark(                        \
              ("perftest-" +                                                 \
               std::filesystem::path{__FILE__}                               \
                   .filename()                                               \
                   .replace_extension("")                                    \
                   .string() +                                               \
               "/function:" + #fun +                                         \
                                                                             \
               "/" #T1name ":" + demangle(typeid(T1).name()) +               \
                                                                             \
               "/" #T2name ":" + demangle(typeid(T2).name()))                \
                  .c_str(),                                                  \
              &fun<decltype(T1), decltype(T2)>))                             \
          ->Unit(benchmark::kMillisecond)                                    \
          ->Iterations(5 + 1)                                                \
          ->UseRealTime();                                                   \
    };                                                                       \
                                                                             \
    std::apply(                                                              \
        [&](auto... y) {                                                     \
          auto fy = [&](auto y2) {                                           \
            std::apply([&](auto... x) { (..., f(x, y2)); }, T1list{});       \
          };                                                                 \
                                                                             \
          (..., fy(y));                                                      \
        },                                                                   \
        T2list{});                                                           \
    return nullptr;                                                          \
  }();

using QuerySet = std::tuple<
    // Group 1
    SSB::Q1_1, SSB::Q1_2, SSB::Q1_3,
    // Group 2
    SSB::Q2_1, SSB::Q2_2, SSB::Q2_3,
    // Group 3
    SSB::Q3_1, SSB::Q3_2, SSB::Q3_3, SSB::Q3_4,
    // Group 4
    SSB::Q4_1, SSB::Q4_2, SSB::Q4_3>;

using ShaperSet = std::tuple<CPUonly, GPUonly>;

BENCHMARK_NAMED_MACRO(query_perf, query, QuerySet, shaper, ShaperSet);
