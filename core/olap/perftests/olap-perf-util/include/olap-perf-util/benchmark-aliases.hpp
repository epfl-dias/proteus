/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#ifndef PROTEUS_BENCHMARK_ALIASES_HPP
#define PROTEUS_BENCHMARK_ALIASES_HPP

#include <filesystem>
#include <olap/util/demangle.hpp>

#define QALIASSF(name, prepFunction, scale)                       \
  struct name {                                                   \
    static constexpr size_t SF = scale;                           \
    PreparedStatement operator()(proteus::QueryShaper &s) const { \
      return prepFunction(s);                                     \
    }                                                             \
  }
#define SALIAS(name, shaperType) \
  struct name {                  \
    using shaper_t = shaperType; \
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

#endif  // PROTEUS_BENCHMARK_ALIASES_HPP
