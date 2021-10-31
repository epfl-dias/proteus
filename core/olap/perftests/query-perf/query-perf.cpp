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

#include <platform/util/glog.hpp>
#include <platform/util/timing.hpp>
#include <ssb/query.hpp>

class SSBFixture : public benchmark::Fixture {
 public:
  size_t SF = 100;
  void actualRun(std::vector<PreparedStatement> &rel, benchmark::State &state);
};

void SSBFixture::actualRun(std::vector<PreparedStatement> &rel,
                           benchmark::State &state) {
  std::chrono::milliseconds exeTime{0};
  size_t its = 0;

  for ([[maybe_unused]] auto _ : state) {
    for (auto &st : rel) {
      std::stringstream ss;
      {
        time_block t{[&](const auto &ms) {
          if (its++) exeTime += ms;
        }};
        ss << st.execute();  // PreparedStatement::SilentExecution
      }
      std::this_thread::sleep_for(std::chrono::seconds{1});
    }
  }

  state.counters["Exec (ms)"] = (exeTime / (its - 1)).count();
}

namespace proteus {
class CPUOnlySingleSever : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] DeviceType getDevice() override { return DeviceType::CPU; }

  std::unique_ptr<Affinitizer> getAffinitizer() override {
    return std::make_unique<CpuNumaNodeAffinitizer>();
  }
};
class GPUOnlySingleSever : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] DeviceType getDevice() override { return DeviceType::GPU; }

  std::unique_ptr<Affinitizer> getAffinitizer() override {
    return std::make_unique<GPUAffinitizer>();
  }
};
}  // namespace proteus

template <typename T>
std::unique_ptr<T> make_shaper(
    size_t SF, decltype(ssb::Query::getStats(std::declval<size_t>())) stat) {
  return std::make_unique<T>("inputs/ssbm" + std::to_string(SF) + "/",
                             std::move(stat));
}

BENCHMARK_DEFINE_F(SSBFixture, PreparedQuery)
(benchmark::State &state) {
  auto stats = ssb::Query::getStats(SF);

  std::vector<std::unique_ptr<proteus::QueryShaper>> shapers;

  switch (state.range(1)) {
    case 0:
      shapers.emplace_back(make_shaper<proteus::CPUOnlySingleSever>(SF, stats));
      break;
    case 1:
      shapers.emplace_back(make_shaper<proteus::GPUOnlySingleSever>(SF, stats));
      break;
    default:
      assert(false);
  }

  using namespace std::chrono_literals;

  std::vector<std::function<PreparedStatement(proteus::QueryShaper &)>> prep;
  prep.emplace_back([&] {
    switch (state.range(0)) {
      case 0: {
        return ssb::Query::prepare11;
      }
      case 1: {
        return ssb::Query::prepare12;
      }
      case 2: {
        return ssb::Query::prepare13;
      }
      case 3: {
        return ssb::Query::prepare21;
      }
      case 4: {
        return ssb::Query::prepare22;
      }
      case 5: {
        return ssb::Query::prepare23;
      }
      case 6: {
        return ssb::Query::prepare31;
      }
      case 7: {
        return ssb::Query::prepare32;
      }
      case 8: {
        return ssb::Query::prepare33;
      }
      case 9: {
        return ssb::Query::prepare34;
      }
      case 10: {
        return ssb::Query::prepare41;
      }
      case 11: {
        return ssb::Query::prepare42;
      }
      case 12: {
        return ssb::Query::prepare43;
      }
    }
    assert(false);
  }());

  std::vector<PreparedStatement> rel;

  for (auto &shaper_ptr : shapers) {
    auto &shaper = *shaper_ptr;

    {
      rel.reserve(rel.size() + prep.size());
      for (auto &p : prep) rel.emplace_back(p(shaper));
    }
  }

  actualRun(rel, state);
}

#define BENCHMARK_SCALEOUT_MACRO(testcase)   \
  BENCHMARK_REGISTER_F(SSBFixture, testcase) \
      ->Unit(benchmark::kMillisecond)        \
      ->Iterations(5 + 1)                    \
      ->UseRealTime()

BENCHMARK_SCALEOUT_MACRO(PreparedQuery)
    ->ArgsProduct({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {0, 1}});
