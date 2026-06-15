#include "tests-tpch-benchmark-cli-common.hpp"

class TpchBenchmarkCliNoneTest : public TpchBenchmarkCliBase
{
};

TEST_P(TpchBenchmarkCliNoneTest, Completes)
{
    RunAndCheckCompletes({
        "NONE",
        "/nvme29/boffa/data/tpch/uncompressed_tpch100",
        GetParam(),
        std::nullopt,
        false,
    });
}

INSTANTIATE_TEST_SUITE_P(Sf100GpuNone, TpchBenchmarkCliNoneTest,
                         ::testing::ValuesIn(BuildTpchQueries()),
                         TpchCaseName);
