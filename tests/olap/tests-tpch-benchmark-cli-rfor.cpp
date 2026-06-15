#include "tests-tpch-benchmark-cli-common.hpp"

class TpchBenchmarkCliRforTest : public TpchBenchmarkCliBase
{
};

TEST_P(TpchBenchmarkCliRforTest, Completes)
{
    RunAndCheckCompletes({
        "RFOR",
        "/nvme29/boffa/data/tpch/compressed_rfor_tpch100_t512",
        GetParam(),
        512,
        true,
    });
}

INSTANTIATE_TEST_SUITE_P(Sf100GpuRfor, TpchBenchmarkCliRforTest,
                         ::testing::ValuesIn(BuildTpchQueries()),
                         TpchCaseName);
