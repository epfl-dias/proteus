#include "tests-tpch-benchmark-cli-common.hpp"

class TpchBenchmarkCliFlbpTest : public TpchBenchmarkCliBase
{
};

TEST_P(TpchBenchmarkCliFlbpTest, Completes)
{
    RunAndCheckCompletes({
        "FLBP",
        "/nvme29/boffa/data/tpch/compressed_flbp_tpch100",
        GetParam(),
        1024,
        true,
    });
}

INSTANTIATE_TEST_SUITE_P(Sf100GpuFlbp, TpchBenchmarkCliFlbpTest,
                         ::testing::ValuesIn(BuildTpchQueries()),
                         TpchCaseName);
