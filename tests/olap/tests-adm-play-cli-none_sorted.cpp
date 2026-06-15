#include "tests-adm-play-cli-common.hpp"

// Correctness tests for the sorted SF100 RFOR dataset on /nvme29.
// Sorted by (lo_orderdate, lo_partkey, lo_custkey); query results must
// match the same golden outputs as the unsorted dataset.

class AdmPlayCliSf100InGpuRforSortedDecomOpT512Test
    : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuRforSortedDecomOpT1024Test
    : public AdmPlayCliCorrectnessBase
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuRforSortedDecomOpT512Test);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuRforSortedDecomOpT1024Test);

TEST_P(AdmPlayCliSf100InGpuRforSortedDecomOpT512Test, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildSortedCase(100, GetParam(), "RFOR", true, true, false,
                        std::nullopt, false, false,
                        std::optional<int>{512}));
}

TEST_P(AdmPlayCliSf100InGpuRforSortedDecomOpT1024Test, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildSortedCase(100, GetParam(), "RFOR", true, true, false,
                        std::nullopt, false, false,
                        std::optional<int>{1024}));
}

INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuRforSortedDecomOpT512,
    AdmPlayCliSf100InGpuRforSortedDecomOpT512Test,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_decomp)),
    QueryCaseName);

INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuRforSortedDecomOpT1024,
    AdmPlayCliSf100InGpuRforSortedDecomOpT1024Test,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_decomp)),
    QueryCaseName);
