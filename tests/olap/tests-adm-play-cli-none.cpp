#include "tests-adm-play-cli-common.hpp"

class AdmPlayCliSf100OutGpuNoneTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuNoneTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuOneKernelOffNoneTest
    : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf1000OutGpuNoneTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuPaperConfigNoneTest
    : public AdmPlayCliPaperConfigBase
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf100OutGpuNoneTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf100InGpuNoneTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuOneKernelOffNoneTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf1000OutGpuNoneTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuPaperConfigNoneTest);

TEST_P(AdmPlayCliSf100OutGpuNoneTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "NONE", false, false));
}

TEST_P(AdmPlayCliSf100InGpuNoneTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "NONE", true, false));
}

TEST_P(AdmPlayCliSf100InGpuOneKernelOffNoneTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildCase(100, GetParam(), "NONE", true, false, false,
                  std::optional<bool>{false}));
}

TEST_P(AdmPlayCliSf1000OutGpuNoneTest, RunsAndProducesResult)
{
    RunAndCheckProducesResult(BuildCase(1000, GetParam(), "NONE", false, false));
}

TEST_P(AdmPlayCliSf100InGpuPaperConfigNoneTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(GetParam());
}

// INSTANTIATE_TEST_SUITE_P(Sf100OutGpuNone, AdmPlayCliSf100OutGpuNoneTest,
//                          ::testing::ValuesIn(BuildQueriesIfEnabled(test_none)),
//                          QueryCaseName);
INSTANTIATE_TEST_SUITE_P(Sf100InGpuNone, AdmPlayCliSf100InGpuNoneTest,
                         ::testing::ValuesIn(BuildQueriesIfEnabled(test_none)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuOneKernelOffNone, AdmPlayCliSf100InGpuOneKernelOffNoneTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_one_kernel && test_none)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(Sf1000OutGpuNone, AdmPlayCliSf1000OutGpuNoneTest,
                         ::testing::ValuesIn(BuildSf1000QueriesIfEnabled(
                             test_none)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuPaperConfigsNone, AdmPlayCliSf100InGpuPaperConfigNoneTest,
    ::testing::ValuesIn(BuildPaperConfigCasesNone()), AdmPlayCaseName);
