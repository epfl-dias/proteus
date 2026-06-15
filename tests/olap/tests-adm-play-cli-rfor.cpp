#include "tests-adm-play-cli-common.hpp"

class AdmPlayCliSf100OutGpuRforTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100OutGpuRforDecomOpTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuRforTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuRforDecomOpTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuRforPreDecomTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuRforDecomRegTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuOneKernelOffRforTest
    : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuOneKernelOffRforDecomOpTest
    : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf1000OutGpuRforTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf1000OutGpuRforDecomOpTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuPaperConfigRforTest
    : public AdmPlayCliPaperConfigBase
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf100OutGpuRforTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100OutGpuRforDecomOpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf100InGpuRforTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuRforDecomOpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuRforPreDecomTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuRforDecomRegTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuOneKernelOffRforTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuOneKernelOffRforDecomOpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf1000OutGpuRforTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf1000OutGpuRforDecomOpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuPaperConfigRforTest);

TEST_P(AdmPlayCliSf100OutGpuRforTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "RFOR", false, false));
}

TEST_P(AdmPlayCliSf100OutGpuRforDecomOpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "RFOR", false, true));
}

TEST_P(AdmPlayCliSf100InGpuRforTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "RFOR", true, false));
}

TEST_P(AdmPlayCliSf100InGpuRforDecomOpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "RFOR", true, true));
}

TEST_P(AdmPlayCliSf100InGpuRforPreDecomTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildCase(100, GetParam(), "RFOR", true, false, true));
}

TEST_P(AdmPlayCliSf100InGpuRforDecomRegTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildDecomRegCase(100, GetParam()));
}

TEST_P(AdmPlayCliSf100InGpuOneKernelOffRforTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildCase(100, GetParam(), "RFOR", true, false, false,
                  std::optional<bool>{false}));
}

TEST_P(AdmPlayCliSf100InGpuOneKernelOffRforDecomOpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildCase(100, GetParam(), "RFOR", true, true, false,
                  std::optional<bool>{false}));
}

TEST_P(AdmPlayCliSf1000OutGpuRforTest, RunsAndProducesResult)
{
    RunAndCheckProducesResult(BuildCase(1000, GetParam(), "RFOR", false, false));
}

TEST_P(AdmPlayCliSf1000OutGpuRforDecomOpTest, RunsAndProducesResult)
{
    RunAndCheckProducesResult(BuildCase(1000, GetParam(), "RFOR", false, true));
}

TEST_P(AdmPlayCliSf100InGpuPaperConfigRforTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(GetParam());
}

// INSTANTIATE_TEST_SUITE_P(Sf100OutGpuRfor, AdmPlayCliSf100OutGpuRforTest,
//                          ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor)),
//                          QueryCaseName);
// INSTANTIATE_TEST_SUITE_P(
//     Sf100OutGpuRforDecomOp, AdmPlayCliSf100OutGpuRforDecomOpTest,
//     ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_decomp)),
//     QueryCaseName);
INSTANTIATE_TEST_SUITE_P(Sf100InGpuRfor, AdmPlayCliSf100InGpuRforTest,
                         ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(Sf100InGpuRforDecomOp, AdmPlayCliSf100InGpuRforDecomOpTest,
                         ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_decomp)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuRforPreDecom, AdmPlayCliSf100InGpuRforPreDecomTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_pre_decomp)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuRforDecomReg, AdmPlayCliSf100InGpuRforDecomRegTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_rfor_decomp_reg)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuOneKernelOffRfor, AdmPlayCliSf100InGpuOneKernelOffRforTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_one_kernel && test_rfor)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuOneKernelOffRforDecomOp,
    AdmPlayCliSf100InGpuOneKernelOffRforDecomOpTest,
    ::testing::ValuesIn(
        BuildQueriesIfEnabled(test_one_kernel && test_rfor_decomp)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(Sf1000OutGpuRfor, AdmPlayCliSf1000OutGpuRforTest,
                         ::testing::ValuesIn(BuildSf1000QueriesIfEnabled(
                             test_rfor)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf1000OutGpuRforDecomOp, AdmPlayCliSf1000OutGpuRforDecomOpTest,
    ::testing::ValuesIn(BuildSf1000QueriesIfEnabled(test_rfor_decomp)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuPaperConfigsRfor, AdmPlayCliSf100InGpuPaperConfigRforTest,
    ::testing::ValuesIn(BuildPaperConfigCasesRfor()), AdmPlayCaseName);
