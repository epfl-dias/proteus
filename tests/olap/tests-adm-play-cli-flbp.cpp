#include "tests-adm-play-cli-common.hpp"

class AdmPlayCliSf100InGpuFlbpTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuFlbpDecomOpTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuFlbpPreDecomTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuFlbpDecomRegTest : public AdmPlayCliCorrectnessBase
{
};

class AdmPlayCliSf100InGpuPaperConfigFlbpTest
    : public AdmPlayCliPaperConfigBase
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AdmPlayCliSf100InGpuFlbpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuFlbpDecomOpTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuFlbpPreDecomTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuFlbpDecomRegTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    AdmPlayCliSf100InGpuPaperConfigFlbpTest);

TEST_P(AdmPlayCliSf100InGpuFlbpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "FLBP", true, false));
}

TEST_P(AdmPlayCliSf100InGpuFlbpDecomOpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildCase(100, GetParam(), "FLBP", true, true));
}

TEST_P(AdmPlayCliSf100InGpuFlbpPreDecomTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(
        BuildCase(100, GetParam(), "FLBP", true, false, true));
}

TEST_P(AdmPlayCliSf100InGpuFlbpDecomRegTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(BuildFlbpDecomRegCase(100, GetParam()));
}

TEST_P(AdmPlayCliSf100InGpuPaperConfigFlbpTest, MatchesGolden)
{
    RunAndCheckAgainstGolden(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Sf100InGpuFlbp, AdmPlayCliSf100InGpuFlbpTest,
                         ::testing::ValuesIn(BuildQueriesIfEnabled(test_flbp)),
                         QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuFlbpDecomOp, AdmPlayCliSf100InGpuFlbpDecomOpTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_flbp_decomp)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuFlbpPreDecom, AdmPlayCliSf100InGpuFlbpPreDecomTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_flbp_pre_decomp)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuFlbpDecomReg, AdmPlayCliSf100InGpuFlbpDecomRegTest,
    ::testing::ValuesIn(BuildQueriesIfEnabled(test_flbp_decomp_reg)),
    QueryCaseName);
INSTANTIATE_TEST_SUITE_P(
    Sf100InGpuPaperConfigsFlbp, AdmPlayCliSf100InGpuPaperConfigFlbpTest,
    ::testing::ValuesIn(BuildPaperConfigCasesFlbp()), AdmPlayCaseName);
