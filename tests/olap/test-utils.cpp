/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "test-utils.hpp"

#include "memory/memory-manager.hpp"
#include "olap/common/olap-common.hpp"
#include "olap/plan/prepared-statement.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "storage/storage-manager.hpp"

void TestEnvironment::SetUp() {
  assert(!has_already_been_setup);

  setbuf(stdout, nullptr);

  google::InstallFailureSignalHandler();

  // FIXME: reenable tracing as soon as we find the issue with libunwind
  set_trace_allocations(false, true);

  olap = std::make_unique<proteus::olap>();

  has_already_been_setup = true;
}

void TestEnvironment::TearDown() {
  if (!is_noop) {
    olap.reset();
    has_already_been_setup = false;
  }
}

bool verifyTestResult(const char *testsPath, const char *testLabel,
                      bool unordered) {
  /* Compare with template answer */
  /* correct */
  struct stat statbuf;
  string correctResult = string(testsPath) + testLabel;
  if (stat(correctResult.c_str(), &statbuf)) {
    fprintf(stderr, "FAILURE to stat test verification! (%s) (path: %s)\n",
            std::strerror(errno), correctResult.c_str());
    return false;
  }
  size_t fsize1 = statbuf.st_size;
  int fd1 = open(correctResult.c_str(), O_RDONLY);
  if (fd1 == -1) {
    throw runtime_error(string(__func__) + string(".open (verification): ") +
                        correctResult);
  }
  char *correctBuf = (char *)mmap(nullptr, fsize1, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE, fd1, 0);

  /* current */
  // if (unordered){
  //  std::system((std::string("sort ") + testLabel + " > " +
  //  testLabel).c_str());
  // }
  int fd2 = shm_open(testLabel, O_RDONLY, S_IRWXU);
  if (fd2 == -1) {
    throw runtime_error(string(__func__) + string(".open (output): ") +
                        testLabel);
  }
  if (fstat(fd2, &statbuf)) {
    fprintf(stderr, "FAILURE to stat test results! (%s)\n",
            std::strerror(errno));
    return false;
  }
  size_t fsize2 = statbuf.st_size;
  char *currResultBuf = (char *)mmap(nullptr, fsize2, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE, fd2, 0);
  bool areEqual = (fsize1 == fsize2);
  if (areEqual) {
    if (unordered) {
      std::vector<std::string> lines;
      std::stringstream ss(currResultBuf);
      std::string str;
      while (std::getline(ss, str)) lines.emplace_back(str);
      std::sort(lines.begin(), lines.end());
      ss.clear();
      for (const auto &s : lines) ss << s << '\n';
      areEqual =
          (fsize1 == 0) || (memcmp(correctBuf, ss.str().c_str(), fsize1) == 0);
    } else {
      areEqual =
          (fsize1 == 0) || (memcmp(correctBuf, currResultBuf, fsize1) == 0);
    }
    // Document document; // Default template parameter uses UTF8 and
    // MemoryPoolAllocator. auto & parsed = document.Parse(currResultBuf); if
    // (parsed.HasParseError()) {
    //     ParseResult ok = (ParseResult) parsed;
    //     fprintf(stderr, "JSON parse error: %s (%lu)",
    //     RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset()); const
    //     char *err = "[PlanExecutor: ] Error parsing physical plan (JSON
    //     parsing error)"; LOG(ERROR)<< err; throw runtime_error(err);
    // }

    // Document document2; // Default template parameter uses UTF8 and
    // MemoryPoolAllocator. auto & parsed2 = document2.Parse(correctBuf); if
    // (parsed2.HasParseError()) {
    //     ParseResult ok = (ParseResult) parsed2;
    //     fprintf(stderr, "JSON parse error: %s (%lu)",
    //     RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset()); const
    //     char *err = "[PlanExecutor: ] Error parsing physical plan (JSON
    //     parsing error)"; LOG(ERROR)<< err; throw runtime_error(err);
    // }

    // // if (parsed2.IsArray() && parsed.IsArray() && unordered){
    // //     std::sort(parsed2.Begin(), parsed2.End());
    // //     std::sort(parsed.Begin(), parsed.End());
    // // }

    // areEqual = (parsed2 == parsed);
  }

  if (!areEqual) {
    fprintf(stderr,
            "##########################################################"
            "############\n");
    fprintf(stderr, "FAILURE:\n");
    if (fsize1 > 0)
      fprintf(stderr, "* Expected (size: %zu):\n%s\n", fsize1, correctBuf);
    else
      fprintf(stderr, "* Expected empty file\n");
    if (fsize2 > 0)
      fprintf(stderr, "* Obtained (size: %zu):\n%s\n", fsize2, currResultBuf);
    else
      fprintf(stderr, "* Obtained empty file\n");
    fprintf(stderr,
            "##########################################################"
            "############\n");
  }

  close(fd1);
  munmap(correctBuf, fsize1);
  // close(fd2);
  munmap(currResultBuf, fsize2);
  // if (remove(testLabel) != 0) {
  //  throw runtime_error(string("Error deleting file"));
  // }

  return areEqual;
}

void runAndVerify(const char *testLabel, const char *planPath,
                  const char *testPath, const char *catalogJSON,
                  bool unordered) {
  auto qr = PreparedStatement::from(planPath, testLabel, catalogJSON).execute();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel, unordered));  // FIXME:!!!!
}

bool TestEnvironment::has_already_been_setup = false;
