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

#include "codegen/plan/prepared-statement.hpp"
#include "codegen/util/parallel-context.hpp"
#include "memory/memory-manager.hpp"
#include "plan/plan-parser.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/schema.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/functions.hpp"
#include "util/jit/pipeline.hpp"

void TestEnvironment::SetUp() {
  if (has_already_been_setup) {
    is_noop = true;
    return;
  }

  setbuf(stdout, NULL);

  google::InstallFailureSignalHandler();

  set_trace_allocations(true, true);

  // int devCount;

  // gpu_run(cuInit(0));
  // gpu_run(cuDeviceGetCount(&devCount));

  // device  = new CUdevice [devCount];
  // context = new CUcontext[devCount];

  // for (int i = 0 ; i < devCount ; ++i){
  //     gpu_run(cuDeviceGet(device  + i, i));
  //     gpu_run(cuCtxCreate(context + i, 0, device[i]));
  // }

  // gpu_run(cudaSetDeviceFlags(cudaDeviceScheduleYield));

  // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

  std::vector<std::thread> thrds;
  for (int i = 0; i < 32; ++i) thrds.emplace_back([] {});
  for (auto &t : thrds) t.join();

  // srand(time(0));

  PipelineGen::init();
  MemoryManager::init();

  gpu_run(cudaSetDevice(0));

  has_already_been_setup = true;
}

void TestEnvironment::TearDown() {
  if (!is_noop) MemoryManager::destroy();
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
  char *correctBuf =
      (char *)mmap(NULL, fsize1, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd1, 0);

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
  char *currResultBuf =
      (char *)mmap(NULL, fsize2, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd2, 0);
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
  shm_unlink(testLabel);
  munmap(currResultBuf, fsize2);
  // if (remove(testLabel) != 0) {
  //  throw runtime_error(string("Error deleting file"));
  // }

  return areEqual;
}

void runAndVerify(const char *testLabel, const char *planPath,
                  const char *testPath, const char *catalogJSON,
                  bool unordered) {
  PreparedStatement::from(planPath, testLabel, catalogJSON).execute();

  EXPECT_TRUE(
      verifyTestResult(testPath, testLabel, unordered));  // FIXME:!!!!!!!!!!!!!
  shm_unlink(testLabel);
}

bool TestEnvironment::has_already_been_setup = false;
