/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <cli-flags.hpp>
#include <filesystem>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/util/glog.hpp>
#include <span>
#include <sstream>
#include <storage/mmap-file.hpp>

#include "compressed-file.hpp"
#include "cpu-benchmarks.hpp"
#include "gpu-benchmarks.hpp"
#include "platform/util/profiling.hpp"

DECLARE_string(input_md_file);
DEFINE_string(input_md_file, "",
              "Path to metadata json for a compressed column.");

DECLARE_bool(decomp_cpu_in_cpu_memory);
DEFINE_bool(decomp_cpu_in_cpu_memory, false,
            "Benchmark CPU decompression on in-memory data");

DECLARE_bool(decomp_gpu_in_gpu_memory);
DEFINE_bool(decomp_gpu_in_gpu_memory, false,
            "Benchmark GPU decompression on in-gpu-memory data");

DECLARE_bool(decomp_gpu_in_cpu_memory);
DEFINE_bool(decomp_gpu_in_cpu_memory, false,
            "Benchmark GPU decompression on in-cpu-memory data");

#ifdef HAVE_NVCOMP
constexpr bool kHaveNvcomp = true;
#else
constexpr bool kHaveNvcomp = false;
#endif

int main(int argc, char* argv[]) {
  auto ctx = proteus::from_cli::olap("compression-perf", &argc, &argv);
  auto& topo = topology::getInstance();
  auto numa_exec_scope = topo.getCpuNumaNodes()[0].set_on_scope();

  std::stringstream results;

  CHECK(!FLAGS_input_md_file.empty()) << "input_md_file is required";
  CHECK(std::filesystem::exists(FLAGS_input_md_file))
      << "input_md_file does not exist. " << FLAGS_input_md_file;
  LOG(INFO) << "Using input metadata file: " << FLAGS_input_md_file;

  if (FLAGS_decomp_cpu_in_cpu_memory) {
    CompressedFile cb(FLAGS_input_md_file, data_loc::PINNED);
    results << benchmark_lz4_cpu_decompression(cb, {1, 2, 4, 8, 16, 24})
            << std::endl;
  }

  if (FLAGS_decomp_gpu_in_gpu_memory) {
    if (!kHaveNvcomp) {
      LOG(ERROR) << "nvcomp not available, skipping GPU decompression";
    } else {
      CHECK_GT(topo.getGpuCount(), 0) << "No GPUs available";
      auto gpu_scope = topo.getGpus()[0].set_on_scope();
      CompressedFile cb(FLAGS_input_md_file, data_loc::GPU_RESIDENT);
      results << benchmark_lz4_gpu_decompression_gpu_mem(cb) << std::endl;
    }
  }

  if (FLAGS_decomp_gpu_in_cpu_memory) {
    if (!kHaveNvcomp) {
      LOG(ERROR) << "nvcomp not available, skipping GPU decompression";
    } else {
      CHECK_GT(topo.getGpuCount(), 0) << "No GPUs available";
      auto gpu_scope = topo.getGpus()[0].set_on_scope();
      CompressedFile cb(FLAGS_input_md_file, data_loc::PINNED);
      results << benchmark_lz4_gpu_decompression_from_cpu_mem(cb) << std::endl;
    }
  }

  std::cout << results.str() << std::endl;
}
