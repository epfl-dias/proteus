/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#ifndef PROTEUS_PIPELINE_ENVIRONMENT_HPP
#define PROTEUS_PIPELINE_ENVIRONMENT_HPP

#include <gtest/gtest.h>

#include <codegen/jit/pipeline.hpp>
#include <memory>
#include <platform/common/common.hpp>

/**
 * @class PipelineTestEnvironment
 * is used to create a minimal environment to use Pipeline-related functions.
 *
 * The main purpose of this class is to initialize platform and LLVM JIT.
 */
class PipelineTestEnvironment : public ::testing::Environment {
 public:
  /**
   * Initialize platform with memory constants and LLVM JIT
   */
  void SetUp() override;

  /**
   * Destruct platform
   */
  void TearDown() override;

 private:
  static constexpr float gpu_mem_pool_percentage = 0.05;
  static constexpr float cpu_mem_pool_percentage = 0.05;
  static constexpr size_t log_buffers = 0;

  std::unique_ptr<proteus::platform> platform;
};

#endif  // PROTEUS_PIPELINE_ENVIRONMENT_HPP
