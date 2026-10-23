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
#ifndef PROTEUS_OLAP_PIPELINE_HPP
#define PROTEUS_OLAP_PIPELINE_HPP

#include <codegen/jit/cpu-pipeline.hpp>
#include <codegen/jit/gpu-pipeline.hpp>

class OlapCpuPipelineGenFactory : public CpuPipelineGenFactory {
 protected:
  OlapCpuPipelineGenFactory() = default;

  void registerFunctions(PipelineGen *pipelineGen) override;

 public:
  static OlapCpuPipelineGenFactory &getInstance() {
    static OlapCpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) override;
};

class OlapGpuPipelineGenFactory : public GpuPipelineGenFactory {
 protected:
  OlapGpuPipelineGenFactory() = default;

  void registerFunctions(PipelineGen *pipelineGen) override;

 public:
  static OlapGpuPipelineGenFactory &getInstance() {
    static OlapGpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) override;
};

#endif  // PROTEUS_OLAP_PIPELINE_HPP
