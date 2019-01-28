/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef RAW_GPU_PIPELINE_HPP_
#define RAW_GPU_PIPELINE_HPP_

#include "util/jit/raw-cpu-module.hpp"
#include "util/jit/raw-gpu-module.hpp"
#include "util/raw-pipeline.hpp"

class RawGpuPipelineGen : public RawPipelineGen {
 protected:
  RawGpuModule module;
  RawCpuModule wrapper_module;

  bool wrapperModuleActive;
  size_t kernel_id;
  size_t strm_id;

  Function *Fconsume;
  Function *subpipelineSync;
  map<string, Function *> availableWrapperFunctions;

 protected:
  RawGpuPipelineGen(RawContext *context, std::string pipName = "pip",
                    RawPipelineGen *copyStateFrom = NULL);

  friend class RawGpuPipelineGenFactory;

 public:
  virtual void compileAndLoad();

  virtual Function *prepare();

  virtual RawPipeline *getPipeline(int group_id = 0);
  virtual void *getKernel() const;

  // virtual size_t appendStateVar (llvm::Type * ptype);
  // virtual size_t appendStateVar (llvm::Type * ptype,
  // std::function<init_func_t> init, std::function<deinit_func_t> deinit);
  virtual Module *getModule() const {
    if (wrapperModuleActive) return wrapper_module.getModule();
    return module.getModule();
  }

  virtual void *getConsume() const;
  virtual Function *getLLVMConsume() const { return Fconsume; }

  virtual void registerFunction(const char *, Function *);
  virtual Function *const getFunction(string funcName) const;

  virtual Function *const createHelperFunction(string funcName,
                                               std::vector<llvm::Type *> ins,
                                               std::vector<bool> readonly,
                                               std::vector<bool> noalias) const;

 protected:
  virtual size_t prepareStateArgument();
  virtual llvm::Value *getStateLLVMValue();
  virtual llvm::Value *getStateVar() const;

  virtual void prepareInitDeinit();

 public:
  virtual void *getCompiledFunction(Function *f);

 protected:
  virtual void registerFunctions();
  virtual Function *prepareConsumeWrapper();
  virtual void markAsKernel(Function *F) const;
};

class RawGpuPipelineGenFactory : public RawPipelineGenFactory {
 protected:
  RawGpuPipelineGenFactory() {}

 public:
  static RawPipelineGenFactory &getInstance() {
    static RawGpuPipelineGenFactory instance;
    return instance;
  }

  RawPipelineGen *create(RawContext *context, std::string pipName,
                         RawPipelineGen *copyStateFrom) {
    return new RawGpuPipelineGen(context, pipName, copyStateFrom);
  }
};

extern "C" {
void *getPipKernel(RawPipelineGen *pip);

cudaStream_t createCudaStream();

void sync_strm(cudaStream_t strm);

void destroyCudaStream(cudaStream_t strm);
}

#endif /* RAW_GPU_PIPELINE_HPP_ */