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

#ifndef RAW_GPU_PIPELINE_HPP_
#define RAW_GPU_PIPELINE_HPP_

#include "pipeline.hpp"
#include "util/jit/cpu-module.hpp"
#include "util/jit/gpu-module.hpp"

class GpuPipelineGen : public PipelineGen {
 protected:
  GpuModule module;
  CpuModule wrapper_module;

  bool wrapperModuleActive;
  StateVar kernel_id;
  StateVar strm_id;

  llvm::Function *Fconsume;
  std::map<std::string, llvm::Function *> availableWrapperFunctions;

 protected:
  GpuPipelineGen(Context *context, std::string pipName = "pip",
                 PipelineGen *copyStateFrom = nullptr);

  friend class GpuPipelineGenFactory;

 public:
  virtual void compileAndLoad();

  virtual llvm::Function *prepare();

  virtual std::unique_ptr<Pipeline> getPipeline(int group_id = 0);
  virtual void *getKernel() const;

  // virtual size_t appendStateVar (llvm::Type * ptype);
  // virtual size_t appendStateVar (llvm::Type * ptype,
  // std::function<init_func_t> init, std::function<deinit_func_t> deinit);
  virtual llvm::Module *getModule() const {
    if (wrapperModuleActive) return wrapper_module.getModule();
    return module.getModule();
  }

  virtual void *getConsume() const;
  virtual llvm::Function *getLLVMConsume() const { return Fconsume; }

  virtual void registerFunction(std::string, llvm::Function *);
  virtual llvm::Function *const getFunction(std::string funcName) const;

  virtual llvm::Function *const createHelperFunction(
      std::string funcName, std::vector<llvm::Type *> ins,
      std::vector<bool> readonly, std::vector<bool> noalias) const;

  virtual llvm::Value *workerScopedAtomicAdd(llvm::Value *ptr,
                                             llvm::Value *inc);
  virtual llvm::Value *workerScopedAtomicXchg(llvm::Value *ptr,
                                              llvm::Value *val);

  virtual void workerScopedMembar();

 protected:
  virtual size_t prepareStateArgument();
  virtual llvm::Value *getStateLLVMValue();
  virtual llvm::Value *getStateVar() const;

  virtual void prepareInitDeinit();

 public:
  virtual void *getCompiledFunction(llvm::Function *f);

 protected:
  virtual void registerFunctions();
  virtual llvm::Function *prepareConsumeWrapper();
  virtual void markAsKernel(llvm::Function *F) const;
};

class GpuPipelineGenFactory : public PipelineGenFactory {
 protected:
  GpuPipelineGenFactory() {}

 public:
  static PipelineGenFactory &getInstance() {
    static GpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) {
    return new GpuPipelineGen(context, pipName, copyStateFrom);
  }
};

extern "C" {
void *getPipKernel(PipelineGen *pip);

cudaStream_t createCudaStream();

void sync_strm(cudaStream_t strm);

void destroyCudaStream(cudaStream_t strm);
}

#endif /* RAW_GPU_PIPELINE_HPP_ */
