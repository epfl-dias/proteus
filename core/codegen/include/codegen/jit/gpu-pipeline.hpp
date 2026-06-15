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
#ifndef PROTEUS_GPU_PIPELINE_HPP
#define PROTEUS_GPU_PIPELINE_HPP

#include <codegen/jit/cpu-module.hpp>
#include <codegen/jit/gpu-module.hpp>
#include <codegen/jit/pipeline.hpp>

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
  void compileAndLoad() override;

  llvm::Function *prepare() override;

  std::unique_ptr<Pipeline> getPipeline(int group_id = 0) override;
  void *getKernel() override;

  // virtual size_t appendStateVar (llvm::Type * ptype);
  // virtual size_t appendStateVar (llvm::Type * ptype,
  // std::function<init_func_t> init, std::function<deinit_func_t> deinit);
  llvm::Module *getModule() const override {
    if (wrapperModuleActive) return wrapper_module.getModule();
    return module.getModule();
  }

  void *getConsume() override;
  llvm::Function *getLLVMConsume() const override { return Fconsume; }

  void registerFunction(std::string, llvm::Function *) override;
  [[nodiscard]] llvm::Function *getFunction(
      std::string funcName) const override;

  llvm::Function *const createHelperFunction(
      std::string funcName, std::vector<llvm::Type *> ins,
      std::vector<bool> readonly, std::vector<bool> noalias) override;

  llvm::Value *workerScopedAtomicAdd(llvm::Value *ptr,
                                     llvm::Value *inc) override;
  llvm::Value *workerScopedAtomicXchg(llvm::Value *ptr,
                                      llvm::Value *val) override;

  void workerScopedMembar() override;

  void enableWrapperModule();

  void disableWrapperModule();

 protected:
  size_t prepareStateArgument() override;
  llvm::Value *getStateLLVMValue() override;
  llvm::Value *getStateVar() const override;

  void prepareInitDeinit() override;

 public:
  void *getCompiledFunction(llvm::Function *f) override;

 protected:
  virtual llvm::Function *prepareConsumeWrapper();
  virtual void markAsKernel(llvm::Function *F) const;
};

class GpuPipelineGenFactory : public PipelineGenFactory {
 protected:
  GpuPipelineGenFactory() {}

  void registerFunctions(PipelineGen *pipelineGen) override;

 public:
  static PipelineGenFactory &getInstance() {
    static GpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) override;
};

extern "C" {
void *getPipKernel(PipelineGen *pip);

cudaStream_t createCudaStream();

void sync_strm(cudaStream_t strm);

void destroyCudaStream(cudaStream_t strm);
}

#endif  // PROTEUS_GPU_PIPELINE_HPP
