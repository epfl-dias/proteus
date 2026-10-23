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
#include <codegen/jit/cpu-pipeline.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/util/timing.hpp>

using namespace llvm;

CpuPipelineGen::CpuPipelineGen(Context *context, std::string pipName,
                               PipelineGen *copyStateFrom)
    : PipelineGen(context, pipName, copyStateFrom),
      module(std::make_unique<CpuModule>(context, pipName)) {
  registerSubPipeline();

  if (copyStateFrom) {
    Type *charPtrType = Type::getInt8PtrTy(getModule()->getContext());
    appendStateVar(charPtrType);
  }
}

void *CpuPipelineGen::getCompiledFunction(Function *f) {
  time_block t(TimeRegistry::Key{"Compile and Load (CPU, waiting - critical)"});
  return module->getCompiledFunction(f);
}

const llvm::DataLayout &CpuPipelineGen::getDataLayout() const {
  return CpuModule::getDL();
}

void CpuPipelineGen::compileAndLoad() {
  module->compileAndLoad();
  compiledFunctionFuture = std::async(
      std::launch::async, [m = module.get(), fname = F->getName().str()]() {
        return m->getCompiledFunction(fname);
      });
}

void CpuPipelineGenFactory::registerFunctions(PipelineGen *pipelineGen) {
  PipelineGenFactory::registerFunctions(
      pipelineGen);  // FIXME: do we have to register them every time ?

  auto *llvmModule = pipelineGen->getModule();
  Type *bool_type = Type::getInt1Ty(llvmModule->getContext());
  Type *int32_type = Type::getInt32Ty(llvmModule->getContext());
  Type *int64_type = Type::getInt64Ty(llvmModule->getContext());
  Type *void_type = Type::getVoidTy(llvmModule->getContext());
  Type *charPtrType = Type::getInt8PtrTy(llvmModule->getContext());

  Type *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  FunctionType *FTlaunch_kernel = FunctionType::get(
      void_type,
      std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0)},
      false);

  Function *launch_kernel_ = Function::Create(
      FTlaunch_kernel, Function::ExternalLinkage, "launch_kernel", llvmModule);

  pipelineGen->registerFunction("launch_kernel", launch_kernel_);

  FunctionType *FTlaunch_kernel_strm = FunctionType::get(
      void_type,
      std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0),
                          charPtrType},
      false);

  Function *launch_kernel_strm_ =
      Function::Create(FTlaunch_kernel_strm, Function::ExternalLinkage,
                       "launch_kernel_strm", llvmModule);

  pipelineGen->registerFunction("launch_kernel_strm", launch_kernel_strm_);

  FunctionType *FTlaunch_kernel_strm_sized = FunctionType::get(
      void_type,
      std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0),
                          charPtrType, int32_type, int32_type},
      false);
  Function *launch_kernel_strm_sized_ =
      Function::Create(FTlaunch_kernel_strm_sized, Function::ExternalLinkage,
                       "launch_kernel_strm_sized", llvmModule);
  pipelineGen->registerFunction("launch_kernel_strm_sized",
                                launch_kernel_strm_sized_);

  pipelineGen->registerFunction(
      "memset", Intrinsic::getDeclaration(llvmModule, Intrinsic::memset,
                                          {charPtrType, int64_type}));

  Type *pair_type = StructType::get(
      llvmModule->getContext(), std::vector<Type *>{charPtrType, charPtrType});
  FunctionType *make_mem_move_local_to = FunctionType::get(
      pair_type,
      std::vector<Type *>{charPtrType, size_type, int32_type, charPtrType},
      false);
  Function *fmake_mem_move_local_to =
      Function::Create(make_mem_move_local_to, Function::ExternalLinkage,
                       "make_mem_move_local_to", llvmModule);
  pipelineGen->registerFunction("make_mem_move_local_to",
                                fmake_mem_move_local_to);

  FunctionType *allocate =
      FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
  Function *fallocate = Function::Create(allocate, Function::ExternalLinkage,
                                         "allocate_pinned", llvmModule);
  std::vector<std::pair<unsigned, Attribute>> attrs;
  Attribute noAlias =
      Attribute::get(llvmModule->getContext(), Attribute::AttrKind::NoAlias);
  attrs.emplace_back(0, noAlias);
  fallocate->setAttributes(AttributeList::get(llvmModule->getContext(), attrs));
  pipelineGen->registerFunction("allocate", fallocate);

  FunctionType *deallocate =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fdeallocate = Function::Create(
      deallocate, Function::ExternalLinkage, "deallocate_pinned", llvmModule);
  pipelineGen->registerFunction("deallocate", fdeallocate);

  FunctionType *crand =
      FunctionType::get(int32_type, std::vector<Type *>{}, false);
  Function *fcrand =
      Function::Create(crand, Function::ExternalLinkage, "rand", llvmModule);
  pipelineGen->registerFunction("rand", fcrand);

  FunctionType *get_buffer =
      FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
  Function *fget_buffer = Function::Create(
      get_buffer, Function::ExternalLinkage, "get_buffer", llvmModule);
  fget_buffer->setReturnDoesNotAlias();
  pipelineGen->registerFunction("get_buffer", fget_buffer);

  Function *fget_pinned_buffer = Function::Create(
      get_buffer, Function::ExternalLinkage, "get_pinned_buffer", llvmModule);
  fget_pinned_buffer->setReturnDoesNotAlias();
  pipelineGen->registerFunction("get_pinned_buffer", fget_pinned_buffer);

  FunctionType *release_buffer =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *frelease_buffer = Function::Create(
      release_buffer, Function::ExternalLinkage, "release_buffer", llvmModule);
  pipelineGen->registerFunction("release_buffer", frelease_buffer);
  pipelineGen->registerFunction("release_buffers", frelease_buffer);

  Function *frelease_pinned_buffer =
      Function::Create(release_buffer, Function::ExternalLinkage,
                       "release_pinned_buffer", llvmModule);
  pipelineGen->registerFunction("release_pinned_buffer",
                                frelease_pinned_buffer);

  FunctionType *yield =
      FunctionType::get(void_type, std::vector<Type *>{}, false);
  Function *fyield =
      Function::Create(yield, Function::ExternalLinkage, "yield", llvmModule);
  pipelineGen->registerFunction("yield", fyield);

  FunctionType *rand_local_cpu = FunctionType::get(
      int32_type, std::vector<Type *>{charPtrType, int64_type}, false);
  Function *frand_local_cpu = Function::Create(
      rand_local_cpu, Function::ExternalLinkage, "rand_local_cpu", llvmModule);
  pipelineGen->registerFunction("rand_local_cpu", frand_local_cpu);

  FunctionType *mem_move_local_to_acquireWorkUnit =
      FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
  Function *fmem_move_local_to_acquireWorkUnit = Function::Create(
      mem_move_local_to_acquireWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_acquireWorkUnit", llvmModule);
  pipelineGen->registerFunction("mem_move_local_to_acquireWorkUnit",
                                fmem_move_local_to_acquireWorkUnit);

  FunctionType *mem_move_local_to_propagateWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, bool_type},
      false);
  Function *fmem_move_local_to_propagateWorkUnit = Function::Create(
      mem_move_local_to_propagateWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_propagateWorkUnit", llvmModule);
  pipelineGen->registerFunction("mem_move_local_to_propagateWorkUnit",
                                fmem_move_local_to_propagateWorkUnit);

  FunctionType *mem_move_local_to_acquirePendingWorkUnit = FunctionType::get(
      bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fmem_move_local_to_acquirePendingWorkUnit = Function::Create(
      mem_move_local_to_acquirePendingWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_acquirePendingWorkUnit", llvmModule);
  pipelineGen->registerFunction("mem_move_local_to_acquirePendingWorkUnit",
                                fmem_move_local_to_acquirePendingWorkUnit);

  FunctionType *mem_move_local_to_releaseWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fmem_move_local_to_releaseWorkUnit = Function::Create(
      mem_move_local_to_releaseWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_releaseWorkUnit", llvmModule);
  pipelineGen->registerFunction("mem_move_local_to_releaseWorkUnit",
                                fmem_move_local_to_releaseWorkUnit);

  FunctionType *cfree =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fcfree =
      Function::Create(cfree, Function::ExternalLinkage, "free", llvmModule);
  pipelineGen->registerFunction("free", fcfree);

  FunctionType *qsort_cmp = FunctionType::get(
      int32_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  FunctionType *qsort =
      FunctionType::get(void_type,
                        std::vector<Type *>{charPtrType, size_type, size_type,
                                            PointerType::getUnqual(qsort_cmp)},
                        false);
  Function *fqsort =
      Function::Create(qsort, Function::ExternalLinkage, "qsort", llvmModule);
  pipelineGen->registerFunction("qsort", fqsort);
}

PipelineGen *CpuPipelineGenFactory::create(Context *context,
                                           std::string pipName,
                                           PipelineGen *copyStateFrom) {
  auto *pipelineGen = new CpuPipelineGen(context, pipName, copyStateFrom);
  CpuPipelineGenFactory::registerFunctions(pipelineGen);
  return static_cast<PipelineGen *>(pipelineGen);
}
