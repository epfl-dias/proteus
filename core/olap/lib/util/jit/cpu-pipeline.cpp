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

#include "cpu-pipeline.hpp"

#include "memory/block-manager.hpp"
#include "util/timing.hpp"

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
  //     /* OPTIMIZER PIPELINE, function passes */
  //     TheFPM = new legacy::FunctionPassManager(getModule());
  //     addOptimizerPipelineDefault(TheFPM);
  //     // TheFPM->add(createLoadCombinePass());

  //     //LSC: Seems to be faster without the vectorization, at least
  //     //while running the unit-tests, but this might be because the
  //     //datasets are too small.
  //     addOptimizerPipelineVectorization(TheFPM);

  // #if MODULEPASS
  //     /* OPTIMIZER PIPELINE, module passes */
  //     PassManagerBuilder pmb;
  //     pmb.OptLevel=3;
  //     TheMPM = new ModulePassManager();
  //     pmb.populateModulePassManager(*TheMPM);
  //     addOptimizerPipelineInlining(TheMPM);
  // #endif

  // TheFPM->doInitialization();
  Type *bool_type = Type::getInt1Ty(getModule()->getContext());
  // Type * cpp_bool_type = Type::getInt8Ty   (context->getLLVMContext());
  // static_assert(sizeof(bool) == 1, "Fix datatype");
  Type *int32_type = Type::getInt32Ty(getModule()->getContext());
  Type *int64_type = Type::getInt64Ty(getModule()->getContext());
  Type *void_type = Type::getVoidTy(getModule()->getContext());
  Type *charPtrType = Type::getInt8PtrTy(getModule()->getContext());
  Type *int32PtrType = Type::getInt32PtrTy(getModule()->getContext());

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
      FTlaunch_kernel, Function::ExternalLinkage, "launch_kernel", getModule());

  registerFunction("launch_kernel", launch_kernel_);

  FunctionType *FTlaunch_kernel_strm = FunctionType::get(
      void_type,
      std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0),
                          charPtrType},
      false);

  Function *launch_kernel_strm_ =
      Function::Create(FTlaunch_kernel_strm, Function::ExternalLinkage,
                       "launch_kernel_strm", getModule());

  registerFunction("launch_kernel_strm", launch_kernel_strm_);

  registerFunction("memset",
                   Intrinsic::getDeclaration(getModule(), Intrinsic::memset,
                                             {charPtrType, int64_type}));

  Type *pair_type = StructType::get(
      getModule()->getContext(), std::vector<Type *>{charPtrType, charPtrType});
  FunctionType *make_mem_move_device =
      FunctionType::get(pair_type,
                        std::vector<Type *>{charPtrType, size_type, int32_type,
                                            int64_type, charPtrType},
                        false);
  Function *fmake_mem_move_device =
      Function::Create(make_mem_move_device, Function::ExternalLinkage,
                       "make_mem_move_device", getModule());
  registerFunction("make_mem_move_device", fmake_mem_move_device);

  FunctionType *MemMoveConf_pull = FunctionType::get(
      charPtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fMemMoveConf_pull =
      Function::Create(MemMoveConf_pull, Function::ExternalLinkage,
                       "MemMoveConf_pull", getModule());
  registerFunction("MemMoveConf_pull", fMemMoveConf_pull);

  FunctionType *make_mem_move_broadcast_device =
      FunctionType::get(pair_type,
                        std::vector<Type *>{charPtrType, size_type, int32_type,
                                            charPtrType, bool_type},
                        false);  // cpp_bool_type
  Function *fmake_mem_move_broadcast_device = Function::Create(
      make_mem_move_broadcast_device, Function::ExternalLinkage,
      "make_mem_move_broadcast_device", getModule());
  registerFunction("make_mem_move_broadcast_device",
                   fmake_mem_move_broadcast_device);

  FunctionType *make_mem_move_local_to = FunctionType::get(
      pair_type,
      std::vector<Type *>{charPtrType, size_type, int32_type, charPtrType},
      false);
  Function *fmake_mem_move_local_to =
      Function::Create(make_mem_move_local_to, Function::ExternalLinkage,
                       "make_mem_move_local_to", getModule());
  registerFunction("make_mem_move_local_to", fmake_mem_move_local_to);

  FunctionType *step_mmc_mem_move_broadcast_device =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fstep_mmc_mem_move_broadcast_device = Function::Create(
      step_mmc_mem_move_broadcast_device, Function::ExternalLinkage,
      "step_mmc_mem_move_broadcast_device", getModule());
  registerFunction("step_mmc_mem_move_broadcast_device",
                   fstep_mmc_mem_move_broadcast_device);

  FunctionType *acquireBuffer = FunctionType::get(
      charPtrType, std::vector<Type *>{int32_type, charPtrType}, false);
  Function *facquireBuffer = Function::Create(
      acquireBuffer, Function::ExternalLinkage, "acquireBuffer", getModule());
  {
    std::vector<std::pair<unsigned, Attribute>> attrs;
    Attribute def = Attribute::getWithDereferenceableBytes(
        getModule()->getContext(),
        BlockManager::block_size);  // FIXME: at some point this should
                                    // change...
    attrs.emplace_back(0, def);
    facquireBuffer->setAttributes(
        AttributeList::get(getModule()->getContext(), attrs));
  }
  registerFunction("acquireBuffer", facquireBuffer);

  FunctionType *try_acquireBuffer = FunctionType::get(
      charPtrType, std::vector<Type *>{int32_type, charPtrType}, false);
  Function *ftry_acquireBuffer =
      Function::Create(try_acquireBuffer, Function::ExternalLinkage,
                       "try_acquireBuffer", getModule());
  {
    std::vector<std::pair<unsigned, Attribute>> attrs;
    Attribute def = Attribute::getWithDereferenceableOrNullBytes(
        getModule()->getContext(),
        BlockManager::block_size);  // FIXME: at some point this should
                                    // change...
    attrs.emplace_back(0, def);
    ftry_acquireBuffer->setAttributes(
        AttributeList::get(getModule()->getContext(), attrs));
  }
  registerFunction("try_acquireBuffer", ftry_acquireBuffer);

  FunctionType *allocate =
      FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
  Function *fallocate = Function::Create(allocate, Function::ExternalLinkage,
                                         "allocate_pinned", getModule());
  std::vector<std::pair<unsigned, Attribute>> attrs;
  Attribute noAlias =
      Attribute::get(getModule()->getContext(), Attribute::AttrKind::NoAlias);
  attrs.emplace_back(0, noAlias);
  fallocate->setAttributes(
      AttributeList::get(getModule()->getContext(), attrs));
  registerFunction("allocate", fallocate);

  FunctionType *deallocate =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fdeallocate = Function::Create(
      deallocate, Function::ExternalLinkage, "deallocate_pinned", getModule());
  registerFunction("deallocate", fdeallocate);

  FunctionType *releaseBuffer = FunctionType::get(
      void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType},
      false);
  Function *freleaseBuffer = Function::Create(
      releaseBuffer, Function::ExternalLinkage, "releaseBuffer", getModule());
  registerFunction("releaseBuffer", freleaseBuffer);

  FunctionType *freeBuffer = FunctionType::get(
      void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType},
      false);
  Function *ffreeBuffer = Function::Create(
      freeBuffer, Function::ExternalLinkage, "freeBuffer", getModule());
  registerFunction("freeBuffer", ffreeBuffer);

  FunctionType *crand =
      FunctionType::get(int32_type, std::vector<Type *>{}, false);
  Function *fcrand =
      Function::Create(crand, Function::ExternalLinkage, "rand", getModule());
  registerFunction("rand", fcrand);

  FunctionType *get_buffer =
      FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
  Function *fget_buffer = Function::Create(
      get_buffer, Function::ExternalLinkage, "get_buffer", getModule());
  fget_buffer->setReturnDoesNotAlias();
  registerFunction("get_buffer", fget_buffer);

  FunctionType *release_buffer =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *frelease_buffer = Function::Create(
      release_buffer, Function::ExternalLinkage, "release_buffer", getModule());
  registerFunction("release_buffer", frelease_buffer);
  registerFunction("release_buffers", frelease_buffer);

  FunctionType *yield =
      FunctionType::get(void_type, std::vector<Type *>{}, false);
  Function *fyield =
      Function::Create(yield, Function::ExternalLinkage, "yield", getModule());
  registerFunction("yield", fyield);

  FunctionType *get_ptr_device =
      FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
  Function *fget_ptr_device = Function::Create(
      get_ptr_device, Function::ExternalLinkage, "get_ptr_device", getModule());
  registerFunction("get_ptr_device", fget_ptr_device);

  FunctionType *get_ptr_device_or_rand_for_host =
      FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
  Function *fget_ptr_device_or_rand_for_host = Function::Create(
      get_ptr_device_or_rand_for_host, Function::ExternalLinkage,
      "get_ptr_device_or_rand_for_host", getModule());
  registerFunction("get_ptr_device_or_rand_for_host",
                   fget_ptr_device_or_rand_for_host);

  FunctionType *get_rand_core_local_to_ptr =
      FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
  Function *fget_rand_core_local_to_ptr =
      Function::Create(get_rand_core_local_to_ptr, Function::ExternalLinkage,
                       "get_rand_core_local_to_ptr", getModule());
  registerFunction("get_rand_core_local_to_ptr", fget_rand_core_local_to_ptr);

  FunctionType *rand_local_cpu = FunctionType::get(
      int32_type, std::vector<Type *>{charPtrType, int64_type}, false);
  Function *frand_local_cpu = Function::Create(
      rand_local_cpu, Function::ExternalLinkage, "rand_local_cpu", getModule());
  registerFunction("rand_local_cpu", frand_local_cpu);

  FunctionType *acquireWorkUnit =
      FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
  Function *facquireWorkUnit =
      Function::Create(acquireWorkUnit, Function::ExternalLinkage,
                       "acquireWorkUnit", getModule());
  registerFunction("acquireWorkUnit", facquireWorkUnit);

  FunctionType *propagateWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, bool_type},
      false);
  Function *fpropagateWorkUnit =
      Function::Create(propagateWorkUnit, Function::ExternalLinkage,
                       "propagateWorkUnit", getModule());
  registerFunction("propagateWorkUnit", fpropagateWorkUnit);

  FunctionType *acquirePendingWorkUnit = FunctionType::get(
      bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *facquirePendingWorkUnit =
      Function::Create(acquirePendingWorkUnit, Function::ExternalLinkage,
                       "acquirePendingWorkUnit", getModule());
  registerFunction("acquirePendingWorkUnit", facquirePendingWorkUnit);

  FunctionType *releaseWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *freleaseWorkUnit =
      Function::Create(releaseWorkUnit, Function::ExternalLinkage,
                       "releaseWorkUnit", getModule());
  registerFunction("releaseWorkUnit", freleaseWorkUnit);

  FunctionType *acquireWorkUnitBroadcast =
      FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
  Function *facquireWorkUnitBroadcast =
      Function::Create(acquireWorkUnitBroadcast, Function::ExternalLinkage,
                       "acquireWorkUnitBroadcast", getModule());
  registerFunction("acquireWorkUnitBroadcast", facquireWorkUnitBroadcast);

  FunctionType *propagateWorkUnitBroadcast = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, int32_type},
      false);
  Function *fpropagateWorkUnitBroadcast =
      Function::Create(propagateWorkUnitBroadcast, Function::ExternalLinkage,
                       "propagateWorkUnitBroadcast", getModule());
  registerFunction("propagateWorkUnitBroadcast", fpropagateWorkUnitBroadcast);

  FunctionType *acquirePendingWorkUnitBroadcast = FunctionType::get(
      bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *facquirePendingWorkUnitBroadcast = Function::Create(
      acquirePendingWorkUnitBroadcast, Function::ExternalLinkage,
      "acquirePendingWorkUnitBroadcast", getModule());
  registerFunction("acquirePendingWorkUnitBroadcast",
                   facquirePendingWorkUnitBroadcast);

  FunctionType *releaseWorkUnitBroadcast = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *freleaseWorkUnitBroadcast =
      Function::Create(releaseWorkUnitBroadcast, Function::ExternalLinkage,
                       "releaseWorkUnitBroadcast", getModule());
  registerFunction("releaseWorkUnitBroadcast", freleaseWorkUnitBroadcast);

  FunctionType *mem_move_local_to_acquireWorkUnit =
      FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
  Function *fmem_move_local_to_acquireWorkUnit = Function::Create(
      mem_move_local_to_acquireWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_acquireWorkUnit", getModule());
  registerFunction("mem_move_local_to_acquireWorkUnit",
                   fmem_move_local_to_acquireWorkUnit);

  FunctionType *mem_move_local_to_propagateWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, bool_type},
      false);
  Function *fmem_move_local_to_propagateWorkUnit = Function::Create(
      mem_move_local_to_propagateWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_propagateWorkUnit", getModule());
  registerFunction("mem_move_local_to_propagateWorkUnit",
                   fmem_move_local_to_propagateWorkUnit);

  FunctionType *mem_move_local_to_acquirePendingWorkUnit = FunctionType::get(
      bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fmem_move_local_to_acquirePendingWorkUnit = Function::Create(
      mem_move_local_to_acquirePendingWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_acquirePendingWorkUnit", getModule());
  registerFunction("mem_move_local_to_acquirePendingWorkUnit",
                   fmem_move_local_to_acquirePendingWorkUnit);

  FunctionType *mem_move_local_to_releaseWorkUnit = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fmem_move_local_to_releaseWorkUnit = Function::Create(
      mem_move_local_to_releaseWorkUnit, Function::ExternalLinkage,
      "mem_move_local_to_releaseWorkUnit", getModule());
  registerFunction("mem_move_local_to_releaseWorkUnit",
                   fmem_move_local_to_releaseWorkUnit);

  FunctionType *getClusterCounts = FunctionType::get(
      int32PtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetClusterCounts =
      Function::Create(getClusterCounts, Function::ExternalLinkage,
                       "getClusterCounts", getModule());
  registerFunction("getClusterCounts", fgetClusterCounts);

  FunctionType *getRelationMem = FunctionType::get(
      charPtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetRelationMem = Function::Create(
      getRelationMem, Function::ExternalLinkage, "getRelationMem", getModule());
  registerFunction("getRelationMem", fgetRelationMem);

  FunctionType *getHTMemKV = FunctionType::get(
      charPtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetHTMemKV = Function::Create(
      getHTMemKV, Function::ExternalLinkage, "getHTMemKV", getModule());
  registerFunction("getHTMemKV", fgetHTMemKV);

  FunctionType *registerClusterCounts = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, int32PtrType, charPtrType},
      false);
  Function *fregisterClusterCounts =
      Function::Create(registerClusterCounts, Function::ExternalLinkage,
                       "registerClusterCounts", getModule());
  registerFunction("registerClusterCounts", fregisterClusterCounts);

  FunctionType *registerRelationMem = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, charPtrType},
      false);
  Function *fregisterRelationMem =
      Function::Create(registerRelationMem, Function::ExternalLinkage,
                       "registerRelationMem", getModule());
  registerFunction("registerRelationMem", fregisterRelationMem);

  FunctionType *registerHTMemKV = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, charPtrType},
      false);
  Function *fregisterHTMemKV =
      Function::Create(registerHTMemKV, Function::ExternalLinkage,
                       "registerHTMemKV", getModule());
  registerFunction("registerHTMemKV", fregisterHTMemKV);

  FunctionType *cfree =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fcfree =
      Function::Create(cfree, Function::ExternalLinkage, "free", getModule());
  registerFunction("free", fcfree);

  FunctionType *qsort_cmp = FunctionType::get(
      int32_type, std::vector<Type *>{charPtrType, charPtrType}, false);
  FunctionType *qsort =
      FunctionType::get(void_type,
                        std::vector<Type *>{charPtrType, size_type, size_type,
                                            PointerType::getUnqual(qsort_cmp)},
                        false);
  Function *fqsort =
      Function::Create(qsort, Function::ExternalLinkage, "qsort", getModule());
  registerFunction("qsort", fqsort);

  registerFunctions();  // FIXME: do we have to register them every time ?
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
  func = std::async(std::launch::async,
                    [m = module.get(), fname = F->getName().str()]() {
                      return m->getCompiledFunction(fname);
                    });
}
