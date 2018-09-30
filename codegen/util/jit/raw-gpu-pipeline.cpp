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

#include "raw-gpu-pipeline.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
// #include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/CodeGen/TargetPassConfig.h"

#include "util/jit/raw-cpu-pipeline.hpp"

RawGpuPipelineGen::RawGpuPipelineGen(RawContext * context, std::string pipName, RawPipelineGen * copyStateFrom): 
            RawPipelineGen      (context, pipName, copyStateFrom),
            module              (context, pipName),
            wrapper_module      (context, pipName + "_wrapper"),
            wrapperModuleActive (false){
    //overide defaults
    //NOTE: should we set this ones for the CPU submodule as well?
    maxBlockSize = defaultBlockDim.x * defaultBlockDim.y * defaultBlockDim.z;
    maxGridSize  = defaultGridDim .x * defaultGridDim .y * defaultGridDim .z;

    registerSubPipeline();
    registerFunctions();

    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * int64_type   = Type::getInt64Ty  (context->getLLVMContext());
    Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());
    Type * f32PtrType   = Type::getFloatPtrTy(context->getLLVMContext());
    Type * f32Type      = Type::getFloatTy(context->getLLVMContext());
    Type * f64PtrType   = Type::getDoublePtrTy(context->getLLVMContext());
    Type * f64Type      = Type::getDoubleTy(context->getLLVMContext());

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    kernel_id           = appendStateVar(charPtrType,
                                [=](Value *){
                                    Function * f   = this->getFunction("getPipKernel");
                                    Value    * pip = ConstantInt::get((IntegerType *) int64_type, (int64_t) this);
                                    pip            = getBuilder()->CreateIntToPtr(pip, charPtrType);
                                    return getBuilder()->CreateCall(f, pip);
                                },
                                [=](Value *, Value *){}
                            );
    strm_id             = appendStateVar(charPtrType,
                                [=](Value *){
                                    Function * f = this->getFunction("createCudaStream");
                                    return getBuilder()->CreateCall(f);
                                },
                                [=](Value *, Value * strm){
                                    Function * f = this->getFunction("destroyCudaStream");
                                    getBuilder()->CreateCall(f, strm);
                                }
                            );

    registerFunction("atomicAdd_double"                     , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_atomic_load_add_f64, f64PtrType));
    registerFunction("atomicAdd_float"                      , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_atomic_load_add_f32, f32PtrType));

    registerFunction("llvm.nvvm.shfl.bfly.i32"              , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_bfly_i32)             );

    registerFunction("llvm.nvvm.shfl.idx.i32"               , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_idx_i32)              );

    registerFunction("llvm.nvvm.read.ptx.sreg.ntid.x"       , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_ntid_x)      );

    registerFunction("llvm.nvvm.read.ptx.sreg.tid.x"        , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_tid_x)       );

    registerFunction("llvm.nvvm.read.ptx.sreg.lanemask.lt"  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_lanemask_lt) );

    registerFunction("llvm.nvvm.read.ptx.sreg.lanemask.eq"  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_lanemask_eq) );
    
    registerFunction("llvm.nvvm.read.ptx.sreg.nctaid.x"     , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x)    );

    registerFunction("llvm.nvvm.read.ptx.sreg.ctaid.x"      , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x)     );

    registerFunction("llvm.nvvm.read.ptx.sreg.laneid"       , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_laneid)      );

    registerFunction("llvm.nvvm.membar.cta"                 , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_cta)                );
    registerFunction("threadfence_block"                    , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_cta)                );

    registerFunction("llvm.nvvm.membar.gl"                  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_gl)                 );
    registerFunction("threadfence"                          , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_gl)                 );

    registerFunction("llvm.nvvm.membar.sys"                 , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_sys)                );

    registerFunction("llvm.nvvm.barrier0"                   , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_barrier0)                  );
    registerFunction("syncthreads"                          , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_barrier0)                  );

    registerFunction("llvm.ctpop"                           , Intrinsic::getDeclaration(getModule(), Intrinsic::ctpop         , int32_type     ));




    registerFunction("llvm.nvvm.bar.warp.sync"              , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_bar_warp_sync)             );
    
    registerFunction("llvm.nvvm.shfl.sync.bfly.i32"         , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_sync_bfly_i32)        );
    registerFunction("llvm.nvvm.shfl.sync.idx.i32"          , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_sync_idx_i32)         );




    FunctionType *intrprinti64 = FunctionType::get(void_type, std::vector<Type *>{int64_type}, false);
    Function *intr_pprinti64 = Function::Create(intrprinti64, Function::ExternalLinkage, "dprinti64", getModule());
    registerFunction("printi64", intr_pprinti64);

    FunctionType *intrprinti = FunctionType::get(void_type, std::vector<Type *>{int32_type}, false);
    Function *intr_pprinti = Function::Create(intrprinti, Function::ExternalLinkage, "dprinti", getModule());
    registerFunction("printi", intr_pprinti);

    FunctionType *intrprintptr = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *intr_pprintptr = Function::Create(intrprintptr, Function::ExternalLinkage, "dprintptr", getModule());
    registerFunction("printptr", intr_pprintptr);

    FunctionType *intrget_buffers = FunctionType::get(charPtrType, std::vector<Type *>{}, false);
    Function *intr_pget_buffers = Function::Create(intrget_buffers, Function::ExternalLinkage, "get_buffers", getModule());
    registerFunction("get_buffers", intr_pget_buffers);

    FunctionType *intrrelease_buffers = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *intr_prelease_buffers = Function::Create(intrrelease_buffers, Function::ExternalLinkage, "release_buffers", getModule());
    registerFunction("release_buffers", intr_prelease_buffers);
}

void RawGpuPipelineGen::registerFunctions(){
    RawPipelineGen::registerFunctions();
    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * int64_type   = Type::getInt64Ty  (context->getLLVMContext());
    Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());
    Type * bool_type    = Type::getInt1Ty   (context->getLLVMContext());

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    FunctionType *allocate = FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
    Function *fallocate = Function::Create(allocate, Function::ExternalLinkage, "allocate_gpu", getModule());
    std::vector<std::pair<unsigned, Attribute>> attrs;
    Attribute noAlias  = Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias);
    attrs.emplace_back(0, noAlias);
    fallocate->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    registerFunction("allocate", fallocate);

    FunctionType *deallocate = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *fdeallocate = Function::Create(deallocate, Function::ExternalLinkage, "deallocate_gpu", getModule());
    registerFunction("deallocate", fdeallocate);


    FunctionType *memcpy = FunctionType::get(void_type, std::vector<Type *>{charPtrType, charPtrType, size_type, bool_type}, false);
    Function *fmemcpy = Function::Create(memcpy, Function::ExternalLinkage, "memcpy_gpu", getModule());
    registerFunction("memcpy", fmemcpy);

    FunctionType *intrqsort = FunctionType::get(void_type, std::vector<Type *>{charPtrType, size_type}, false);
    Function *intr_pqsort_i = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_i", getModule());
    registerFunction("qsort_i", intr_pqsort_i);

    Function *intr_pqsort_l = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_l", getModule());
    registerFunction("qsort_l", intr_pqsort_l);

    Function *intr_pqsort_ii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ii", getModule());
    registerFunction("qsort_ii", intr_pqsort_ii);

    Function *intr_pqsort_il = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_il", getModule());
    registerFunction("qsort_il", intr_pqsort_il);

    Function *intr_pqsort_li = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_li", getModule());
    registerFunction("qsort_li", intr_pqsort_li);

    Function *intr_pqsort_ll = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ll", getModule());
    registerFunction("qsort_ll", intr_pqsort_ll);


    Function *intr_pqsort_iii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iii", getModule());
    registerFunction("qsort_iii", intr_pqsort_iii);

    Function *intr_pqsort_iil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iil", getModule());
    registerFunction("qsort_iil", intr_pqsort_iil);

    Function *intr_pqsort_ili = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ili", getModule());
    registerFunction("qsort_ili", intr_pqsort_ili);

    Function *intr_pqsort_ill = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ill", getModule());
    registerFunction("qsort_ill", intr_pqsort_ill);

    Function *intr_pqsort_lii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lii", getModule());
    registerFunction("qsort_lii", intr_pqsort_lii);

    Function *intr_pqsort_lil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lil", getModule());
    registerFunction("qsort_lil", intr_pqsort_lil);

    Function *intr_pqsort_lli = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lli", getModule());
    registerFunction("qsort_lli", intr_pqsort_lli);

    Function *intr_pqsort_lll = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lll", getModule());
    registerFunction("qsort_lll", intr_pqsort_lll);


    Function *intr_pqsort_iiii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iiii", getModule());
    registerFunction("qsort_iiii", intr_pqsort_iiii);

    Function *intr_pqsort_iiil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iiil", getModule());
    registerFunction("qsort_iiil", intr_pqsort_iiil);

    Function *intr_pqsort_iili = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iili", getModule());
    registerFunction("qsort_iili", intr_pqsort_iili);

    Function *intr_pqsort_iill = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iill", getModule());
    registerFunction("qsort_iill", intr_pqsort_iill);

    Function *intr_pqsort_ilii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ilii", getModule());
    registerFunction("qsort_ilii", intr_pqsort_ilii);

    Function *intr_pqsort_ilil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_ilil", getModule());
    registerFunction("qsort_ilil", intr_pqsort_ilil);

    Function *intr_pqsort_illi = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_illi", getModule());
    registerFunction("qsort_illi", intr_pqsort_illi);

    Function *intr_pqsort_illl = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_illl", getModule());
    registerFunction("qsort_illl", intr_pqsort_illl);

    Function *intr_pqsort_liii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_liii", getModule());
    registerFunction("qsort_liii", intr_pqsort_liii);

    Function *intr_pqsort_liil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_liil", getModule());
    registerFunction("qsort_liil", intr_pqsort_liil);

    Function *intr_pqsort_lili = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lili", getModule());
    registerFunction("qsort_lili", intr_pqsort_lili);

    Function *intr_pqsort_lill = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_lill", getModule());
    registerFunction("qsort_lill", intr_pqsort_lill);

    Function *intr_pqsort_llii = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_llii", getModule());
    registerFunction("qsort_llii", intr_pqsort_llii);

    Function *intr_pqsort_llil = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_llil", getModule());
    registerFunction("qsort_llil", intr_pqsort_llil);

    Function *intr_pqsort_llli = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_llli", getModule());
    registerFunction("qsort_llli", intr_pqsort_llli);

    Function *intr_pqsort_llll = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_llll", getModule());
    registerFunction("qsort_llll", intr_pqsort_llll);

    Function *intr_pqsort_iillllllll = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_iillllllll", getModule());
    registerFunction("qsort_iillllllll", intr_pqsort_iillllllll);

    Function *intr_pqsort_llllllllll = Function::Create(intrqsort, Function::ExternalLinkage, "qsort_llllllllll", getModule());
    registerFunction("qsort_llllllllll", intr_pqsort_llllllllll);
}

size_t RawGpuPipelineGen::prepareStateArgument(){
    // if (state_vars.empty()) {
    //     LLVMContext &   llvmContext = context->getLLVMContext();
    //     Type        *   int32Type   = Type::getInt32Ty(llvmContext);

    //     appendStateVar(int32Type); //FIMXE: should not be necessary... there should be some way to bypass it...
    // }

    state_type      = StructType::create(state_vars, pipName + "_state_t");
    size_t state_id = appendParameter(state_type, false, false);//true);

    return state_id;
}

Value * RawGpuPipelineGen::getStateVar() const{
    assert(state);
    if (!wrapperModuleActive){
        Function * Fcurrent = getBuilder()->GetInsertBlock()->getParent();
        if (Fcurrent != F){
            return Fcurrent->arg_end() - 1;
        }
    }
    return state; //getArgument(args.size() - 1);
}

Value * RawGpuPipelineGen::getStateLLVMValue(){
    return getArgument(args.size() - 1);
}

Function * RawGpuPipelineGen::prepareConsumeWrapper(){
    IRBuilder<> * Builder       = getBuilder();
    LLVMContext & llvmContext   = context->getLLVMContext();
    BasicBlock  * BB            = Builder->GetInsertBlock();

    // Function      * cons        = Function::Create(
    //                                                 F->getFunctionType(), 
    //                                                 Function::ExternalLinkage, 
    //                                                 pipName + "_wrapper", 
    //                                                 getModule()
    //                                             );
    
    vector<Type  * > inps;
    for (auto &m: F->args()) inps.push_back(m.getType());
    inps.back() = PointerType::getUnqual(inps.back());

    FunctionType *ftype = FunctionType::get(Type::getVoidTy(context->getLLVMContext()), inps, false);
    //use f_num to overcome an llvm bu with keeping dots in function names when generating PTX (which is invalid in PTX)
    Function * cons = Function::Create(ftype, Function::ExternalLinkage, pipName + "_wrapper", context->getModule());

    BasicBlock    * entryBB     = BasicBlock::Create(llvmContext, "entry", cons);
    Builder->SetInsertPoint(entryBB);

    auto            args        = cons->arg_begin();
    vector<Value *> mems    ;
    for (size_t i = 0 ; i < inputs.size() - 1 ; ++i, ++args){ //handle state separately
        mems.push_back(context->CreateEntryBlockAlloca("arg_" + std::to_string(i), inputs[i]));
        Builder->CreateStore(args, mems.back());
    }
    mems.push_back(args);

    Value         * entry   = Builder->CreateExtractValue(Builder->CreateLoad(args), kernel_id);
    Value         * strm    = Builder->CreateExtractValue(Builder->CreateLoad(args), strm_id  );

    vector<Type  * > types;
    for (auto &m: mems) types.push_back(m->getType());


    Type  * struct_type     = StructType::create(types, pipName + "_launch_t");
    Value * p               = UndefValue::get(struct_type);

    for (size_t i = 0 ; i < inputs.size(); ++i){
        p = Builder->CreateInsertValue(p, mems[i], i);
    }
    
    Value  * params = context->CreateEntryBlockAlloca("params_mem", struct_type);
    Builder->CreateStore(p, params);

    Type  * charPtrType     = Type::getInt8PtrTy(llvmContext);
    Type  * ptr_t           = PointerType::get(charPtrType, 0);
    vector<Value *> kernel_args{
                                entry, 
                                Builder->CreateBitCast(params, ptr_t),
                                strm,
                                context->createInt32(maxGridSize ),
                                context->createInt32(maxBlockSize)
                                };

    Function      * launch      = getFunction("launch_kernel_strm_sized");

    Builder->CreateCall(launch, kernel_args);

    Function      * sync_strm   = getFunction("sync_strm");

    Builder->CreateCall(sync_strm, vector<Value *>{strm});
    Builder->CreateRetVoid();

    // FunctionType * sync_type = FunctionType::get(Type::getVoidTy(context->getLLVMContext()), vector<Type  * >{}, false);
    // //use f_num to overcome an llvm bu with keeping dots in function names when generating PTX (which is invalid in PTX)
    // subpipelineSync = Function::Create(sync_type, Function::ExternalLinkage, pipName + "_sync", context->getModule());
    
    // entryBB     = BasicBlock::Create(llvmContext, "entry", subpipelineSync);
    // Builder->SetInsertPoint(entryBB);

    // strm        = Builder->CreateExtractValue(subpipelineSync->arg_end() - 1, strm_id);

    // Function      * sync_strm   = getFunction("sync_strm");

    // Builder->CreateCall(sync_strm, vector<Value *>{strm});
    // Builder->CreateRetVoid();

    Builder->SetInsertPoint(BB);

    Fconsume = cons;
    return cons;
}

void RawGpuPipelineGen::prepareInitDeinit(){
    wrapperModuleActive = true ;

    Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * size_type    = context->createSizeT(0)->getType();
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    FunctionType * FTlaunch_kernel        = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0)}, 
                                                    false
                                                );

    Function * launch_kernel_             = Function::Create(
                                                    FTlaunch_kernel,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel",launch_kernel_);

    FunctionType * FTlaunch_kernel_strm         = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0), charPtrType}, 
                                                    false
                                                );

    Function * launch_kernel_strm_              = Function::Create(
                                                    FTlaunch_kernel_strm,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel_strm", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel_strm",launch_kernel_strm_);

    FunctionType * FTlaunch_kernel_strm_single  = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0), charPtrType}, 
                                                    false
                                                );

    Function * launch_kernel_strm_single_       = Function::Create(
                                                    FTlaunch_kernel_strm_single,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel_strm_single", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel_strm_single",launch_kernel_strm_single_);

    FunctionType * FTlaunch_kernel_strm_sized   = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0), charPtrType, int32_type, int32_type}, 
                                                    false
                                                );
    Function * launch_kernel_strm_sized_        = Function::Create(
                                                    FTlaunch_kernel_strm_sized,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel_strm_sized", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel_strm_sized",launch_kernel_strm_sized_);

    FunctionType *intrgetPipKernel = FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
    Function *intr_pgetPipKernel = Function::Create(intrgetPipKernel, Function::ExternalLinkage, "getPipKernel", getModule());
    registerFunction("getPipKernel", intr_pgetPipKernel);

    FunctionType *intrcreateCudaStream = FunctionType::get(charPtrType, std::vector<Type *>{}, false);
    Function *intr_pcreateCudaStream = Function::Create(intrcreateCudaStream, Function::ExternalLinkage, "createCudaStream", getModule());
    registerFunction("createCudaStream", intr_pcreateCudaStream);

    FunctionType *intrsync_strm = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *intr_psync_strm = Function::Create(intrsync_strm, Function::ExternalLinkage, "sync_strm", getModule());
    registerFunction("sync_strm", intr_psync_strm);

    FunctionType *intrdestroyCudaStream = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *intr_pdestroyCudaStream = Function::Create(intrdestroyCudaStream, Function::ExternalLinkage, "destroyCudaStream", getModule());
    registerFunction("destroyCudaStream", intr_pdestroyCudaStream);

    FunctionType *intrmemset = FunctionType::get(void_type, std::vector<Type *>{charPtrType, int32_type, size_type}, false);
    Function *intr_pmemset = Function::Create(intrmemset, Function::ExternalLinkage, "gpu_memset", getModule());
    registerFunction("memset", intr_pmemset);

    registerSubPipeline();
    registerFunctions();

    Function * tmpF     = F;
    F = prepareConsumeWrapper();

    RawPipelineGen::prepareInitDeinit();

    F = tmpF;
    wrapperModuleActive = false;
}

void RawGpuPipelineGen::markAsKernel(Function * F) const{
    LLVMContext &llvmContext = context->getLLVMContext();

    Type *int32Type           = Type::getInt32Ty(llvmContext);
    
    std::vector<llvm::Metadata *> Vals;

    NamedMDNode * annot = getModule()->getOrInsertNamedMetadata("nvvm.annotations");
    MDString    * str   = MDString::get(llvmContext, "kernel");
    Value       * one   = ConstantInt::get(int32Type, 1);

    Vals.push_back(ValueAsMetadata::get(F));
    Vals.push_back(str);
    Vals.push_back(ValueAsMetadata::getConstant(one));
    
    MDNode * mdNode = MDNode::get(llvmContext, Vals);

    annot->addOperand(mdNode);
}

Function * const RawGpuPipelineGen::createHelperFunction(string funcName, std::vector<Type *> ins, std::vector<bool> readonly, std::vector<bool> noalias) const{
    assert(readonly.size() == noalias.size());
    assert(readonly.size() == 0 || readonly.size() == args.size());

    ins.push_back(state_type);

    FunctionType *ftype = FunctionType::get(Type::getVoidTy(context->getLLVMContext()), ins, false);
    //use f_num to overcome an llvm bu with keeping dots in function names when generating PTX (which is invalid in PTX)
    Function * helper = Function::Create(ftype, Function::ExternalLinkage, funcName, context->getModule());

    if (readonly.size() == ins.size()) {
        Attribute readOnly = Attribute::get(context->getLLVMContext(), Attribute::AttrKind::ReadOnly);
        Attribute noAlias  = Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias );

        readonly.push_back(true);
        noalias .push_back(true);
        std::vector<std::pair<unsigned, Attribute>> attrs;
        for (size_t i = 1 ; i <= ins.size() ; ++i){ //+1 because 0 is the return value
            if (readonly[i - 1]) attrs.emplace_back(i, readOnly);
            if (noalias [i - 1]) attrs.emplace_back(i, noAlias );
        }

        helper->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    }

    BasicBlock *BB = BasicBlock::Create(context->getLLVMContext(), "entry", helper);
    getBuilder()->SetInsertPoint(BB);

    markAsKernel(helper);
    
    return helper;
}

Function * RawGpuPipelineGen::prepare(){
    assert(!F);
    RawPipelineGen::prepare();

    markAsKernel(F);

    return F;
}

void * RawGpuPipelineGen::getConsume() const{
    return wrapper_module.getCompiledFunction(Fconsume);
}

void * RawGpuPipelineGen::getKernel() const{
    return module.getCompiledFunction(F);
}

RawPipeline * RawGpuPipelineGen::getPipeline(int group_id){
    void       * func       = getKernel();

    std::vector<std::pair<const void *, std::function<opener_t>>> openers{this->openers};
    std::vector<std::pair<const void *, std::function<closer_t>>> closers{this->closers};

    if (copyStateFrom){
        RawPipeline * copyFrom = copyStateFrom->getPipeline(group_id);

        openers.insert(openers.begin(), std::make_pair(this, [copyFrom](RawPipeline * pip){copyFrom->open (); pip->setStateVar(0, copyFrom->state);}));
        // closers.emplace_back([copyFrom](RawPipeline * pip){pip->copyStateBackTo(copyFrom);});
        closers.insert(closers.begin(), std::make_pair(this, [copyFrom](RawPipeline * pip){copyFrom->close();                                      }));
    } else {
        openers.insert(openers.begin(), std::make_pair(this, [        ](RawPipeline * pip){                                                        }));
        // closers.emplace_back([copyFrom](RawPipeline * pip){pip->copyStateBackTo(copyFrom);});
        closers.insert(closers.begin(), std::make_pair(this, [        ](RawPipeline * pip){                                                        }));
    }
    
    return new RawPipeline(func, getModule()->getDataLayout().getTypeAllocSize(state_type), this, state_type, openers, closers, wrapper_module.getCompiledFunction(open__function), wrapper_module.getCompiledFunction(close_function), group_id, execute_after_close ? execute_after_close->getPipeline(group_id) : NULL);
}

void * RawGpuPipelineGen::getCompiledFunction(Function * f){
    //FIXME: should handle cpu functins (open/close)
    if (wrapperModuleActive) return wrapper_module.getCompiledFunction(f);
    return module.getCompiledFunction(f);
}

void RawGpuPipelineGen::compileAndLoad(){
    wrapper_module.compileAndLoad();
    module.compileAndLoad();
    func = getCompiledFunction(F);
}

void RawGpuPipelineGen::registerFunction(const char * funcName, Function * f){
    if (wrapperModuleActive) {
        availableWrapperFunctions[funcName] = f;
    } else {
        RawPipelineGen::registerFunction(funcName, f);
    }
}

Function * const RawGpuPipelineGen::getFunction(string funcName) const{
    if (wrapperModuleActive) {
        map<string, Function*>::const_iterator it;
        it = availableWrapperFunctions.find(funcName);
        if (it == availableWrapperFunctions.end()) {
            for (auto &t: availableWrapperFunctions) std::cout << t.first << std::endl;
            throw runtime_error(string("Unknown function name: ") + funcName + " (" + pipName + ")");
        }
        return it->second;
    }
    return RawPipelineGen::getFunction(funcName);
}


extern "C"{
    void *          getPipKernel(RawPipelineGen * pip){
        return pip->getKernel();
    };

    cudaStream_t    createCudaStream(){
#ifndef NCUDA
        cudaStream_t strm;
        gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
        return strm;
#else
        assert(false);
        return NULL;
#endif
    }

    void            sync_strm(cudaStream_t strm){
        gpu_run(cudaStreamSynchronize(strm));
    }

    void            destroyCudaStream(cudaStream_t strm){
        gpu_run(cudaStreamSynchronize(strm));
        gpu_run(cudaStreamDestroy    (strm));
    }
}