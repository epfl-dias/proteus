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

#include "common/gpu/gpu-common.hpp"
#include "util/gpu/gpu-intrinsics.hpp"
#include "llvm/IR/InlineAsm.h"
#include "llvm/ADT/APFloat.h"

namespace gpu_intrinsic{

Value * all(GpuRawContext * const context, Value * val_in){
    IRBuilder<>* Builder        = context->getBuilder();

    FunctionType    * all_sig   = FunctionType::get(val_in->getType(), 
                                        std::vector<Type *>{val_in->getType()},
                                        false);

    InlineAsm       * all_fun   = InlineAsm::get(all_sig, 
                                        "vote.all.pred $0, $1;",
                                        "=b,b",
                                        false);

    Value           * all       = Builder->CreateCall(all_fun,
                                        std::vector<Value *>{val_in},
                                        "all");

    return all;
}

Value * any(GpuRawContext * const context, Value * val_in){
    IRBuilder<>* Builder        = context->getBuilder();
    
    FunctionType    * any_sig   = FunctionType::get(val_in->getType(), 
                                        std::vector<Type *>{val_in->getType()},
                                        false);

    InlineAsm       * any_fun   = InlineAsm::get(any_sig, 
                                        "vote.any.pred $0, $1;",
                                        "=b,b",
                                        false);

    Value           * any       = Builder->CreateCall(  any_fun,
                                        std::vector<Value *>{val_in},
                                        "any");

    return any;
}

Value * ballot(GpuRawContext * const context, Value * val_in){
    IRBuilder<>* Builder            = context->getBuilder();
    LLVMContext &llvmContext        = context->getLLVMContext();
    IntegerType *int32_type         = Type::getInt32Ty(llvmContext);
    
    FunctionType    * ballot_sig    = FunctionType::get(int32_type, 
                                        std::vector<Type *>{val_in->getType()},
                                        false);

    InlineAsm       * ballot_fun    = InlineAsm::get(ballot_sig, 
                                        "vote.ballot.b32 $0, $1;",
                                        "=r,b",
                                        false);

    Value           * ballot        = Builder->CreateCall(  ballot_fun,
                                        std::vector<Value *>{val_in},
                                        "ballot");

    return ballot;
}

Value * shfl_bfly(GpuRawContext * const context, 
                    Value *             val_in, 
                    uint32_t            vxor, 
                    Value *             mask){
    return shfl_bfly(context, val_in, context->createInt32(vxor), mask);
}

Value * shfl_bfly(GpuRawContext * const context, 
                    Value *             val_in, 
                    Value *             vxor, 
                    Value *             mask){
    
    IRBuilder<> *Builder        = context->getBuilder();
    Module      *mod            = context->getModule();
    LLVMContext &llvmContext    = context->getLLVMContext();
    IntegerType *int32_type     = Type::getInt32Ty(llvmContext);
    IntegerType *int1_type      = Type::getInt1Ty (llvmContext);

    // Aggregate internally to each warp
    Function *shfl_bfly = context->getFunction("llvm.nvvm.shfl.bfly.i32");
    assert(shfl_bfly);
    
    unsigned int bits = 0;

    if (!mask) mask = context->createInt32(warp_size - 1);

    Type * initial_type = val_in->getType();

    if (!val_in->getType()->isIntegerTy()){
        if (val_in->getType()->isFloatingPointTy()){
            const fltSemantics & flt = val_in->getType()->getFltSemantics();
            
            bits = APFloat::semanticsSizeInBits(flt);
        } else { //TODO: do something for vectors as well...
            string error_msg = string(
                "[GpuIntrinsics: ] Still unsupported argument type for shfl");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
        }
    } else {
        bits = val_in->getType()->getIntegerBitWidth();
    }

    unsigned int elems = ((bits + 31) / 32);

    Type  * inttype  = IntegerType::get(llvmContext, 32 * elems);
    if (bits % 32) val_in = Builder->CreateZExtOrBitCast(val_in, inttype);

    Type  * packtype = VectorType::get(int32_type, elems);

    Type  * intcast  = VectorType::get(val_in->getType(), 1);

    val_in           = Builder->CreateZExtOrBitCast(val_in, intcast, "interm");

    Value * val_shfl = Builder->CreateBitCast      (val_in, packtype, "pack");

    std::vector<Value *> ArgsV;
    ArgsV.push_back(NULL);
    ArgsV.push_back(vxor);
    ArgsV.push_back(mask);
    
    Value * val_out = UndefValue::get(packtype);//ConstantVector::getSplat(elems, context->createInt32(0));

    for (unsigned int i = 0 ; i < elems ; ++i){
        ArgsV[0]    = Builder->CreateExtractElement(val_shfl, i);

        Value * tmp = Builder->CreateCall(shfl_bfly, ArgsV, 
                                            "shfl_res_" + std::to_string(i));

        val_out     = Builder->CreateInsertElement(val_out, tmp, i);
    }

    val_out         = Builder->CreateBitCast(val_out, intcast);

    if (bits % 32) val_out = Builder->CreateTruncOrBitCast(val_out, inttype);

    return Builder->CreateTruncOrBitCast(val_out, initial_type);
}

}