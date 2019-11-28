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

#include "util/gpu/gpu-intrinsics.hpp"

#include "common/gpu/gpu-common.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/InlineAsm.h"

namespace gpu_intrinsic {

llvm::Value *activemask(ParallelContext *const context) {
  auto Builder = context->getBuilder();
  auto mask_sig = llvm::FunctionType::get(Builder->getInt32Ty(),
                                          std::vector<llvm::Type *>{}, false);

  auto mask_fun =
      llvm::InlineAsm::get(mask_sig, "activemask.b32 $0;", "=r", true);

  return Builder->CreateCall(mask_fun, std::vector<llvm::Value *>{}, "mask");
}

llvm::Value *load_ca(ParallelContext *const context, llvm::Value *address) {
  auto Builder = context->getBuilder();

  auto lca_sig = llvm::FunctionType::get(
      address->getType()->getPointerElementType(),
      std::vector<llvm::Type *>{address->getType()}, false);

  auto lca_fun =
      llvm::InlineAsm::get(lca_sig, "ld.ca.u32 $0, $1;", "=r,m", true);

  llvm::Value *lval = Builder->CreateCall(lca_fun, {address}, "ld.ca.u32");

  return lval;
}

llvm::Value *load_ca16(ParallelContext *const context, llvm::Value *address) {
  auto Builder = context->getBuilder();

  auto lca_sig = llvm::FunctionType::get(
      address->getType()->getPointerElementType(),
      std::vector<llvm::Type *>{address->getType()}, false);

  auto lca_fun =
      llvm::InlineAsm::get(lca_sig, "ld.ca.u16 $0, $1;", "=r,m", true);

  llvm::Value *lval = Builder->CreateCall(lca_fun, {address}, "ld.ca.u16");

  return lval;
}

llvm::Value *load_cs(ParallelContext *const context, llvm::Value *address) {
  auto Builder = context->getBuilder();

  auto lcs_sig = llvm::FunctionType::get(
      address->getType()->getPointerElementType(),
      std::vector<llvm::Type *>{address->getType()}, false);

  auto lcs_fun =
      llvm::InlineAsm::get(lcs_sig, "ld.cg.u32 $0, $1;", "=r,m", true);

  llvm::Value *lval = Builder->CreateCall(
      lcs_fun, std::vector<llvm::Value *>{address}, "ld.cg.u32");

  return lval;
}

void store_wb32(ParallelContext *const context, llvm::Value *address,
                llvm::Value *value) {
  auto &ctx = context->getLLVMContext();
  auto Builder = context->getBuilder();

  auto void_type = llvm::Type::getVoidTy(ctx);

  auto stw_sig = llvm::FunctionType::get(
      void_type,
      std::vector<llvm::Type *>{address->getType(),
                                address->getType()->getPointerElementType()},
      false);

  auto stw_fun =
      llvm::InlineAsm::get(stw_sig, "st.wb.u32 $0, $1;", "m,r", true);

  Builder->CreateCall(stw_fun, std::vector<llvm::Value *>{address, value});

  return;
}

void store_wb16(ParallelContext *const context, llvm::Value *address,
                llvm::Value *value) {
  auto &ctx = context->getLLVMContext();
  auto Builder = context->getBuilder();

  llvm::Type *void_type = llvm::Type::getVoidTy(ctx);

  auto stw_sig = llvm::FunctionType::get(
      void_type,
      std::vector<llvm::Type *>{address->getType(),
                                address->getType()->getPointerElementType()},
      false);

  auto stw_fun =
      llvm::InlineAsm::get(stw_sig, "st.wb.u16 $0, $1;", "m,r", true);

  Builder->CreateCall(stw_fun, std::vector<llvm::Value *>{address, value});

  return;
}

[[deprecated]] llvm::Value *all(ParallelContext *const context,
                                llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  auto all_fun = context->getFunction("llvm.nvvm.vote.all.sync");
  llvm::Value *all = Builder->CreateCall(
      all_fun, std::vector<llvm::Value *>{activemask(context), val_in}, "all");

  return all;
}

[[deprecated]] llvm::Value *any(ParallelContext *const context,
                                llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  auto any_fun = context->getFunction("llvm.nvvm.vote.any.sync");

  llvm::Value *any = Builder->CreateCall(
      any_fun, std::vector<llvm::Value *>{activemask(context), val_in}, "any");

  return any;
}

[[deprecated]] llvm::Value *ballot(ParallelContext *const context,
                                   llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  auto ballot_fun = context->getFunction("llvm.nvvm.vote.ballot.sync");

  llvm::Value *ballot = Builder->CreateCall(
      ballot_fun, std::vector<llvm::Value *>{activemask(context), val_in},
      "ballot");

  return ballot;
}

llvm::Value *shfl_bfly(ParallelContext *const context, llvm::Value *val_in,
                       uint32_t vxor, llvm::Value *mask) {
  return shfl_bfly(context, val_in, context->createInt32(vxor), mask);
}

llvm::Value *shfl_bfly(ParallelContext *const context, llvm::Value *val_in,
                       llvm::Value *vxor, llvm::Value *mask) {
  auto Builder = context->getBuilder();
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = llvm::Type::getInt32Ty(llvmContext);

  // Aggregate internally to each warp
  auto shfl_bfly = context->getFunction("llvm.nvvm.shfl.sync.bfly.i32");
  assert(shfl_bfly);

  unsigned int bits = 0;

  if (!mask) mask = context->createInt32(warp_size - 1);

  llvm::Type *initial_type = val_in->getType();

  if (!val_in->getType()->isIntegerTy()) {
    if (val_in->getType()->isFloatingPointTy()) {
      const llvm::fltSemantics &flt = val_in->getType()->getFltSemantics();

      bits = llvm::APFloat::semanticsSizeInBits(flt);
    } else {  // TODO: do something for vectors as well...
      string error_msg =
          string("[GpuIntrinsics: ] Still unsupported argument type for shfl");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  } else {
    bits = val_in->getType()->getIntegerBitWidth();
  }

  unsigned int elems = ((bits + 31) / 32);

  llvm::Type *inttype = llvm::IntegerType::get(llvmContext, 32 * elems);
  if (bits % 32) val_in = Builder->CreateZExtOrBitCast(val_in, inttype);

  llvm::Type *packtype = llvm::VectorType::get(int32_type, elems);

  llvm::Type *intcast = llvm::VectorType::get(val_in->getType(), 1);

  val_in = Builder->CreateZExtOrBitCast(val_in, intcast, "interm");

  llvm::Value *val_shfl = Builder->CreateBitCast(val_in, packtype, "pack");

  std::vector<llvm::Value *> ArgsV;
  // For whatever reason, LLVM has the membermask first,
  // instead of last, as in PTX...
  ArgsV.push_back(context->createInt32(~0));
  ArgsV.push_back(nullptr);
  ArgsV.push_back(vxor);
  ArgsV.push_back(mask);

  llvm::Value *val_out = llvm::UndefValue::get(
      packtype);  // ConstantVector::getSplat(elems, context->createInt32(0));

  for (unsigned int i = 0; i < elems; ++i) {
    // ArgsV[0]    = Builder->CreateExtractElement(val_shfl, i);
    ArgsV[1] = Builder->CreateExtractElement(val_shfl, i);

    llvm::Value *tmp =
        Builder->CreateCall(shfl_bfly, ArgsV, "shfl_res_" + std::to_string(i));

    val_out = Builder->CreateInsertElement(val_out, tmp, i);
  }

  val_out = Builder->CreateBitCast(val_out, intcast);

  if (bits % 32) val_out = Builder->CreateTruncOrBitCast(val_out, inttype);

  return Builder->CreateTruncOrBitCast(val_out, initial_type);
}

}  // namespace gpu_intrinsic
