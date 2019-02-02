/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

/*
 * atois.hpp
 *
 *  Created on: Apr 1, 2015
 *      Author: manolee
 */

#include "util/atois.hpp"
#include "parallel-context.hpp"

using namespace llvm;

/* (buf[0] - '0') */
void atoi1(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  ConstantInt *val_0 = context->createInt32(0);

  Value *buf_0 = context->getArrayElem(buf, val_0);

  Value *val_res = Builder->CreateSub(buf_0, char_0);
  val_res = Builder->CreateSExt(val_res, int32Type);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/* ((buf[0] - '0') * 10) + \
    (buf[1] - '0'); */
void atoi2(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_10, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 100) + \
    ((buf[1] - '0') * 10) + \
    (buf[2] - '0'); */
void atoi3(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_100, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_10, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 1000) + \
    ((buf[1] - '0') * 100) + \
    ((buf[2] - '0') * 10) + \
    (buf[3] - '0')*/
void atoi4(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_1000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_100, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_10, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 10000) + \
    ((buf[1] - '0') * 1000) + \
    ((buf[2] - '0') * 100) + \
    ((buf[3] - '0') * 10) + \
    (buf[4] - '0')*/
void atoi5(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_10000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_1000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_100, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_10, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 100000) + \
    ((buf[1] - '0') * 10000) + \
    ((buf[2] - '0') * 1000) + \
    ((buf[3] - '0') * 100) + \
    ((buf[4] - '0') * 10) + \
    (buf[5] - '0')*/
void atoi6(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);
  ConstantInt *val_100000 = context->createInt32(100000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);
  Value *buf_5 = context->getArrayElem(buf, val_5);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_100000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_10000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_1000, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_100, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);
  val_res4 = Builder->CreateMul(val_10, val_res4);

  Value *val_res5 = Builder->CreateSub(buf_5, char_0);
  val_res5 = Builder->CreateSExt(val_res5, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
  val_res = Builder->CreateAdd(val_res, val_res5);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 1000000) + \
    ((buf[1] - '0') * 100000) + \
    ((buf[2] - '0') * 10000) + \
    ((buf[3] - '0') * 1000) + \
    ((buf[4] - '0') * 100) + \
    ((buf[5] - '0') * 10) + \
    (buf[6] - '0')*/
void atoi7(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);
  ConstantInt *val_6 = context->createInt32(6);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);
  ConstantInt *val_100000 = context->createInt32(100000);
  ConstantInt *val_1000000 = context->createInt32(1000000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);
  Value *buf_5 = context->getArrayElem(buf, val_5);
  Value *buf_6 = context->getArrayElem(buf, val_6);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_1000000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_100000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_10000, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_1000, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);
  val_res4 = Builder->CreateMul(val_100, val_res4);

  Value *val_res5 = Builder->CreateSub(buf_5, char_0);
  val_res5 = Builder->CreateSExt(val_res5, int32Type);
  val_res5 = Builder->CreateMul(val_10, val_res5);

  Value *val_res6 = Builder->CreateSub(buf_6, char_0);
  val_res6 = Builder->CreateSExt(val_res6, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
  val_res = Builder->CreateAdd(val_res, val_res5);
  val_res = Builder->CreateAdd(val_res, val_res6);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 10000000) + \
    ((buf[1] - '0') * 1000000) + \
    ((buf[2] - '0') * 100000) + \
    ((buf[3] - '0') * 10000) + \
    ((buf[4] - '0') * 1000) + \
    ((buf[5] - '0') * 100) + \
    ((buf[6] - '0') * 10) + \
    (buf[7] - '0')*/
void atoi8(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);
  ConstantInt *val_6 = context->createInt32(6);
  ConstantInt *val_7 = context->createInt32(7);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);
  ConstantInt *val_100000 = context->createInt32(100000);
  ConstantInt *val_1000000 = context->createInt32(1000000);
  ConstantInt *val_10000000 = context->createInt32(10000000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);
  Value *buf_5 = context->getArrayElem(buf, val_5);
  Value *buf_6 = context->getArrayElem(buf, val_6);
  Value *buf_7 = context->getArrayElem(buf, val_7);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_10000000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_1000000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_100000, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_10000, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);
  val_res4 = Builder->CreateMul(val_1000, val_res4);

  Value *val_res5 = Builder->CreateSub(buf_5, char_0);
  val_res5 = Builder->CreateSExt(val_res5, int32Type);
  val_res5 = Builder->CreateMul(val_100, val_res5);

  Value *val_res6 = Builder->CreateSub(buf_6, char_0);
  val_res6 = Builder->CreateSExt(val_res6, int32Type);
  val_res6 = Builder->CreateMul(val_10, val_res6);

  Value *val_res7 = Builder->CreateSub(buf_7, char_0);
  val_res7 = Builder->CreateSExt(val_res7, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
  val_res = Builder->CreateAdd(val_res, val_res5);
  val_res = Builder->CreateAdd(val_res, val_res6);
  val_res = Builder->CreateAdd(val_res, val_res7);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 100000000) + \
    ((buf[1] - '0') * 10000000) + \
    ((buf[2] - '0') * 1000000) + \
    ((buf[3] - '0') * 100000) + \
    ((buf[4] - '0') * 10000) + \
    ((buf[5] - '0') * 1000) + \
    ((buf[6] - '0') * 100) + \
    ((buf[7] - '0') * 10) + \
    (buf[8] - '0');*/
void atoi9(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);
  ConstantInt *val_6 = context->createInt32(6);
  ConstantInt *val_7 = context->createInt32(7);
  ConstantInt *val_8 = context->createInt32(8);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);
  ConstantInt *val_100000 = context->createInt32(100000);
  ConstantInt *val_1000000 = context->createInt32(1000000);
  ConstantInt *val_10000000 = context->createInt32(10000000);
  ConstantInt *val_100000000 = context->createInt32(100000000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);
  Value *buf_5 = context->getArrayElem(buf, val_5);
  Value *buf_6 = context->getArrayElem(buf, val_6);
  Value *buf_7 = context->getArrayElem(buf, val_7);
  Value *buf_8 = context->getArrayElem(buf, val_8);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_100000000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_10000000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_1000000, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_100000, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);
  val_res4 = Builder->CreateMul(val_10000, val_res4);

  Value *val_res5 = Builder->CreateSub(buf_5, char_0);
  val_res5 = Builder->CreateSExt(val_res5, int32Type);
  val_res5 = Builder->CreateMul(val_1000, val_res5);

  Value *val_res6 = Builder->CreateSub(buf_6, char_0);
  val_res6 = Builder->CreateSExt(val_res6, int32Type);
  val_res6 = Builder->CreateMul(val_100, val_res6);

  Value *val_res7 = Builder->CreateSub(buf_7, char_0);
  val_res7 = Builder->CreateSExt(val_res7, int32Type);
  val_res7 = Builder->CreateMul(val_10, val_res7);

  Value *val_res8 = Builder->CreateSub(buf_8, char_0);
  val_res8 = Builder->CreateSExt(val_res8, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
  val_res = Builder->CreateAdd(val_res, val_res5);
  val_res = Builder->CreateAdd(val_res, val_res6);
  val_res = Builder->CreateAdd(val_res, val_res7);
  val_res = Builder->CreateAdd(val_res, val_res8);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

/*    ((buf[0] - '0') * 1000000000) + \
    ((buf[1] - '0') * 100000000) + \
    ((buf[2] - '0') * 10000000) + \
    ((buf[3] - '0') * 1000000) + \
    ((buf[4] - '0') * 100000) + \
    ((buf[5] - '0') * 10000) + \
    ((buf[6] - '0') * 1000) + \
    ((buf[7] - '0') * 100) + \
    ((buf[8] - '0') * 10) + \
    (buf[9] - '0')*/
void atoi10(Value *buf, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  ConstantInt *char_0 =
      ConstantInt::get(llvmContext, APInt(8, StringRef("48"), 10));
  /* Indexes */
  ConstantInt *val_0 = context->createInt32(0);
  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);
  ConstantInt *val_6 = context->createInt32(6);
  ConstantInt *val_7 = context->createInt32(7);
  ConstantInt *val_8 = context->createInt32(8);
  ConstantInt *val_9 = context->createInt32(9);

  /* Multipliers */
  ConstantInt *val_10 = context->createInt32(10);
  ConstantInt *val_100 = context->createInt32(100);
  ConstantInt *val_1000 = context->createInt32(1000);
  ConstantInt *val_10000 = context->createInt32(10000);
  ConstantInt *val_100000 = context->createInt32(100000);
  ConstantInt *val_1000000 = context->createInt32(1000000);
  ConstantInt *val_10000000 = context->createInt32(10000000);
  ConstantInt *val_100000000 = context->createInt32(100000000);
  ConstantInt *val_1000000000 = context->createInt32(1000000000);

  Value *buf_0 = context->getArrayElem(buf, val_0);
  Value *buf_1 = context->getArrayElem(buf, val_1);
  Value *buf_2 = context->getArrayElem(buf, val_2);
  Value *buf_3 = context->getArrayElem(buf, val_3);
  Value *buf_4 = context->getArrayElem(buf, val_4);
  Value *buf_5 = context->getArrayElem(buf, val_5);
  Value *buf_6 = context->getArrayElem(buf, val_6);
  Value *buf_7 = context->getArrayElem(buf, val_7);
  Value *buf_8 = context->getArrayElem(buf, val_8);
  Value *buf_9 = context->getArrayElem(buf, val_9);

  /* Partial Results */
  Value *val_res0 = Builder->CreateSub(buf_0, char_0);
  val_res0 = Builder->CreateSExt(val_res0, int32Type);
  val_res0 = Builder->CreateMul(val_1000000000, val_res0);

  Value *val_res1 = Builder->CreateSub(buf_1, char_0);
  val_res1 = Builder->CreateSExt(val_res1, int32Type);
  val_res1 = Builder->CreateMul(val_100000000, val_res1);

  Value *val_res2 = Builder->CreateSub(buf_2, char_0);
  val_res2 = Builder->CreateSExt(val_res2, int32Type);
  val_res2 = Builder->CreateMul(val_10000000, val_res2);

  Value *val_res3 = Builder->CreateSub(buf_3, char_0);
  val_res3 = Builder->CreateSExt(val_res3, int32Type);
  val_res3 = Builder->CreateMul(val_1000000, val_res3);

  Value *val_res4 = Builder->CreateSub(buf_4, char_0);
  val_res4 = Builder->CreateSExt(val_res4, int32Type);
  val_res4 = Builder->CreateMul(val_100000, val_res4);

  Value *val_res5 = Builder->CreateSub(buf_5, char_0);
  val_res5 = Builder->CreateSExt(val_res5, int32Type);
  val_res5 = Builder->CreateMul(val_10000, val_res5);

  Value *val_res6 = Builder->CreateSub(buf_6, char_0);
  val_res6 = Builder->CreateSExt(val_res6, int32Type);
  val_res6 = Builder->CreateMul(val_1000, val_res6);

  Value *val_res7 = Builder->CreateSub(buf_7, char_0);
  val_res7 = Builder->CreateSExt(val_res7, int32Type);
  val_res7 = Builder->CreateMul(val_100, val_res7);

  Value *val_res8 = Builder->CreateSub(buf_8, char_0);
  val_res8 = Builder->CreateSExt(val_res8, int32Type);
  val_res8 = Builder->CreateMul(val_10, val_res8);

  Value *val_res9 = Builder->CreateSub(buf_9, char_0);
  val_res9 = Builder->CreateSExt(val_res9, int32Type);

  /* Final Result */
  Value *val_res = Builder->CreateAdd(val_res0, val_res1);
  val_res = Builder->CreateAdd(val_res, val_res2);
  val_res = Builder->CreateAdd(val_res, val_res3);
  val_res = Builder->CreateAdd(val_res, val_res4);
  val_res = Builder->CreateAdd(val_res, val_res5);
  val_res = Builder->CreateAdd(val_res, val_res6);
  val_res = Builder->CreateAdd(val_res, val_res7);
  val_res = Builder->CreateAdd(val_res, val_res8);
  val_res = Builder->CreateAdd(val_res, val_res9);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    ArgsV.push_back(val_res);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  Builder->CreateStore(val_res, mem_result);
}

void atois(Value *buf, Value *len, AllocaInst *mem_result, Context *context) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();

  /* Preparing switch building blocks */
  BasicBlock *defaultBlock;
  BasicBlock *returnBlock;
  BasicBlock *blockLen1, *blockLen2, *blockLen3, *blockLen4, *blockLen5,
      *blockLen6, *blockLen7, *blockLen8, *blockLen9, *blockLen10;
  blockLen1 = BasicBlock::Create(llvmContext, "blockLen1", F);
  blockLen2 = BasicBlock::Create(llvmContext, "blockLen2", F);
  blockLen3 = BasicBlock::Create(llvmContext, "blockLen3", F);
  blockLen4 = BasicBlock::Create(llvmContext, "blockLen4", F);
  blockLen5 = BasicBlock::Create(llvmContext, "blockLen5", F);
  blockLen6 = BasicBlock::Create(llvmContext, "blockLen6", F);
  blockLen7 = BasicBlock::Create(llvmContext, "blockLen7", F);
  blockLen8 = BasicBlock::Create(llvmContext, "blockLen8", F);
  blockLen9 = BasicBlock::Create(llvmContext, "blockLen9", F);
  blockLen10 = BasicBlock::Create(llvmContext, "blockLen10", F);
  defaultBlock = BasicBlock::Create(llvmContext, "defaultAtoiSwitch", F);
  returnBlock = BasicBlock::Create(llvmContext, "returnAtoiSwitch", F);

  ConstantInt *val_1 = context->createInt32(1);
  ConstantInt *val_2 = context->createInt32(2);
  ConstantInt *val_3 = context->createInt32(3);
  ConstantInt *val_4 = context->createInt32(4);
  ConstantInt *val_5 = context->createInt32(5);
  ConstantInt *val_6 = context->createInt32(6);
  ConstantInt *val_7 = context->createInt32(7);
  ConstantInt *val_8 = context->createInt32(8);
  ConstantInt *val_9 = context->createInt32(9);
  ConstantInt *val_10 = context->createInt32(10);

  SwitchInst *atoiSwitch = Builder->CreateSwitch(len, defaultBlock, 10);
  atoiSwitch->addCase(val_1, blockLen1);
  atoiSwitch->addCase(val_2, blockLen2);
  atoiSwitch->addCase(val_3, blockLen3);
  atoiSwitch->addCase(val_4, blockLen4);
  atoiSwitch->addCase(val_5, blockLen5);
  atoiSwitch->addCase(val_6, blockLen6);
  atoiSwitch->addCase(val_7, blockLen7);
  atoiSwitch->addCase(val_8, blockLen8);
  atoiSwitch->addCase(val_9, blockLen9);
  atoiSwitch->addCase(val_10, blockLen10);

  Builder->SetInsertPoint(blockLen1);
  atoi1(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen2);
  atoi2(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen3);
  atoi3(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen4);
  atoi4(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen5);
  atoi5(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen6);
  atoi6(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen7);
  atoi7(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen8);
  atoi8(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen9);
  atoi9(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  Builder->SetInsertPoint(blockLen10);
  atoi10(buf, mem_result, context);
  Builder->CreateBr(returnBlock);

  /* Handle 'error' case */
  Builder->SetInsertPoint(defaultBlock);
#ifdef DEBUGATOIS
  {
    vector<Value *> ArgsV;
    Function *debugInt = context->getFunction("printi");
    Value *val_tmp = Builder->getInt32(-111);
    ArgsV.push_back(val_tmp);
    Builder->CreateCall(debugInt, ArgsV);
    ArgsV.clear();
    ArgsV.push_back(len);
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  if (dynamic_cast<ParallelContext *>(context)) {
    Builder->CreateCall(
        Intrinsic::getDeclaration(context->getModule(), Intrinsic::trap));
    Builder->CreateUnreachable();
  } else {
    Value *val_error = Builder->getInt32(-1);
    Builder->CreateRet(val_error);
  }
  /* Back to normal */
  Builder->SetInsertPoint(returnBlock);
}
