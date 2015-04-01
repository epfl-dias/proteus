/*
 * atois.hpp
 *
 *  Created on: Apr 1, 2015
 *      Author: manolee
 */

#include "util/atois.hpp"

void atois(Value *buf, Value* len, AllocaInst *mem_result, RawContext* const context)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();

	/* Preparing switch building blocks */
	BasicBlock *defaultBlock;
	BasicBlock *returnBlock;
	BasicBlock *blockLen1, *blockLen2, *blockLen3, *blockLen4, *blockLen5,
			*blockLen6, *blockLen7, *blockLen8, *blockLen9, *blockLen10;
	blockLen1 	 = BasicBlock::Create(llvmContext, "blockLen1", F);
	blockLen2 	 = BasicBlock::Create(llvmContext, "blockLen2", F);
	blockLen3 	 = BasicBlock::Create(llvmContext, "blockLen3", F);
	blockLen4 	 = BasicBlock::Create(llvmContext, "blockLen4", F);
	blockLen5 	 = BasicBlock::Create(llvmContext, "blockLen5", F);
	blockLen6 	 = BasicBlock::Create(llvmContext, "blockLen6", F);
	blockLen7 	 = BasicBlock::Create(llvmContext, "blockLen7", F);
	blockLen8 	 = BasicBlock::Create(llvmContext, "blockLen8", F);
	blockLen9 	 = BasicBlock::Create(llvmContext, "blockLen9", F);
	blockLen10 	 = BasicBlock::Create(llvmContext, "blockLen10", F);
	defaultBlock = BasicBlock::Create(llvmContext, "defaultAtoiSwitch", F);
	defaultBlock = BasicBlock::Create(llvmContext, "returnAtoiSwitch", F);

	ConstantInt *val_1  = context->createInt32(1);
	ConstantInt *val_2  = context->createInt32(1);
	ConstantInt *val_3  = context->createInt32(1);
	ConstantInt *val_4  = context->createInt32(1);
	ConstantInt *val_5  = context->createInt32(1);
	ConstantInt *val_6  = context->createInt32(1);
	ConstantInt *val_7  = context->createInt32(1);
	ConstantInt *val_8  = context->createInt32(1);
	ConstantInt *val_9  = context->createInt32(1);
	ConstantInt *val_10 = context->createInt32(1);

	SwitchInst *atoiSwitch = Builder->CreateSwitch(len, defaultBlock,10);
	atoiSwitch->addCase(val_1,blockLen1);
	atoiSwitch->addCase(val_2,blockLen2);
	atoiSwitch->addCase(val_3,blockLen3);
	atoiSwitch->addCase(val_4,blockLen4);
	atoiSwitch->addCase(val_5,blockLen5);
	atoiSwitch->addCase(val_6,blockLen6);
	atoiSwitch->addCase(val_7,blockLen7);
	atoiSwitch->addCase(val_8,blockLen8);
	atoiSwitch->addCase(val_9,blockLen9);
	atoiSwitch->addCase(val_10,blockLen10);

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
}

void atoi1(Value *buf, AllocaInst *mem_result, RawContext* const context) {
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int8Type = Type::getInt8Ty(llvmContext);
	Type* int1Type = Type::getInt1Ty(llvmContext);
	llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();

	// ConstantInt *char_0 = ?
	ConstantInt *val_0  = context->createInt32(0);
	context->getArrayElem(buf,val_0);
}



