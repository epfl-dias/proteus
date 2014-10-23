/*
	RAW -- High-performance querying over raw, never-seen-before data.

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

#include "plugins/csv-plugin.hpp"

CSVPlugin::CSVPlugin(RawContext* const context, string& fname, RecordType& rec, vector<RecordAttribute*>& whichFields)
	: fname(fname), rec(rec), wantedFields(whichFields), context(context), posVar("offset"), bufVar("buf"), fsizeVar("fileSize") {

	pos = 0;
	fd = -1;
	buf = NULL;

	LOG(INFO) << "[CSVPlugin: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;

	fd = open(name_c, O_RDONLY);
	if (fd == -1) {
		throw runtime_error(string("csv.open"));
	}
}

CSVPlugin::~CSVPlugin() {}

void CSVPlugin::init()	{

	buf = (char*) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
	if (buf == MAP_FAILED) {
		throw runtime_error(string("csv.mmap"));
	}

	//Preparing the codegen part

	//(Can probably wrap some of these calls in one function)
	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Allocating memory
	AllocaInst *offsetMem = context->CreateEntryBlockAlloca(F,std::string(posVar),Type::getInt64Ty(llvmContext));
	AllocaInst *bufMem = context->CreateEntryBlockAlloca(F,std::string(bufVar),charPtrType);
	AllocaInst *fsizeMem = context->CreateEntryBlockAlloca(F,std::string(fsizeVar),Type::getInt64Ty(llvmContext));
	Value* offsetVal = Builder->getInt64(0);
	Builder->CreateStore(offsetVal,offsetMem);
	NamedValuesCSV[posVar] = offsetMem;

	Value* fsizeVal = Builder->getInt64(fsize);
	Builder->CreateStore(fsizeVal,fsizeMem);
	NamedValuesCSV[fsizeVar] = fsizeMem;

	//Typical way to pass a pointer via the LLVM API
	AllocaInst *AllocaPtr = context->CreateEntryBlockAlloca(F,std::string("charPtr"),charPtrType);
	Value* ptrVal = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* unshiftedPtr = Builder->CreateIntToPtr(ptrVal,charPtrType);
	Builder->CreateStore(unshiftedPtr,bufMem);
	NamedValuesCSV[bufVar] = bufMem;

};

void CSVPlugin::generate(const RawOperator &producer) {
	return scanCSV(producer, context->getGlobalFunction());
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
AllocaInst* CSVPlugin::readPath(Bindings bindings, const char* pathVar)	{
	AllocaInst* mem_projection;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, AllocaInst*>& csvProjections = state->getBindings();
		RecordAttribute tmpKey = RecordAttribute(fname,pathVar);
		map<RecordAttribute, AllocaInst*>::const_iterator it;
		it = csvProjections.find(tmpKey);
			if (it == csvProjections.end()) {
				string error_msg = string("[CSV plugin - readPath ]: Unknown variable name ")+pathVar;
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		mem_projection = it->second;
	}
	return mem_projection;
}

AllocaInst* CSVPlugin::readValue(AllocaInst* mem_value, const ExpressionType* type)	{
	return mem_value;
}

void CSVPlugin::finish()	{
	close(fd);
	munmap(buf,fsize);
}

//Private Functions

void CSVPlugin::skip()
{
	while (pos < fsize && buf[pos] != ';' && buf[pos] != '\n') {
		pos++;
	}
	pos++;
}

inline size_t CSVPlugin::skipDelim(size_t pos, char* buf, char delim)	{
	while (buf[pos] != delim) {
		pos++;
	}
	pos++;
	return pos;
}

// Gist of the code generated:
// Output this as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:
void CSVPlugin::skipDelimLLVM(Value* delim,Function* debugChar, Function* debugInt)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Fetch values from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(posVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(bufVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}

	Function* TheFunction = Builder->GetInsertBlock()->getParent();
	// Create an alloca for the variable in the entry block.
	AllocaInst* Alloca = context->CreateEntryBlockAlloca(TheFunction, "cur_pos",int64Type);
	// Store the value into the alloca.
	Builder->CreateStore(Builder->CreateLoad(pos, "start_pos"), Alloca);

	// Make the new basic block for the loop header, inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "skipDelimLoop", TheFunction);

	// Insert an explicit fall through from the current block to the LoopBB.
	Builder->CreateBr(LoopBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);

	// Emit the body of the loop.
	// Here we essentially have no body; we only need to take care of the 'step'
	// Emit the step value. (+1)
	Value *StepVal= Builder->getInt64(1);

	// Compute the end condition.
	// Involves pointer arithmetics
	Value* index = Builder->CreateLoad(Alloca);
	Value* lhsPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
	Value* lhs = Builder->CreateLoad(lhsShiftedPtr,"bufVal");
	Value* rhs = delim;
	Value *EndCond = Builder->CreateICmpNE(lhs,rhs);

	// Reload, increment, and restore the alloca.
	//This handles the case where the body of the loop mutates the variable.
	Value *CurVar = Builder->CreateLoad(Alloca);
	Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
	Builder->CreateStore(NextVar, Alloca);

	// Create the "after loop" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "afterSkipDelimLoop", TheFunction);

	// Insert the conditional branch into the end of LoopEndBB.
	Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

	// Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);
	////////////////////////////////

	//'return' pos value
	Value *finalVar = Builder->CreateLoad(Alloca);
	Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);

}

void CSVPlugin::skipLLVM(Function* debug)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Value* delimInner = ConstantInt::get(llvmContext, APInt(8,';'));
	Value* delimEnd = ConstantInt::get(llvmContext, APInt(8,'\n'));

	//Fetch values from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(posVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(bufVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}
	AllocaInst* fsizePtr;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(fsizeVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + fsizeVar);
		}
		fsizePtr = it->second;
	}
	//Since we are the ones dictating what is flushed, file size should never have to be used in a check

	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	// Create an alloca for the variable in the entry block.
	AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction, "cur_pos",int64Type);
	Value* fsizeVal = Builder->CreateLoad(fsizePtr, "file_size");
	// Store the current pos value into the alloca, so that loop starts from appropriate point.
	// Redundant store / loads will be simplified by opt.pass
	Value* toInit = Builder->CreateLoad(pos, "start_pos");
	Builder->CreateStore(toInit, Alloca);

	// Make the new basic block for the loop header, inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "skipLoop", TheFunction);

	// Insert an explicit fall through from the current block to the LoopBB.
	Builder->CreateBr(LoopBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);

	// Emit the body of the loop.

	// Here we essentially have no body; we only need to take care of the 'step'
	// Emit the step value. (+1)
	Value *StepVal= Builder->getInt64(1);

	// Compute the end condition. More complex in this scenario (3 ands)
	Value* index = Builder->CreateLoad(Alloca);
	Value* lhsPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
	//equivalent to buf[pos]
	Value* lhs_ = Builder->CreateLoad(lhsShiftedPtr,"bufVal");
	//Only difference between skip() and skipDelim()!!!
	Value* rhs1 = delimInner;
	Value* rhs2 = delimEnd;
	Value *EndCond1 = Builder->CreateICmpNE(lhs_,rhs1);
	Value *EndCond2 = Builder->CreateICmpNE(lhs_,rhs2);
	Value *EndCond3 = Builder->CreateICmpSLT(Builder->CreateLoad(Alloca),fsizeVal);
	Value *EndCond_ = Builder->CreateAnd(EndCond1,EndCond2);
	Value *EndCond  = Builder->CreateAnd(EndCond_,EndCond3);

	// Reload, increment, and restore the alloca.  This handles the case where
	// the body of the loop mutates the variable.
	Value *CurVar = Builder->CreateLoad(Alloca);
	Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
	Builder->CreateStore(NextVar, Alloca);

	// Create the "after loop" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "afterSkipLoop", TheFunction);

	// Insert the conditional branch into the end of LoopEndBB.
	Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

	// Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);

	//'return' pos value
	Value *finalVar = Builder->CreateLoad(Alloca);
	Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);
}


void CSVPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, AllocaInst*>& variables, Function* atoi_,Function* debugChar,Function* debugInt)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Fetch values from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(posVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(bufVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}

	Value* start = Builder->CreateLoad(pos, "start_pos_atoi");
	skipLLVM(debugChar);
	//index must be different than start!
	Value* index = Builder->CreateLoad(pos, "end_pos_atoi");
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);

	std::vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(bufShiftedPtr);

	Value* parsedInt = Builder->CreateCall(atoi_, ArgsV, "atoi");
	AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
	Builder->CreateStore(parsedInt,Alloca);
	LOG(INFO) << "[READ INT: ] Atoi Successful";

	ArgsV.clear();
	ArgsV.push_back(parsedInt);
	//Debug
	//Builder->CreateCall(debugInt, ArgsV, "printi");
	variables[attName] = Alloca;
}

void CSVPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, AllocaInst*>& variables, Function* atof_,Function* debugChar,Function* debugFloat)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();


	//Fetch values from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(posVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(bufVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}

	Value* start = Builder->CreateLoad(pos, "start_pos_atoi");
	skipLLVM(debugChar);
	//index must be different than start!
	Value* index = Builder->CreateLoad(pos, "end_pos_atoi");
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
	std::vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(bufShiftedPtr);
	Value* parsedFloat = Builder->CreateCall(atof_, ArgsV, "atof");
	AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
	Builder->CreateStore(parsedFloat,Alloca);
	LOG(INFO) << "[READ FLOAT: ] Atof Successful";

	ArgsV.clear();
	ArgsV.push_back(parsedFloat);
	//Debug
	//Builder->CreateCall(debugFloat, ArgsV, "printf");
	variables[attName] = Alloca;
}

void CSVPlugin::scanCSV(const RawOperator& producer, Function* debug)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Container for the variable bindings
	map<RecordAttribute, AllocaInst*>* variableBindings = new map<RecordAttribute, AllocaInst*>();

	//Fetch value from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(posVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* fsizePtr;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesCSV.find(fsizeVar);
		if (it == NamedValuesCSV.end()) {
			throw runtime_error(string("Unknown variable name: ") + fsizeVar);
		}
		fsizePtr = it->second;
	}

	//  BYTECODE
	//	entry:
	//	%pos.addr = alloca i32, align 4
	//	%fsize.addr = alloca i32, align 4
	//	store i32 %pos, i32* %pos.addr, align 4
	//	store i32 %fsize, i32* %fsize.addr, align 4
	//	br label %for.cond

	//  API equivalent: Only the branch is needed. The allocas were taken care of before

	//Get the ENTRY BLOCK
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", TheFunction);

	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(CondBB);
	// Start insertion in CondBB.
	Builder->SetInsertPoint(CondBB);

	//	BYTECODE
	//	for.cond:                                         ; preds = %for.inc, %entry
	//	  %0 = load i32* %pos.addr, align 4
	//	  %1 = load i32* %fsize.addr, align 4
	//	  %cmp = icmp slt i32 %0, %1
	//	  br i1 %cmp, label %for.body, label %for.end

	Value* lhs = Builder->CreateLoad(pos);
	Value* rhs = Builder->CreateLoad(fsizePtr);
	Value *cond = Builder->CreateICmpSLT(lhs,rhs);

	// Make the new basic block for the loop header (BODY), inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", TheFunction);

	// Create the "AFTER LOOP" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", TheFunction);

	// Insert the conditional branch into the end of CondBB.
	Builder->CreateCondBr(cond, LoopBB, AfterBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);

	//Get the starting position of each record and pass it along.
	//More general/lazy CSV plugins will only perform this action,
	//instead of eagerly converting fields
	RecordAttribute tupleIdentifier = RecordAttribute(fname,activeLoop);
	(*variableBindings)[tupleIdentifier] = pos;

	//	BYTECODE
	//	for.body:                                         ; preds = %for.cond
	//	  br label %for.inc

	//Actual Work (Loop through attributes etc.)
	int cur_col = 0;
	Value* delimInner = ConstantInt::get(llvmContext, APInt(8,';'));
	Value* delimEnd = ConstantInt::get(llvmContext, APInt(8,'\n'));
	int lastFieldNo = -1;
	Function* atoi_ 		= context->getFunction("atoi");
	Function* atof_ 		= context->getFunction("atof");
	Function* debugChar 	= context->getFunction("printc");
	Function* debugInt 		= context->getFunction("printi");
	Function* debugFloat 	= context->getFunction("printFloat");

	if(atoi_ == 0 || atof_ == 0 || debugChar == 0 || debugInt == 0 || debugFloat == 0) {
		LOG(ERROR) <<"One of the functions needed not found!";
		throw runtime_error(string("One of the functions needed not found!"));
	}

	for (std::vector<RecordAttribute*>::iterator it = wantedFields.begin(); it != wantedFields.end(); it++)
	{
		int neededAttr = (*it)->getAttrNo() - 1;
		for( ; cur_col < neededAttr; cur_col++)	{
			skipDelimLLVM(delimInner,debugChar,debugInt);
		}

		std::string attrName = (*it)->getName();
		RecordAttribute attr = *(*it);
		switch ((*it)->getOriginalType()->getTypeID()) {
		case BOOL:
			LOG(ERROR)<< "[CSV PLUGIN: ] Booleans not supported yet";
			throw runtime_error(string("[CSV PLUGIN: ] Booleans not supported yet"));
		case STRING:
			LOG(ERROR) << "[CSV PLUGIN: ] String datatypes not supported yet";
			throw runtime_error(string("[CSV PLUGIN: ] String datatypes not supported yet"));
		case FLOAT:
			readAsFloatLLVM(attr,*variableBindings,atof_,debugChar,debugFloat);
			break;
		case INT:
			readAsIntLLVM(attr,*variableBindings,atoi_,debugChar,debugInt);
			break;
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain collections";
			throw runtime_error(string("[CSV PLUGIN: ] CSV files do not contain collections"));
		case RECORD:
			LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain record-valued attributes";
			throw runtime_error(string("[CSV PLUGIN: ] CSV files do not contain record-valued attributes"));
		default:
			LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
			throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
		}

		//Using it to know if a final skip is needed
		lastFieldNo = neededAttr + 1;
		cur_col++;
	}

	if(lastFieldNo < rec.getArgsNo()) {
		//Skip rest of line
		skipDelimLLVM(delimEnd,debugChar,debugInt);
	}

	// Make the new basic block for the increment, inserting after current block.
	BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

	// Insert an explicit fall through from the current (body) block to IncBB.
	Builder->CreateBr(IncBB);
	// Start insertion in IncBB.
	Builder->SetInsertPoint(IncBB);


	//Triggering parent
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context,*state);

	//	BYTECODE
	//	for.inc:                                          ; preds = %for.body
	//	  %2 = load i32* %pos.addr, align 4
	//	  %inc = add nsw i32 %2, 1
	//	  store i32 %inc, i32* %pos.addr, align 4
	//	  br label %for.cond

	Builder->CreateBr(CondBB);

	//	Finish up with end (the AfterLoop)
	// 	Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);
}

int CSVPlugin::readAsInt() {
	int start = pos;
	skip();
	return std::atoi(buf + start);
}

int CSVPlugin::eof() {
	return (pos >= fsize);
}
