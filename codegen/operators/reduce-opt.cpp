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

#include "operators/reduce-opt.hpp"
#include "util/raw-memory-manager.hpp"
#include "util/gpu/gpu-raw-context.hpp"

namespace opt {
Reduce::Reduce(vector<Monoid> accs,
		vector<expression_t> outputExprs,
		expression_t pred, RawOperator* const child,
		RawContext* context, bool flushResults, const char *outPath) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs),
		pred(std::move(pred)), context(context), flushResults(flushResults), outPath(outPath) {
	if (accs.size() != outputExprs.size()) {
		string error_msg = string("[REDUCE: ] Erroneous constructor args");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
}

void Reduce::produce() {
	flushResults = flushResults && !getParent(); //TODO: is this the best way to do it ?
	generate_flush();

	((GpuRawContext *) context)->popPipeline();

	auto flush_pip = ((GpuRawContext *) context)->removeLatestPipeline();
	flush_fun = flush_pip->getKernel();

	((GpuRawContext *) context)->pushPipeline(flush_pip);

	assert(mem_accumulators.empty());
	if (mem_accumulators.empty()){
		vector<Monoid>::const_iterator itAcc;
		vector<expression_t>::const_iterator itExpr;
		itAcc = accs.begin();
		itExpr = outputExprs.begin();

		int aggsNo = accs.size();
		/* Prepare accumulator FOREACH outputExpr */
		for (; itAcc != accs.end(); itAcc++, itExpr++) {
			auto acc = *itAcc;
			auto outputExpr = *itExpr;
			bool flushDelim = (aggsNo > 1) && (itAcc != accs.end() - 1);
			bool is_first   = (itAcc == accs.begin()  );
			bool is_last    = (itAcc == accs.end() - 1);
			size_t mem_accumulator = resetAccumulator(outputExpr, acc, flushDelim, is_first, is_last);
			mem_accumulators.push_back(mem_accumulator);
		}
	}

	getChild()->produce();
}

void Reduce::consume(RawContext* const context, const OperatorState& childState) {
	generate(context, childState);
	//flushResult();
}

void Reduce::flushResult() {
//	StringBuffer s;
//	Writer<StringBuffer> writer(s);
//	writer.StartObject();
//
//	switch (acc) {
//		case SUM:
//
//			break;
//		case MULTIPLY:
//
//			break;
//		case MAX:
//
//			break;
//		case OR:
//
//			break;
//		case AND:
//
//			break;
//		case UNION:
//		case BAGUNION:
//		case APPEND:
//		default: {
//			string error_msg = string("[Reduce: ] Unknown accumulator");
//			LOG(ERROR)<< error_msg;
//			throw runtime_error(error_msg);
//		}
//		}
//
//	writer.EndObject();
//	cout << s.GetString() << endl;
}

void Reduce::generate(RawContext* const context,
		const OperatorState& childState) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	int aggsNo = accs.size();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator{context, childState};
	RawValue condition = pred.accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "reduceCondEnd",
			TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "reduceIfCond",
			&ifBlock, endBlock);

	/**
	 * IF(pred) Block
	 */
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);

	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);

	/* Time to Compute Aggs */
	auto itAcc  = accs.begin();
	auto itExpr = outputExprs.begin();
	auto itMem  = mem_accumulators.begin();

	for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
		auto acc 					= *itAcc	;
		auto outputExpr 			= *itExpr	;
		Value * mem_accumulating	= NULL 		;

		switch (acc) {
		case SUM:
		case MULTIPLY:
		case MAX:
		case OR:
		case AND:{
			BasicBlock *cBB = Builder->GetInsertBlock();
			Builder->SetInsertPoint(context->getCurrentEntryBlock());

			mem_accumulating = context->getStateVar(*itMem);
			Value * acc_init = Builder->CreateLoad(mem_accumulating);
			Value * acc_mem  = context->CreateEntryBlockAlloca("acc", acc_init->getType());
			Builder->CreateStore(acc_init, acc_mem);

			Builder->SetInsertPoint(context->getEndingBlock());
			Builder->CreateStore(Builder->CreateLoad(acc_mem), mem_accumulating);

			Builder->SetInsertPoint(cBB);

			ExpressionGeneratorVisitor outputExprGenerator{context, childState};

			// Load accumulator -> acc_value
			RawValue acc_value;
			acc_value.value  = Builder->CreateLoad(acc_mem);
			acc_value.isNull = context->createFalse();

			// new_value = acc_value op outputExpr
			expressions::RawValueExpression val{outputExpr.getExpressionType(), acc_value};
			auto upd = toExpression(acc, val, outputExpr);
			RawValue new_val = upd.accept(outputExprGenerator);

			// store new_val to accumulator
			Builder->CreateStore(new_val.value, acc_mem);
			break;
		}
		case BAGUNION:
			generateBagUnion(outputExpr, context, childState, context->getStateVar(*itMem));
			break;
		case APPEND:
			//		generateAppend(context, childState);
			//		break;
		case UNION:
		default: {
			string error_msg = string(
					"[Reduce: ] Unknown / Still Unsupported accumulator");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
	}

	Builder->CreateBr(endBlock);

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

//Flush out whatever you received
//FIXME Need 'output plugin' / 'serializer'
void Reduce::generateBagUnion(expression_t outputExpr,
				RawContext* const context, const OperatorState& state, Value * cnt_mem) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	ExpressionFlusherVisitor flusher{context, state, outPath, outputExpr.getRegisteredRelName()};

	//Backing up insertion block
	BasicBlock *currBlock = Builder->GetInsertBlock();

	//Preparing collection output (e.g., flushing out '{' in the case of JSON)
	BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();

	Builder->SetInsertPoint(loopEntryBlock);
	flusher.beginList();

	//Restoring
	Builder->SetInsertPoint(currBlock);

	//results so far
	Value* resultCtr = Builder->CreateLoad(cnt_mem);

	//flushing out delimiter (IF NEEDED)
	flusher.flushDelim(resultCtr);

	outputExpr.accept(flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,context->createInt64(1));
	Builder->CreateStore(resultCtrInc, cnt_mem);

	//Backing up insertion block
	currBlock = Builder->GetInsertBlock();

	//Prepare final result output (e.g., flushing out '}' in the case of JSON)
	Builder->SetInsertPoint(context->getEndingBlock());
	flusher.endList();
	flusher.flushOutput();

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(currBlock);

	// ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){
	// });

	// ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){
	// 	((void (*)()) gpu_pip->getKernel(pip->get))();
	// });
}

//Materializes collection (in HT?)
//Use the catalog for the materialization
void Reduce::generateAppend(expression_t outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {

}

void Reduce::generate_flush(){
	LLVMContext & llvmContext   = context->getLLVMContext();

	(*((GpuRawContext *) context))->setMaxWorkerSize(1, 1);

	vector<size_t> params;

	auto itAcc  = accs.begin();
	auto itExpr = outputExprs.begin();

	for (; itAcc != accs.end(); itAcc++, itExpr++) {
		switch (*itAcc) {
			case SUM:
			case MULTIPLY:
			case MAX:
			case OR:
			case AND: {
				params.emplace_back(
					((GpuRawContext *) context)->appendParameter(
						PointerType::getUnqual(
							(*itExpr).getExpressionType()->getLLVMType(llvmContext)
						),
						true,
						true
					)
				);
				break;
			}
			case UNION: 
			case BAGUNION:
			case APPEND: {
				params.emplace_back(~((size_t) 0));
				break;
			}
			default: {
				string error_msg = string("[Reduce: ] Unknown accumulator");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
		}
	}

	context->setGlobalFunction();

	IRBuilder<> * Builder       = context->getBuilder    ();
	BasicBlock  * insBB         = Builder->GetInsertBlock();
	Function    * F             = insBB->getParent();


	BasicBlock * AfterBB = BasicBlock::Create(llvmContext, "end" , F);
	BasicBlock * MainBB  = BasicBlock::Create(llvmContext, "main", F);

	context->setCurrentEntryBlock(Builder->GetInsertBlock());
	context->setEndingBlock(AfterBB);

	Builder->SetInsertPoint(MainBB);

	map<RecordAttribute, RawValueMemory> variableBindings;

	std::string rel_name;
	bool found = false;
	for (const auto& t: outputExprs){
		if (t.isRegistered()){
			rel_name = t.getRegisteredRelName();
			found = true;
			break;
		}
	}

	if (found){
		Plugin     * pg = RawCatalog::getInstance().getPlugin(rel_name);

		{
			RecordAttribute tupleOID = RecordAttribute(rel_name, activeLoop, pg->getOIDType()); //FIXME: OID type for blocks ?

			Value      * oid = ConstantInt::get(pg->getOIDType()->getLLVMType(llvmContext), 0);
			oid->setName("oid");
			AllocaInst * mem = context->CreateEntryBlockAlloca(F, "activeLoop_ptr", oid->getType());
			Builder->CreateStore(oid, mem);

			RawValueMemory tmp;
			tmp.mem     = mem;
			tmp.isNull  = context->createFalse();

			variableBindings[tupleOID] = tmp;
		}

		{
			RecordAttribute tupleCnt = RecordAttribute(rel_name, "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?
			Value      * N   = ConstantInt::get(pg->getOIDType()->getLLVMType(llvmContext), 1);
			N->setName("cnt");
			AllocaInst * mem = context->CreateEntryBlockAlloca(F, "activeCnt_ptr", N->getType());
			Builder->CreateStore(N, mem);

			RawValueMemory tmp;
			tmp.mem     = mem;
			tmp.isNull  = context->createFalse();

			variableBindings[tupleCnt] = tmp;
		}
	}

	itAcc  = accs.begin();
	itExpr = outputExprs.begin();
	auto itMem  = params.begin();

	if (getParent()){
		for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
			auto acc 					= *itAcc	;
			auto outputExpr 			= *itExpr	;
			Value * mem_accumulating	= NULL 		;

			if (*itMem == ~((size_t) 0) || acc == BAGUNION) {
				string error_msg = string("[Reduce: ] Not implemented yet");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}

			if (!outputExpr.isRegistered()) {
				string error_msg = string("[Reduce: ] All expressions must be registered to forward them to the parent");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}

			Value      * val_mem    = ((GpuRawContext *) context)->getArgument(*itMem);
			val_mem->setName(outputExpr.getRegisteredAttrName() + "_ptr");
			Value      * val_acc    = Builder->CreateLoad(val_mem);
			AllocaInst * acc_alloca = context->CreateEntryBlockAlloca(outputExpr.getRegisteredAttrName(), val_acc->getType());

			context->getBuilder()->CreateStore(val_acc, acc_alloca);
			
			RawValueMemory acc_mem{acc_alloca, context->createFalse()};
			variableBindings[outputExpr.getRegisteredAs()] = acc_mem;
		}
	}

	if (flushResults){
		OperatorState state{*this, variableBindings};
		ExpressionFlusherVisitor flusher{context, state, outPath, outputExprs[0].getRegisteredRelName()};

		if (accs.size() > 1) flusher.beginList();

		itAcc  = accs.begin();
		itExpr = outputExprs.begin();
		itMem  = params.begin();

		for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
			auto acc 					= *itAcc	;
			auto outputExpr 			= *itExpr	;
			Value * mem_accumulating	= NULL 		;

			if (*itMem == ~((size_t) 0) || acc == BAGUNION) continue;
			
			Value      * val_mem    = ((GpuRawContext *) context)->getArgument(*itMem);
			Value      * val_acc    = Builder->CreateLoad(val_mem);
			
			flusher.flushValue(val_acc, outputExpr.getExpressionType()->getTypeID());
			bool flushDelim = (accs.size() > 1) && (itAcc != accs.end() - 1);
			if (flushDelim) flusher.flushDelim();
		}
		if (accs.size() > 1) flusher.endList();
		
		flusher.flushOutput();
	}

	if (getParent()){
		OperatorState state{*this, variableBindings};
		getParent()->consume(context, state);
	}
	// Insert an explicit fall through from the current (body) block to AfterBB.
	Builder->CreateBr(AfterBB);

	Builder->SetInsertPoint(context->getCurrentEntryBlock());
	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(MainBB);

	//  Finish up with end (the AfterLoop)
	//  Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(context->getEndingBlock());
}

size_t Reduce::resetAccumulator(expression_t outputExpr,
		Monoid acc, bool flushDelim, bool is_first, bool is_last) const {
	size_t mem_accum_id = ~((size_t) 0);

	//Deal with 'memory allocations' as per monoid type requested
	switch (acc) {
		case SUM:
		case MULTIPLY:
		case MAX:
		case OR:
		case AND: {
			Type * t = outputExpr.getExpressionType()
									->getLLVMType(context->getLLVMContext());

			mem_accum_id = context->appendStateVar(
				PointerType::getUnqual(t),

				[=](llvm::Value *){
					IRBuilder<> * Builder = context->getBuilder();

					Value * mem_acc = context->allocateStateVar(t);

					Constant * val_id = getIdentityElementIfSimple(
						acc,
						outputExpr.getExpressionType(),
						context
					);
					Builder->CreateStore(val_id, mem_acc);

					return mem_acc;
				},

				[=](llvm::Value *, llvm::Value * s){
					// if (flushResults && is_first && accs.size() > 1) flusher->beginList();

					// Value* val_acc =  context->getBuilder()->CreateLoad(s);

					// if (outputExpr.isRegistered()){
					// 	map<RecordAttribute, RawValueMemory> binding{};
					// 	AllocaInst * acc_alloca = context->CreateEntryBlockAlloca(outputExpr.getRegisteredAttrName(), val_acc->getType());
					// 	context->getBuilder()->CreateStore(val_acc, acc_alloca);
					// 	RawValueMemory acc_mem{acc_alloca, context->createFalse()};
					// 	binding[outputExpr.getRegisteredAs()] = acc_mem;
					// }

					if (is_first){	
						auto itAcc  = accs.begin();
						auto itExpr = outputExprs.begin();
						auto itMem  = mem_accumulators.begin();

						vector<Value *> args;
						for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
							auto acc 					= *itAcc	;
							auto outputExpr 			= *itExpr	;
							Value * mem_accumulating	= NULL 		;

							if (*itMem == ~((size_t) 0) || acc == BAGUNION) continue;
							
							args.emplace_back(context->getStateVar(*itMem));
						}

						IRBuilder<> * Builder = context->getBuilder();

						Type  * charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

						Function * f = context->getFunction("subpipeline_consume");
						FunctionType * f_t  = f->getFunctionType();

						Type  * substate_t  = f_t->getParamType(f_t->getNumParams()-1);
						
						Value * substate    = Builder->CreateBitCast(((GpuRawContext *) context)->getSubStateVar(), substate_t);
						args.emplace_back(substate);

						Builder->CreateCall(f, args);
					}

					// if (flushResults){
					// 	flusher->flushValue(val_acc, outputExpr.getExpressionType()->getTypeID());
					// 	if (flushDelim) flusher->flushDelim();
					// }
					context->deallocateStateVar(s);

					// if (flushResults && is_last  && accs.size() > 1) flusher->endList();

					// if (flushResults && is_last  ) flusher->flushOutput();
				}
			);
			break;
		}
		case UNION: {
			string error_msg = string("[Reduce: ] Not implemented yet");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		case BAGUNION:{
			Type * t = Type::getInt64Ty(context->getLLVMContext());

			mem_accum_id = context->appendStateVar(
				PointerType::getUnqual(t),

				[=](llvm::Value *){
					Value * m = context->allocateStateVar(t);
					IRBuilder<> * Builder = context->getBuilder();
					Builder->CreateStore(context->createInt64(0), m);
					return m;
				},

				[=](llvm::Value *, llvm::Value * s){
					context->deallocateStateVar(s);
				}
			);
			break;
		}
		case APPEND: {
			/*XXX Bags and Lists can be processed in streaming fashion -> No accumulator needed */
			break;
		}
		default: {
			string error_msg = string("[Reduce: ] Unknown accumulator");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
	}

	return mem_accum_id;
}

}
