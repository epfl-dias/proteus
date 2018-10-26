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

#include "operators/radix-nest.hpp"
#include  "plugins/csv-plugin-pm.hpp"
#include "expressions/expressions-flusher.hpp"

namespace radix	{



Nest::Nest(RawContext* const context, vector<Monoid> accs,
            vector<expressions::Expression*> outputExprs,
            vector<string> aggrLabels,
            expressions::Expression *pred,
            expressions::Expression *f_grouping,
            expressions::Expression *g_nullToZero,
            RawOperator* const child, const char* opLabel, Materializer& mat):
Nest(context, accs, outputExprs, aggrLabels, pred, std::vector<expressions::Expression *>{f_grouping}, g_nullToZero, child, opLabel, mat)
{}
/**
 * XXX NOTE on materializer:
 * While in the case of JSON the OID is enough to reconstruct anything needed,
 * the other plugins need to explicitly materialize the key constituents!
 * They are needed for dot equality evaluation, regardless of final output expr.
 *
 * FIXME do the null check for expression g!!
 * Previous nest version performs it
 */
Nest::Nest(RawContext* const context, vector<Monoid> accs,
		vector<expressions::Expression*> outputExprs, vector<string> aggrLabels,
		expressions::Expression *pred, std::vector<expressions::Expression *> f_grouping,
		expressions::Expression *g_nullToZero, RawOperator* const child,
		const char *opLabel, Materializer& mat) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs), aggregateLabels(
				aggrLabels), pred(pred), g_nullToZero(g_nullToZero), mat(mat), htName(opLabel), context((GpuRawContext * const) context)
{
	if (accs.size() != outputExprs.size() || accs.size() != aggrLabels.size()) {
		string error_msg = string("[NEST: ] Erroneous constructor args");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

    RawCatalog& catalog = RawCatalog::getInstance();

    Plugin * htPlugin;
    {
        //TODO: using a binary internal plugin seems more appropriate, but creates some problems, especially with records and lists for now
        RecordType * rec = new RecordType();

        vector<RecordAttribute *> projs;
        std::string * htString = new std::string(htName);
        htPlugin = new pm::CSVPlugin(context, *htString, *rec, projs, 1, 1);
        catalog.registerPlugin(*htString, htPlugin);
    }
    // Plugin *htPlugin = new BinaryInternalPlugin(context, htName);
    // catalog.registerPlugin(htName, htPlugin);

    assert(dynamic_cast<GpuRawContext * const>(context) && "Should update caller to use the new context!");
    LLVMContext& llvmContext = context->getLLVMContext();

    Type *int64_type = Type::getInt64Ty(llvmContext);
    Type *int32_type = Type::getInt32Ty(llvmContext);

    Type *int32_ptr_type = PointerType::getUnqual(int32_type);

    if (f_grouping.size() > 1){
        list<expressions::AttributeConstruction> *attrs = new list<expressions::AttributeConstruction>();
        std::vector<RecordAttribute *> recattr;
        std::string attrName = "__key";
        for (auto expr: f_grouping){
            assert(expr->isRegistered() && "All output expressions must be registered!");
            expressions::AttributeConstruction *newAttr =
                                            new expressions::AttributeConstruction(
                                                expr->getRegisteredAttrName(),
                                                expr
                                            );
            attrs->push_back(*newAttr);
            recattr.push_back(new RecordAttribute{expr->getRegisteredAs()});
            attrName += "_" + expr->getRegisteredAttrName();
            f_grouping_vec.push_back(expr);
        }

        this->f_grouping = new expressions::RecordConstruction(new RecordType(recattr), *attrs);
        this->f_grouping->registerAs(f_grouping[0]->getRegisteredRelName(), attrName);
    } else {
        this->f_grouping = f_grouping[0];
        f_grouping_vec.push_back(this->f_grouping);
    }





	/* What the type of internal radix HT per cluster is 	*/
	/* (int32*, int32*, unit32_t, void*, int32) */
	vector<Type*> htRadixClusterMembers;
	htRadixClusterMembers.push_back(int32_ptr_type);
	htRadixClusterMembers.push_back(int32_ptr_type);
	/* LLVM does not make a distinction between signed and unsigned integer type:
	 * Both are lowered to i32
	 */
	htRadixClusterMembers.push_back(int32_type);
	htRadixClusterMembers.push_back(int32_type);
	htClusterType = StructType::get(context->getLLVMContext(),htRadixClusterMembers);

    expressions::Expression * he = new expressions::HashExpression(this->f_grouping);
    const ExpressionType * he_type = he->getExpressionType();
	/* XXX What the type of HT entries is */
	/* (size_t, size_t) */
	vector<Type*> htEntryMembers;
	htEntryMembers.push_back(he_type->getLLVMType(llvmContext)); //32 ?
	htEntryMembers.push_back(int64_type);
	htEntryType = StructType::get(context->getLLVMContext(), htEntryMembers);
    int htEntrySize = context->getSizeOf(htEntryType); //sizeof(int32_t) + ...

	keyType = htEntryMembers[0];

	/* Arbitrary initial buffer sizes */
	/* No realloc will be required with these sizes for synthetic large-scale numbers */

	//XXX Meant for tpch-sf100. Reduce for smaller datasets
	//size_t sizeR = 15000000000; //Not enough
#ifdef LOCAL_EXEC
	size_t size = 30000;
#else
	size_t size = 30000000000;
#endif
	//size_t size = 1000;
	size_t kvSize = size;

    assert(context->getSizeOf(he_type->getLLVMType(llvmContext)) == (64/8));
    build = new RadixJoinBuild( he,
                                child,
                                this->context,
                                htName,
                                mat,
                                htEntryType,
                                size,
                                kvSize,
                                true);

//  /* Defined in consume() */
    payloadType = build->getPayloadType();
}

void Nest::produce() {
    probeHT();
    
    context->popPipeline();

    auto flush_pip = context->removeLatestPipeline();

    context->pushPipeline();

    // context->appendStateVar(
    //     Type::getInt32Ty(context->getLLVMContext()),
    //     [=](llvm::Value *){
    //         return UndefValue::get(Type::getInt32Ty(context->getLLVMContext()));
    //     },
    //     [=](llvm::Value *, llvm::Value * s){
    //         IRBuilder<> * Builder = context->getBuilder();

    //         Type  * charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

    //         Function * f = context->getFunction("subpipeline_consume");
    //         FunctionType * f_t  = f->getFunctionType();

    //         Type  * substate_t  = f_t->getParamType(f_t->getNumParams()-1);
            
    //         Value * substate    = Builder->CreateBitCast(context->getSubStateVar(), substate_t);

    //         Builder->CreateCall(f, vector<Value *>{substate});
    //     }
    // );

    RawOperator *child = getChild();
    child->setParent(build);
    build->setParent(this);
    setChild(build);
    
    context->setChainedPipeline(flush_pip);
    build->produce();

    // context->popNewPipeline();


// 	getChild()->produce();

	//generateProbe(this->context);
	// updateRelationPointers();
}

void Nest::consume(RawContext* const context, const OperatorState& childState) {
    assert(false && "Function should not be called! Is RadixJoinBuilders correctly set as child of this operator?");
}

map<RecordAttribute, RawValueMemory>* Nest::reconstructResults(Value *htBuffer, Value *idx, size_t relR_mem_relation_id) const {

	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();
	/*************************************/
	/**
	 * -> RECONSTRUCT RESULTS
	 * -> Need to do this at this point to check key equality
	 */
	Value *htRshiftedPtr_hit = Builder->CreateInBoundsGEP(htBuffer,
			idx);
	map<RecordAttribute, RawValueMemory>* allGroupBindings = new map<
			RecordAttribute, RawValueMemory>();

	/* Payloads (Relative Offsets): size_t */
	/* Must be added to relR accordingly */
	Value *val_payload_r_offset = context->getStructElem(htRshiftedPtr_hit, 1);

	/* Cast payload */
	PointerType *payloadPtrType = PointerType::get(payloadType, 0);

	Value *val_relR = context->getStateVar(relR_mem_relation_id);

	Value *val_ptr_payloadR = Builder->CreateInBoundsGEP(val_relR,
			val_payload_r_offset);

	Value *mem_payload = Builder->CreateBitCast(val_ptr_payloadR,
			payloadPtrType);
	Value *val_payload_r = Builder->CreateLoad(mem_payload);

	{
		//Retrieving activeTuple(s) from HT
		// AllocaInst *mem_activeTuple = NULL;
		int i = 0;
// //		const set<RecordAttribute>& tuplesIdentifiers =
// //				mat.getTupleIdentifiers();
// 		const vector<RecordAttribute*>& tuplesIdentifiers = mat.getWantedOIDs();
// //		cout << "How many OIDs? " << tuplesIdentifiers.size() << endl;
// 		vector<RecordAttribute*>::const_iterator it = tuplesIdentifiers.begin();
// 		for (; it != tuplesIdentifiers.end(); it++) {
// 			RecordAttribute *attr = *it;
// //			cout << "Dealing with " << attr->getRelationName() << "_"
// //					<< attr->getAttrName() << endl;
// 			mem_activeTuple = context->CreateEntryBlockAlloca(F,
// 					"mem_activeTuple", payloadType->getElementType(i));
// 			vector<Value*> idxList = vector<Value*>();
// 			idxList.push_back(context->createInt32(0));
// 			idxList.push_back(context->createInt32(i));

// 			Value *elem_ptr = Builder->CreateGEP(mem_payload, idxList);
// 			Value *val_activeTuple = Builder->CreateLoad(elem_ptr);
// 			StoreInst *store_activeTuple = Builder->CreateStore(val_activeTuple,
// 					mem_activeTuple);
// 			store_activeTuple->setAlignment(8);

// 			RawValueMemory mem_valWrapper;
// 			mem_valWrapper.mem = mem_activeTuple;
// 			mem_valWrapper.isNull = context->createFalse();
// 			(*allGroupBindings)[*attr] = mem_valWrapper;
// 			i++;
// 		}

		// AllocaInst *mem_field = NULL;
// 		const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
// 		vector<RecordAttribute*>::const_iterator it2 = wantedFields.begin();
// 		for (; it2 != wantedFields.end(); it2++) {
        for (const auto &expr2: mat.getWantedExpressions()) {
			// RecordAttribute *attr = *it2;
			// cout << "Dealing with " << expr2->getRegisteredAs().getRelationName() << "_"
			// 		<< expr2->getRegisteredAs().getAttrName() << endl;

			// string currField = (*it2)->getName();
            string currField = expr2->getRegisteredAttrName();
			AllocaInst * mem_field = context->CreateEntryBlockAlloca(F, "mem_" + currField,
					payloadType->getElementType(i));
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(i));

			Value *elem_ptr = Builder->CreateGEP(mem_payload, idxList);
			Value *val_field = Builder->CreateLoad(elem_ptr);
			Builder->CreateStore(val_field, mem_field);

			RawValueMemory mem_valWrapper;
			mem_valWrapper.mem = mem_field;
			mem_valWrapper.isNull = context->createFalse();

			(*allGroupBindings)[expr2->getRegisteredAs()] = mem_valWrapper;
			i++;
		}
	}
	/*****************************************/
//	OperatorState* newState = new OperatorState(*this, *allGroupBindings);
//	return newState;
	return allGroupBindings;
}

void Nest::probeHT() const	{
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder();
    RawCatalog  & catalog       = RawCatalog::getInstance();

    Type        * int64_type    = Type::getInt64Ty  (llvmContext);
    Type        * int32_type    = Type::getInt32Ty  (llvmContext);
    PointerType * char_ptr_type = Type::getInt8PtrTy(llvmContext);

    size_t clusterCountR_id = context->appendStateVar(
        PointerType::getUnqual(int32_type),
        [=](llvm::Value * pip){
            LLVMContext & llvmContext   = context->getLLVMContext();
            IRBuilder<> * Builder       = context->getBuilder();

            Value       * build         = context->CastPtrToLlvmPtr(char_ptr_type, this->build);
            Function    * clusterCnts   = context->getFunction("getClusterCounts");
            return Builder->CreateCall(clusterCnts, vector<Value *>{pip, build});
        },
        [=](llvm::Value *, llvm::Value * s){
            Function    * f = context->getFunction("free");
            s = Builder->CreateBitCast(s, char_ptr_type);
            Builder->CreateCall(f, s);
        },
        "clusterCountR"
    );

    size_t htR_mem_kv_id = context->appendStateVar(
        PointerType::getUnqual(htEntryType),
        [=](llvm::Value * pip){
            LLVMContext & llvmContext   = context->getLLVMContext();
            IRBuilder<> * Builder       = context->getBuilder();

            Value       * build         = context->CastPtrToLlvmPtr(char_ptr_type, this->build);
            Function    * ht_mem_kv     = context->getFunction("getHTMemKV");
            Value       * char_ht_mem   = Builder->CreateCall(ht_mem_kv, vector<Value *>{pip, build});
            return Builder->CreateBitCast(char_ht_mem, PointerType::getUnqual(htEntryType));
        },
        [=](llvm::Value *, llvm::Value * s){
            Function    * f = context->getFunction("releaseMemoryChunk");
            s = Builder->CreateBitCast(s, char_ptr_type);
            Builder->CreateCall(f, s);
        },
        "htR_mem_kv"
    ); //FIXME: read-only, we do not even have to maintain it as state variable

    size_t relR_mem_relation_id = context->appendStateVar(
        char_ptr_type,
        [=](llvm::Value * pip){
            LLVMContext & llvmContext   = context->getLLVMContext();
            IRBuilder<> * Builder       = context->getBuilder();

            Value       * build         = context->CastPtrToLlvmPtr(char_ptr_type, this->build);
            Function    * rel_mem       = context->getFunction("getRelationMem");
            return Builder->CreateCall(rel_mem, vector<Value *>{pip, build});
        },
        [=](llvm::Value *, llvm::Value * s){
            Function    * f = context->getFunction("releaseMemoryChunk");
            s = Builder->CreateBitCast(s, char_ptr_type);
            Builder->CreateCall(f, s);
        },
        "relR_mem_relation"
    ); //FIXME: read-only, we do not even have to maintain it as state variable

    context->setGlobalFunction();
    Function * F = context->getGlobalFunction();

    Value *val_zero = context->createInt32(0);
    Value *val_one = context->createInt32(1);

	// Function* debugInt = context->getFunction("printi");
	// Function* debugInt64 = context->getFunction("printi64");

	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *bool_ptr_type = PointerType::get(int8_type,0);
	Value *val_true = context->createInt8(1);
	Value *val_false = context->createInt8(0);

	/* Partition and Cluster 'R' (the corresponding htEntries) */
    Value *clusterCountR = context->getStateVar(clusterCountR_id);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());

	/* Bookkeeping for next steps - not sure which ones we will end up using */
	AllocaInst *mem_rCount = Builder->CreateAlloca(int32_type, 0, "rCount");
	AllocaInst *mem_clusterCount = Builder->CreateAlloca(int32_type, 0,
			"clusterCount");
	Builder->CreateStore(val_zero, mem_rCount);
	Builder->CreateStore(val_zero, mem_clusterCount);

	uint32_t clusterNo = (1 << NUM_RADIX_BITS);
	Value *val_clusterNo = context->createInt32(clusterNo);

    /* Request memory for HT(s) construction        */
    /* Note: Does not allocate mem. for buckets too */
    size_t htSize = (1 << NUM_RADIX_BITS) * sizeof(HT);
    // HT * HT_per_cluster = (HT *) getMemoryChunk(htSize); //FIXME: do in codegen, otherwise it prevent parallelization!!!!

	Builder->CreateAlloca(htClusterType, 0, "HTimpl");
	PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
	// Value *val_htPerCluster = context->CastPtrToLlvmPtr(htClusterPtrType,
	// 		HT_per_cluster);
    Function *f_getMemoryChunk = context->getFunction("getMemoryChunk");
    Value * HT_mem = Builder->CreateCall(f_getMemoryChunk, vector<Value *>{context->createSizeT(htSize)});
    Value * val_htPerCluster = Builder->CreateBitCast(HT_mem, htClusterPtrType);

	AllocaInst *mem_probesNo = Builder->CreateAlloca(int32_type, 0,
			"mem_counter");
	Builder->CreateStore(val_zero, mem_probesNo);

	vector<Monoid>::const_iterator itAcc;
	vector<expressions::Expression*>::const_iterator itExpr;
	vector<AllocaInst*> mem_accumulators;
	/*************************************************************************/
	/**
	 * ACTUAL PROBES
	 */

	/* Loop through clusters */
	/* for (i = 0; i < (1 << NUM_RADIX_BITS); i++) */

	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	context->CreateForLoop("clusterLoopCond", "clusterLoopBody",
			"clusterLoopInc", "clusterLoopEnd", &loopCond, &loopBody, &loopInc,
			&loopEnd);
	context->setEndingBlock(loopEnd);

	/* 1. Loop Condition - Unsigned integers operation */
	// Builder->CreateBr(loopCond);
	Builder->SetInsertPoint(loopCond);
	Value *val_clusterCount = Builder->CreateLoad(mem_clusterCount);
	Value *val_cond = Builder->CreateICmpULT(val_clusterCount, val_clusterNo);
	Builder->CreateCondBr(val_cond, loopBody, loopEnd);

	/* 2. Loop Body */
	Builder->SetInsertPoint(loopBody);

	/* Check cluster contents */
	/* if (R_count_per_cluster[i] > 0)
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(),
			"ifNotEmptyCluster", "elseEmptyCluster", &ifBlock, &elseBlock,
			loopInc);

	{
		/* If Condition */
		Value *val_r_i_count = context->getArrayElem(clusterCountR,
				val_clusterCount);
		Value *val_cond = Builder->CreateICmpSGT(val_r_i_count, val_zero);

		Builder->CreateCondBr(val_cond, ifBlock, elseBlock);

		/* If clusters non-empty */
		Builder->SetInsertPoint(ifBlock);
		/* start index of next cluster */
		Value *val_rCount = Builder->CreateLoad(mem_rCount);

		/* tmpR.tuples = relR->tuples + r; */
        Value *val_htR = context->getStateVar(htR_mem_kv_id);
		Value* htRshiftedPtr = Builder->CreateInBoundsGEP(val_htR, val_rCount);

		Function *bucketChainingAggPrepare = context->getFunction(
				"bucketChainingAggPrepare");

		PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
		Value *val_htPerClusterShiftedPtr = Builder->CreateInBoundsGEP(
				val_htPerCluster, val_clusterCount);

		//Prepare args and call function
		vector<Value*> Args;
		Args.push_back(htRshiftedPtr);
		Args.push_back(val_r_i_count);
		Args.push_back(val_htPerClusterShiftedPtr);
		Builder->CreateCall(bucketChainingAggPrepare, Args);
#ifdef DEBUGRADIX_NEST
		{
//			vector<Value*> ArgsV;
//			ArgsV.clear();
//			ArgsV.push_back(val_clusterCount);
//			Builder->CreateCall(debugInt, ArgsV);
//
//			ArgsV.clear();
//			ArgsV.push_back(val_r_i_count);
//			Builder->CreateCall(debugInt, ArgsV);
		}
#endif

		/*
		 * r += R_count_per_cluster[i];
		 */
		val_rCount = Builder->CreateAdd(val_rCount, val_r_i_count);
		Builder->CreateStore(val_rCount, mem_rCount);

		/* DO WORK HERE -> Needed Cluster Found! */
		/* Loop over R cluster (tmpR) and use its tuples to probe its own HT */
		/* Remember: Any clustering/grouping obtained so far only used a few bits! */
		//XXX Reduce number of visits here!
		BasicBlock *rLoopCond, *rLoopBody, *rLoopInc, *rLoopEnd;
		context->CreateForLoop("rLoopCond", "rLoopBody", "rLoopInc", "rLoopEnd",
				&rLoopCond, &rLoopBody, &rLoopInc, &rLoopEnd);
		{
			/* ENTRY:
			 * -> Allocate  'marked' array to know which elems have participated
			 * -> Could be bitmap too!
			 * bool marked[]
			 */
			Value *val_r_i_count64 = Builder->CreateSExt(val_r_i_count,int64_type);
			Function *func_getMemory = context->getFunction("getMemoryChunk");
			vector<Value*> ArgsV;
			ArgsV.push_back(val_r_i_count64);
			//void*
			Value *val_array_marked = Builder->CreateCall(func_getMemory,ArgsV);

			AllocaInst *mem_j = Builder->CreateAlloca(int32_type, 0, "j_cnt");
			/* A consecutive identifier, to act as OID later on */
			AllocaInst *mem_groupCnt = Builder->CreateAlloca(int32_type, 0, "group_cnt");
			Builder->CreateStore(val_zero, mem_j);
			Builder->CreateStore(val_zero, mem_groupCnt);
			Builder->CreateBr(rLoopCond);

			/* Loop Condition */
			Builder->SetInsertPoint(rLoopCond);
			Value *val_j = Builder->CreateLoad(mem_j);
			Value *val_groupCnt = Builder->CreateLoad(mem_groupCnt);

			val_cond = Builder->CreateICmpSLT(val_j, val_r_i_count);

			Builder->CreateCondBr(val_cond, rLoopBody, rLoopEnd);

			Builder->SetInsertPoint(rLoopBody);

			/*
			 * Break the following in pieces:
			 * result += bucket_chaining_join_probe(&tmpR,
			 *			&(HT_per_cluster[i]), &(tmpS.tuples[j]));
			 */
			itAcc = accs.begin();
			itExpr = outputExprs.begin();
			/* Prepare accumulator FOREACH outputExpr */
			for (; itAcc != accs.end(); itAcc++, itExpr++) {
				Monoid acc = *itAcc;
				expressions::Expression *outputExpr = *itExpr;
				AllocaInst *mem_accumulator = resetAccumulator(outputExpr, acc);
				mem_accumulators.push_back(mem_accumulator);
			}
			/* XXX I HAVE THE HASHKEY READY!!!
			 * NO REASON TO RE-CALC (?)!!
			 *
			 * BUT:*/
			//Get key of current r tuple (tmpR[j])
//			Value *htRshiftedPtr_j = Builder->CreateInBoundsGEP(htRshiftedPtr,
//					val_j);
//			Value *val_key_r_j = context->getStructElem(htRshiftedPtr_j, 0);
//			Value *val_idx = val_key_r_j;

			/* I think diff't hash causes issues with the buckets */
			/* uint32_t idx = HASH_BIT_MODULO(s->key, ht->mask, NUM_RADIX_BITS); */
			Value *val_num_radix_bits64 = context->createInt64(NUM_RADIX_BITS);
			Value *val_mask =
					context->getStructElem(val_htPerClusterShiftedPtr,2);
            {Function* debugInt = context->getFunction("printi");
                vector<Value*> ArgsV;
                ArgsV.clear();
                ArgsV.push_back(val_mask);
                Builder->CreateCall(debugInt, ArgsV);
            }
			Value *val_mask64 = Builder->CreateZExt(val_mask,int64_type);
			//Get key of current tuple (tmpR[j])
			Value *htRshiftedPtr_j = Builder->CreateInBoundsGEP(htRshiftedPtr,
					val_j);
			Value *val_hashed_key_r_j = context->getStructElem(htRshiftedPtr_j, 0);
			Value *val_idx = Builder->CreateBinOp(Instruction::And, val_hashed_key_r_j,
					val_mask64);
			val_idx = Builder->CreateAShr(val_idx,val_num_radix_bits64);

			/* Also getting value to reassemble ACTUAL KEY when needed */
			map<RecordAttribute, RawValueMemory> *currKeyBindings =
					reconstructResults(htRshiftedPtr,val_j, relR_mem_relation_id);
			OperatorState *currKeyState =
					new OperatorState(*this,*currKeyBindings);
			map<RecordAttribute, RawValueMemory> *retrievedBindings;

			/**
			 * Checking actual hits (when applicable)
			 * for(int hit = (ht->bucket)[idx]; hit > 0; hit = (ht->next)[hit-1])
			 */
			BasicBlock *hitLoopCond, *hitLoopBody, *hitLoopInc, *hitLoopEnd;
			context->CreateForLoop("hitLoopCond", "hitLoopBody", "hitLoopInc",
					"hitLoopEnd", &hitLoopCond, &hitLoopBody, &hitLoopInc,
					&hitLoopEnd);

			{
				AllocaInst *mem_hit = Builder->CreateAlloca(int32_type, 0,
						"hit");
				//(ht->bucket)
				Value *val_bucket = context->getStructElem(
						val_htPerClusterShiftedPtr, 0);
				//(ht->bucket)[idx]
				Value *val_bucket_idx = context->getArrayElem(val_bucket,
						val_idx);

				Builder->CreateStore(val_bucket_idx, mem_hit);
				Builder->CreateBr(hitLoopCond);
				/* 1. Loop Condition */
				Builder->SetInsertPoint(hitLoopCond);
				Value *val_hit = Builder->CreateLoad(mem_hit);
				val_cond = Builder->CreateICmpSGT(val_hit, val_zero);

				Builder->CreateCondBr(val_cond, hitLoopBody, hitLoopEnd);

				/* 2. Body */
				Builder->SetInsertPoint(hitLoopBody);

				/* XXX TIME TO CALCULATE ACTUAL KEY */
				/* Can reduce comparisons if I skip 'marked' matches */
				BasicBlock *ifNotMarked;
				BasicBlock *ifKeyMatch;
				/* Must flag 'marked' array accordingly */
				Value *val_hit_idx_dec = Builder->CreateSub(val_hit, val_one);
				Value *mem_toFlag = context->getArrayElemMem(val_array_marked,
											val_hit_idx_dec);
				context->CreateIfBlock(context->getGlobalFunction(),
						"htMatchIfCond", &ifNotMarked, hitLoopInc);
				{
					Value *val_flag = Builder->CreateLoad(mem_toFlag);
					Value *val_isNotMarked = Builder->CreateICmpEQ(val_flag,val_false);
					Builder->CreateCondBr(val_isNotMarked,ifNotMarked,hitLoopInc);
					Builder->SetInsertPoint(ifNotMarked);
				}
				/* if (r->key == Rtuples[hit - 1].key) */

				context->CreateIfBlock(context->getGlobalFunction(),
						"htMatchIfCond", &ifKeyMatch, hitLoopInc);
				{
					retrievedBindings =
							reconstructResults(htRshiftedPtr,val_hit_idx_dec, relR_mem_relation_id);
					OperatorState *retrievedState = new OperatorState(*this,*retrievedBindings);

					/* Condition: Checking dot equality */
					ExpressionDotVisitor dotVisitor{context,*currKeyState,*retrievedState};

                    Value * val_cond = context->createTrue();

                    for (const auto &k: f_grouping_vec){
                        // std::cout << "===> " << k->getRegisteredAs().getRelationName() << " " << k->getRegisteredAs().getAttrName() <<std::endl;
                        // for (const auto &t: currKeyState->getBindings()) std::cout << t.first.getRelationName() << " " << t.first.getAttrName() <<std::endl;
                        RawValueMemory currKeyMem = currKeyState->getBindings().at(k->getRegisteredAs());
                        Value *        currKey    = Builder->CreateLoad(currKeyMem.mem);
                        RawValueMemory retrKeyMem = retrievedState->getBindings().at(k->getRegisteredAs());
                        Value *        retrKey    = Builder->CreateLoad(retrKeyMem.mem);
         //                expressions::RawValueExpression currKey{f_grouping->getExpressionType(), };


         //                RecordAttribute                 att{f_grouping->getRegisteredAs()};
         //                expressions::InputArgument      arg{att.getOriginalType(), -1, list<RecordAttribute>{att}};
         //                expressions::RecordProjection   key{&arg, att};

                        // std::cout << "=1?" << k->getRegisteredAs().getOriginalRelationName() << std::endl;
                        // std::cout << "=2?" << k->getRegisteredAs().getRelationName() << std::endl;
                        // std::cout << "=3?" << k->getRegisteredAs().getAttrName() << std::endl;
                        // std::cout << "=4?" << k->getRegisteredAs().getType() << std::endl;
                        // // RawValue val_condWrapper = key.acceptTandem(dotVisitor, &key);
                        // if (k->getRegisteredAs().getOriginalType()->getTypeID() == DSTRING){
                        //     // ExpressionFlusherVisitor fl{context, *currKeyState, "tmp.txt"};
                        //     // RawValue v{currKey, context->createFalse()};
                        //     // expressions::RawValueExpression{k->getExpressionType(), v}.accept(fl);
                        //     Function *f = context->getFunction("printi");
                        //     Builder->CreateCall(f, currKey);
                        // }

                        expressions::RawValueExpression curre{k->getExpressionType(), RawValue{currKey, currKeyMem.isNull}};
                        expressions::RawValueExpression retre{k->getExpressionType(), RawValue{retrKey, retrKeyMem.isNull}};
                        RawValue eq = curre.acceptTandem(dotVisitor, &retre);
                        val_cond = Builder->CreateAnd(val_cond, eq.value);
                    }
					//Value *val_cond = context->createTrue();
					//val_cond = Builder->CreateICmpEQ(tmp, tmp);
					Builder->CreateCondBr(val_cond, ifKeyMatch, hitLoopInc);

					Builder->SetInsertPoint(ifKeyMatch);
#ifdef DEBUGRADIX_NEST
//					{
//						/* Printing the hashKey*/
//						vector<Value*> ArgsV;
//						ArgsV.clear();
//						ArgsV.push_back(val_key_r_j);
//						Builder->CreateCall(debugInt64, ArgsV);
//					}
//					{
//						/* Printing the pos. to be marked */
//						vector<Value*> ArgsV;
//						ArgsV.clear();
//						ArgsV.push_back(val_hit_idx_dec);
//						Builder->CreateCall(debugInt, ArgsV);
//					}
#endif
					/* marked[hit -1] = true; */
					Builder->CreateStore(val_true,mem_toFlag);

					/* Time to Compute Aggs */

                    //Generate condition
                    ExpressionGeneratorVisitor predExprGenerator{context, *retrievedState};
                    RawValue condition = pred->accept(predExprGenerator);
                    /**
                     * Predicate Evaluation:
                     */
                    BasicBlock* entryBlock = Builder->GetInsertBlock();

                    BasicBlock *ifBlock;
                    context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond",
                            &ifBlock, hitLoopInc);

                    /**
                     * IF(pred) Block
                     */
                    RawValue val_output;
                    Builder->SetInsertPoint(entryBlock);

                    Builder->CreateCondBr(condition.value, ifBlock, hitLoopInc);
                    // Builder->CreateBr(ifBlock);

                    Builder->SetInsertPoint(ifBlock);

					itAcc = accs.begin();
					itExpr = outputExprs.begin();
					vector<AllocaInst*>::const_iterator itMem =
							mem_accumulators.begin();
					vector<string>::const_iterator itLabels =
							aggregateLabels.begin();
					/* Accumulate FOREACH outputExpr */
					for (; itAcc != accs.end();
							itAcc++, itExpr++, itMem++, itLabels++) { // increment only when using materialized results: 
						Monoid acc = *itAcc;
						expressions::Expression *outputExpr = *itExpr;
						AllocaInst *mem_accumulating = *itMem;
						string aggregateName = *itLabels;

						switch (acc) {
                        case SUM:
                        case MULTIPLY:
                        case MAX:
                        case OR:
                        case AND:{
                            ExpressionGeneratorVisitor outputExprGenerator{context, *retrievedState};

                            // Load accumulator -> acc_value
                            RawValue acc_value;
                            acc_value.value  = Builder->CreateLoad(mem_accumulating);
                            acc_value.isNull = context->createFalse();

                            RawValue acc_value2;
                            bool val_unset = true;
                            if (outputExpr->isRegistered()){
                                const auto &f = retrievedState->getBindings().find(outputExpr->getRegisteredAs());
                                if (f != retrievedState->getBindings().end()){
                                    RawValueMemory mem_val = f->second;
                                    acc_value2.value = Builder->CreateLoad(mem_val.mem);
                                    acc_value2.isNull = mem_val.isNull;
                                    // itExpr++;
                                    val_unset = false;
                                }
                            }

                            if (val_unset) acc_value2 = outputExpr->accept(outputExprGenerator);

                            // new_value = acc_value op outputExpr
                            expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc_value);
                            expressions::Expression * val2 = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc_value2);
                            expressions::Expression * upd = toExpression(acc, val, val2);//outputExpr);
                            assert(upd && "Monoid is not convertible to expression!");
                            RawValue new_val = upd->accept(outputExprGenerator);

                            // store new_val to accumulator
                            Builder->CreateStore(new_val.value, mem_accumulating);
                            break;
                        }
						case UNION:
							//		generateUnion(context, childState);
							//		break;
						case BAGUNION:
							//		generateBagUnion(context, childState);
							//		break;
						case APPEND:
							//		generateAppend(context, childState);
							//		break;
						default: {
							string error_msg =
									string(
											"[Nest: ] Unknown / Still Unsupported accumulator");
							LOG(ERROR)<< error_msg;
							throw runtime_error(error_msg);
						}
						}

						RecordAttribute attr_aggr = RecordAttribute(htName,
								aggregateName, outputExpr->getExpressionType());
						//cout << "Registering custom pg for " << htName << endl;
						RawValueMemory mem_aggrWrapper;
						mem_aggrWrapper.mem = mem_accumulating;
						mem_aggrWrapper.isNull = context->createFalse();
						(*retrievedBindings)[attr_aggr] = mem_aggrWrapper;
					}

					Builder->CreateBr(hitLoopInc);
				}

				/* 3. Inc: hit = (ht->next)[hit-1]) */
				Builder->SetInsertPoint(hitLoopInc);
				//(ht->next)
				Value *val_next = context->getStructElem(
						val_htPerClusterShiftedPtr, 1);
				Value *val_hit_idx = Builder->CreateSub(val_hit, val_one);
				//(ht->next)[hit-1])
				val_hit = context->getArrayElem(val_next, val_hit_idx);
				Builder->CreateStore(val_hit, mem_hit);
				Builder->CreateBr(hitLoopCond);

				/* 4. End */
				Builder->SetInsertPoint(hitLoopEnd);
			}

			Builder->CreateBr(rLoopInc);
			Builder->SetInsertPoint(rLoopInc);

			/* No longer moving relCounter once.
			 * -> Move until no position is marked!
			 *
			val_j = Builder->CreateLoad(mem_j);
			val_j = Builder->CreateAdd(val_j, val_one);
			Builder->CreateStore(val_j, mem_j);
			Builder->CreateBr(rLoopCond);
			*/
			/*
			 * NEW INC!
			 * XXX callParent();
			 * while(marked[j] == true) j++;
			 */
			/* Explicit OID (i.e., groupNo) materialization */
			RawValueMemory mem_oidWrapper;
			mem_oidWrapper.mem = mem_groupCnt;
			mem_oidWrapper.isNull = context->createFalse();
			ExpressionType *oidType = new IntType();
			RecordAttribute attr_oid = RecordAttribute(htName, activeLoop,
					oidType);
			(*retrievedBindings)[attr_oid] = mem_oidWrapper;
			OperatorState *retrievedState = new OperatorState(*this,*retrievedBindings);

			val_groupCnt = Builder->CreateLoad(mem_groupCnt);
			val_groupCnt = Builder->CreateAdd(val_groupCnt, val_one);
			Builder->CreateStore(val_groupCnt, mem_groupCnt);

			getParent()->consume(context, *retrievedState);
			BasicBlock *nextUnmarkedCond, *nextUnmarkedLoopBody,
					*nextUnmarkedLoopInc, *nextUnmarkedLoopEnd;
			context->CreateForLoop("nextUnmarkedCond", "nextUnmarkedBody",
					"nextUnmarkedInc", "nextUnmarkedEnd", &nextUnmarkedCond,
					&nextUnmarkedLoopBody, &nextUnmarkedLoopInc,
					&nextUnmarkedLoopEnd);
			{
				Builder->CreateBr(nextUnmarkedCond);
				/* Create condition */
				Builder->SetInsertPoint(nextUnmarkedCond);
				val_j = Builder->CreateLoad(mem_j);
				Value *mem_flag = context->getArrayElemMem(val_array_marked,
						val_j);
				Value *val_flag = Builder->CreateLoad(mem_flag);
				Value *val_cond = Builder->CreateICmpEQ(val_flag, val_true);
				Builder->CreateCondBr(val_cond,nextUnmarkedLoopBody,nextUnmarkedLoopEnd);

				Builder->SetInsertPoint(nextUnmarkedLoopBody);
				/* Nothing to do really - job done in inc block*/
				Builder->CreateBr(nextUnmarkedLoopInc);

				Builder->SetInsertPoint(nextUnmarkedLoopInc);
				val_j = Builder->CreateLoad(mem_j);
				val_j = Builder->CreateAdd(val_j, val_one);
				Builder->CreateStore(val_j, mem_j);
				Builder->CreateBr(nextUnmarkedCond);

				Builder->SetInsertPoint(nextUnmarkedLoopEnd);
			}
			Builder->CreateBr(rLoopCond);
			/* END OF NEW INC */


			Builder->SetInsertPoint(rLoopEnd);
			/* Free tmp marked array */
			Function *func_releaseMemory = context->getFunction(
					"releaseMemoryChunk");
			ArgsV.clear();
			ArgsV.push_back(val_array_marked);
			Builder->CreateCall(func_releaseMemory, ArgsV);
		}
		/* END OF WORK HERE*/
		Builder->CreateBr(loopInc);

		/* If cluster is empty */
		/*
		 * r += R_count_per_cluster[i];
		 */
		Builder->SetInsertPoint(elseBlock);
		val_rCount = Builder->CreateLoad(mem_rCount);
		val_rCount = Builder->CreateAdd(val_rCount, val_r_i_count);
		Builder->CreateStore(val_rCount, mem_rCount);
		Builder->CreateBr(loopInc);
	}

	/* 3. Loop Inc. */
	Builder->SetInsertPoint(loopInc);
	val_clusterCount = Builder->CreateLoad(mem_clusterCount);
	val_clusterCount = Builder->CreateAdd(val_clusterCount, val_one);
	#ifdef DEBUGRADIX_NEST
//			vector<Value*> ArgsV0;
//			ArgsV0.push_back(val_clusterCount);
//			Builder->CreateCall(debugInt,ArgsV0);
	#endif
	Builder->CreateStore(val_clusterCount, mem_clusterCount);

	Builder->CreateBr(loopCond);

	Builder->SetInsertPoint(context->getCurrentEntryBlock());
	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(loopCond);
	
	/* 4. Loop End */
	Builder->SetInsertPoint(context->getEndingBlock());
}

AllocaInst* Nest::resetAccumulator(expressions::Expression* outputExpr, Monoid acc) const {
    IRBuilder<>* Builder = context->getBuilder();
    LLVMContext& llvmContext = context->getLLVMContext();
    Function *f = Builder->GetInsertBlock()->getParent();

    Type * t = outputExpr->getExpressionType()->getLLVMType(llvmContext);
    AllocaInst * mem_acc = context->CreateEntryBlockAlloca(f, "dest_acc", t);

    switch (acc) {
        case SUM:
        case MULTIPLY:
        case MAX:
        case OR:
        case AND: {
            Constant * val_id = getIdentityElementIfSimple(
                acc,
                outputExpr->getExpressionType(),
                context
            );
            Builder->CreateStore(val_id, mem_acc);
            break;
        }
        case UNION:
        case BAGUNION: {
            string error_msg = string("[Nest: ] Not implemented yet");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        case APPEND: {
            //XXX Reduce has some more stuff on this
            string error_msg = string("[Nest: ] Not implemented yet");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        default: {
            string error_msg = string("[Nest: ] Unknown accumulator");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
    }
    return mem_acc;
}

}
