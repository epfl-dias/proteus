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

#include "plugins/gpu-col-scan-plugin.hpp"

GpuColScanPlugin::GpuColScanPlugin(GpuRawContext* const context, string fnamePrefix, RecordType rec, vector<RecordAttribute*>& whichFields, RawOperator* const child)
    : fnamePrefix(fnamePrefix), rec(rec), wantedFields(whichFields), context(context),
      posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr"),
      isCached(false), val_size(NULL), child(child) {
    // if (wantedFields.size() == 0) {
    //     string error_msg = string("[Binary Col Plugin]: Invalid number of fields");
    //     LOG(ERROR) << error_msg;
    //     throw runtime_error(error_msg);
    // }
}

GpuColScanPlugin::~GpuColScanPlugin() {}

void GpuColScanPlugin::init()    {
    for (const auto &in: wantedFields){
        const ExpressionType* tin = in->getOriginalType();
        if (!tin->isPrimitive()){
            LOG(ERROR)<< "[GpuColScanPlugin: ] Only primitive inputs are currently supported";
            throw runtime_error(string("[GpuColScanPlugin: ] Only primitive inputs are currently supported"));
        }
        
        //FIXME: consider if address space should be global memory rather than generic
        Type * t = PointerType::get(((const PrimitiveType *) tin)->getLLVMType(context->getLLVMContext()), /* address space */ 0);

        wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));
    }

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty(context->getLLVMContext());
    else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty(context->getLLVMContext());
    else                          assert(false);

    tupleCntArg_id = context->appendParameter(size_type, false, false);

    context->setGlobalFunction(true);

    Function* F = context->getGlobalFunction();
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    PointerType* int64PtrType = Type::getInt64PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    IRBuilder<>* Builder = context->getBuilder();


    AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(F, itemCtrVar, int64Type);
    Builder->CreateStore(
                Builder->CreateIntCast(context->threadId(),
                                        int64Type,
                                        false),
                mem_itemCtr);

    NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;
}

void GpuColScanPlugin::generate(const RawOperator &producer) {
    scan(producer);
    if (child) child->produce();
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory GpuColScanPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr)   {
    RawValueMemory mem_projection;
    {
        const OperatorState* state = bindings.state;
        const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
        //XXX Make sure that using fnamePrefix in this search does not cause issues
        RecordAttribute tmpKey = RecordAttribute(fnamePrefix,pathVar,this->getOIDType());
        map<RecordAttribute, RawValueMemory>::const_iterator it;
        it = binProjections.find(tmpKey);
        if (it == binProjections.end()) {
            string error_msg = string(
                    "[Binary Col. plugin - readPath ]: Unknown variable name ")
                    + pathVar;
//          for (it = binProjections.begin(); it != binProjections.end();
//                  it++) {
//              RecordAttribute attr = it->first;
//              cout << attr.getRelationName() << "_" << attr.getAttrName()
//                      << endl;
//          }
            cout << "How many bindings? " << binProjections.size();
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        mem_projection = it->second;
    }
    return mem_projection;
}

/* FIXME Differentiate between operations that need the code and the ones needing the materialized string */
RawValueMemory GpuColScanPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type) {
    return mem_value;
}

RawValue GpuColScanPlugin::readCachedValue(CacheInfo info, const OperatorState& currState)   {
    return readCachedValue(info, currState.getBindings());
}

RawValue GpuColScanPlugin::readCachedValue(CacheInfo info, const map<RecordAttribute, RawValueMemory>& bindings) {
    IRBuilder<>* const Builder = context->getBuilder();
    Function *F = context->getGlobalFunction();

    /* Need OID to retrieve corresponding value from bin. cache */
    RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix, activeLoop,
            getOIDType());
    map<RecordAttribute, RawValueMemory>::const_iterator it =
            bindings.find(tupleIdentifier);
    if (it == bindings.end()) {
        string error_msg =
                "[Expression Generator: ] Current tuple binding / OID not found";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
    RawValueMemory mem_oidWrapper = it->second;
    /* OID is a plain integer - starts from 1!!! */
    Value *val_oid = Builder->CreateLoad(mem_oidWrapper.mem);
    val_oid = Builder->CreateSub(val_oid,context->createInt64(1));

    /* Need to find appropriate position in cache now -- should be OK(?) */

    StructType *cacheType = context->ReproduceCustomStruct(info.objectTypes);
    Value *typeSize = ConstantExpr::getSizeOf(cacheType);
    char* rawPtr = *(info.payloadPtr);
    int posInStruct = info.structFieldNo;

    /* Cast to appr. type */
    PointerType *ptr_cacheType = PointerType::get(cacheType, 0);
    Value *val_cachePtr = context->CastPtrToLlvmPtr(ptr_cacheType, rawPtr);

    Value *val_cacheShiftedPtr = context->getArrayElemMem(val_cachePtr,
            val_oid);
//  val_cacheShiftedPtr->getType()->dump();
//  cout << "Pos in struct? " << posInStruct << endl;
//  Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
//          posInStruct - 1);

    // XXX
    // -1 because bin files has no OID (?)
    Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
            posInStruct/* - 1*/);
    Type *fieldType = val_cachedField->getType();

    /* This Alloca should not appear in optimized code */
    AllocaInst *mem_cachedField = context->CreateEntryBlockAlloca(F,
            "tmpCachedField", fieldType);
    Builder->CreateStore(val_cachedField, mem_cachedField);

    RawValue valWrapper;
    valWrapper.value = Builder->CreateLoad(mem_cachedField);
    valWrapper.isNull = context->createFalse();
#ifdef DEBUG
    {
        vector<Value*> ArgsV;

        Function* debugSth = context->getFunction("printi64");
        ArgsV.push_back(val_oid);
        Builder->CreateCall(debugSth, ArgsV);
    }
#endif
    return valWrapper;
}

//RawValue GpuColScanPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type) {
//  IRBuilder<>* Builder = context->getBuilder();
//  RawValue value;
//  value.isNull = mem_value.isNull;
//  value.value = Builder->CreateLoad(mem_value.mem);
//  return value;
//}
RawValue GpuColScanPlugin::hashValue(RawValueMemory mem_value,
        const ExpressionType* type) {
    IRBuilder<>* Builder = context->getBuilder();
    switch (type->getTypeID()) {
    case BOOL: {
        Function *hashBoolean = context->getFunction("hashBoolean");
        vector<Value*> ArgsV;
        ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
        Value *hashResult = context->getBuilder()->CreateCall(hashBoolean,
                ArgsV, "hashBoolean");

        RawValue valWrapper;
        valWrapper.value = hashResult;
        valWrapper.isNull = context->createFalse();
        return valWrapper;
    }
    case STRING: {
        LOG(ERROR)<< "[GpuColScanPlugin: ] String datatypes not supported yet";
        throw runtime_error(string("[GpuColScanPlugin: ] String datatypes not supported yet"));
    }
    case FLOAT:
    {
        Function *hashDouble = context->getFunction("hashDouble");
        vector<Value*> ArgsV;
        ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
        Value *hashResult = context->getBuilder()->CreateCall(hashDouble, ArgsV, "hashDouble");

        RawValue valWrapper;
        valWrapper.value = hashResult;
        valWrapper.isNull = context->createFalse();
        return valWrapper;
    }
    case INT:
    {
        Function *hashInt = context->getFunction("hashInt");
        vector<Value*> ArgsV;
        ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
        Value *hashResult = context->getBuilder()->CreateCall(hashInt, ArgsV, "hashInt");

        RawValue valWrapper;
        valWrapper.value = hashResult;
        valWrapper.isNull = context->createFalse();
        return valWrapper;
    }
    case BAG:
    case LIST:
    case SET:
    LOG(ERROR) << "[GpuColScanPlugin: ] Cannot contain collections";
    throw runtime_error(string("[GpuColScanPlugin: ] Cannot contain collections"));
    case RECORD:
    LOG(ERROR) << "[GpuColScanPlugin: ] Cannot contain record-valued attributes";
    throw runtime_error(string("[GpuColScanPlugin: ] Cannot contain record-valued attributes"));
    default:
    LOG(ERROR) << "[GpuColScanPlugin: ] Unknown datatype";
    throw runtime_error(string("[GpuColScanPlugin: ] Unknown datatype"));
}
}

RawValue GpuColScanPlugin::hashValueEager(RawValue valWrapper,
        const ExpressionType* type) {
    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();
    Value *tmp = valWrapper.value;
    AllocaInst *mem_tmp = context->CreateEntryBlockAlloca(F, "mem_cachedToHash",
            tmp->getType());
    Builder->CreateStore(tmp, mem_tmp);
    RawValueMemory mem_tmpWrapper = { mem_tmp, valWrapper.isNull };
    return hashValue(mem_tmpWrapper, type);
}

void GpuColScanPlugin::finish()  {
    vector<RecordAttribute*>::iterator it;
    int cnt = 0;
    for (it = wantedFields.begin(); it != wantedFields.end(); it++) {
        close(fd[cnt]);
        munmap(buf[cnt], colFilesize[cnt]);

        if ((*it)->getOriginalType()->getTypeID() == STRING)    {
            int dictionaryFd = dictionaries[cnt];
            close(dictionaryFd);
            munmap(dictionariesBuf[cnt], dictionaryFilesizes[cnt]);
        }
        cnt++;
    }
}

Value* GpuColScanPlugin::getValueSize(RawValueMemory mem_value,
        const ExpressionType* type) {
    switch (type->getTypeID()) {
    case BOOL:
    case INT:
    case FLOAT: {
        Type *explicitType = (mem_value.mem)->getAllocatedType();
        return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
    }
    case STRING: {
        string error_msg = string(
                "[Binary Col Plugin]: Strings not supported yet");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
    case BAG:
    case LIST:
    case SET: {
        string error_msg = string(
                "[Binary Col Plugin]: Cannot contain collections");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
    case RECORD: {
        string error_msg = string(
                "[Binary Col Plugin]: Cannot contain records");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
    default: {
        string error_msg = string("[Binary Col Plugin]: Unknown datatype");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    }
}

void GpuColScanPlugin::nextEntry()   {

    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    IRBuilder<>* Builder = context->getBuilder();

    //Necessary because it's the itemCtr that affects the scan loop
    AllocaInst* mem_itemCtr;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(itemCtrVar);
        if (it == NamedValuesBinaryCol.end())
        {
            throw runtime_error(string("Unknown variable name: ") + itemCtrVar);
        }
        mem_itemCtr = it->second;
    }

    //Increment and store back

    Value* val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);
    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr,
        Builder->CreateIntCast(context->threadNum(), val_curr_itemCtr->getType(), false));
    Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

void GpuColScanPlugin::scan(const RawOperator& producer){
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);

    //Container for the variable bindings
    map<RecordAttribute, RawValueMemory> variableBindings;

    //Get the ENTRY BLOCK
    context->setCurrentEntryBlock(Builder->GetInsertBlock());

    BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

    // // Start insertion in CondBB.
    // Builder->SetInsertPoint(CondBB);

    // Make the new basic block for the loop header (BODY), inserting after current block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);

    // Make the new basic block for the increment, inserting after current block.
    BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
    context->setEndingBlock(AfterBB);

    Builder->SetInsertPoint(CondBB);

    /**
     * Equivalent:
     * while(itemCtr < size)
     */
    AllocaInst *mem_itemCtr = NamedValuesBinaryCol[itemCtrVar];
    Value    *lhs = Builder->CreateLoad(mem_itemCtr, "i");
    Argument *rhs = context->getArgument(tupleCntArg_id);
    rhs->setName("cnt");
    
    Value   *cond = Builder->CreateICmpSLT(lhs, rhs);

    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    //Get the 'oid' of each record and pass it along.
    //More general/lazy plugins will only perform this action,
    //instead of eagerly 'converting' fields
    //FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix,activeLoop,this->getOIDType());

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    variableBindings[tupleIdentifier] = mem_posWrapper;

    //Actual Work (Loop through attributes etc.)
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr = *(wantedFields[i]);

        Argument * arg = context->getArgument(wantedFieldsArg_id[i]);
        arg->setName(attr.getAttrName() + "_ptr");

        // size_t offset = 0;

        /* Read wanted field.
         * Reminder: Primitive, scalar types have the (raw) buffer
         * already cast to appr. type*/
        if (!attr.getOriginalType()->isPrimitive()){
            LOG(ERROR)<< "[BINARY COL PLUGIN: ] Only primitives are currently supported";
            throw runtime_error(string("[BINARY COL PLUGIN: ] Only primitives are currently supported"));
        }

        const PrimitiveType * t = dynamic_cast<const PrimitiveType *>(attr.getOriginalType());

        // AllocaInst * attr_alloca = context->CreateEntryBlockAlloca(F, attr.getAttrName(), t->getLLVMType(llvmContext));

        // variableBindings[tupleIdentifier] = mem_posWrapper;
        //Move to next position
        // Value* val_offset = context->createInt64(offset);
        // skipLLVM(attr, val_offset);



        // string posVarStr = string(posVar);
        // string currPosVar = posVarStr + "." + attr.getAttrName();
        string bufVarStr = string(bufVar);
        string currBufVar = bufVarStr + "." + attr.getAttrName();

        // Value *parsed = Builder->CreateLoad(bufShiftedPtr); //attr_alloca
        Value *parsed = Builder->CreateLoad(Builder->CreateGEP(arg, lhs)); //TODO : use CreateAllignedLoad 

        AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(F, currBufVar, t->getLLVMType(llvmContext));
        Builder->CreateStore(parsed, mem_currResult);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_currResult;
        mem_valWrapper.isNull = context->createFalse();
        variableBindings[attr] = mem_valWrapper;
    }

    // Start insertion in IncBB.
    Builder->SetInsertPoint(IncBB);
    nextEntry();
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(LoopBB);

    //Triggering parent
    OperatorState state{producer, variableBindings};
    producer.getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(CondBB);

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(context->getEndingBlock());
}



