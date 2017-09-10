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

#include "plugins/gpu-col-scan-to-blocks-plugin.hpp"

GpuColScanToBlockPlugin::GpuColScanToBlockPlugin(GpuRawContext* const context, string fnamePrefix, RecordType rec, vector<RecordAttribute*>& whichFields)
    : fnamePrefix(fnamePrefix), rec(rec), wantedFields(whichFields), context(context),
      posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr"),
      isCached(false), val_size(NULL) {
    // if (wantedFields.size() == 0) {
    //     string error_msg = string("[Binary Col Plugin]: Invalid number of fields");
    //     LOG(ERROR) << error_msg;
    //     throw runtime_error(error_msg);
    // }
}

GpuColScanToBlockPlugin::~GpuColScanToBlockPlugin() {std::cout << "freeing plugin..." << std::endl;}

void GpuColScanToBlockPlugin::init()    {
}

void GpuColScanToBlockPlugin::generate(const RawOperator &producer) {
    //Triggering parent



    data_loc loc = GPU_RESIDENT;
    for (const auto &in: wantedFields){
        const ExpressionType* tin = in->getOriginalType();
        if (!tin->isPrimitive()){
            LOG(ERROR)<< "[GpuColScanToBlockPlugin: ] Only primitive inputs are currently supported";
            throw runtime_error(string("[GpuColScanToBlockPlugin: ] Only primitive inputs are currently supported"));
        }
        
        string fileName = fnamePrefix + "." + in->getAttrName();

        wantedFieldsFiles.emplace_back(new mmap_file(fileName, loc));
        wantedFieldsWidth.emplace_back((((const PrimitiveType *) tin)->getSizeInBits(context->getLLVMContext()) + 7) / 8);

        //FIXME: consider if address space should be global memory rather than generic
        // Type * t = PointerType::get(((const PrimitiveType *) tin)->getLLVMType(context->getLLVMContext()), /* address space */ 0);

        // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));
    }

    Ntuples = 0;
    if (wantedFields.size() > 0) Ntuples = wantedFieldsFiles[0]->getFileSize() / wantedFieldsWidth[0];
    for (size_t i = 1 ; i < wantedFields.size() ; ++i){
        size_t Ntuples_loc = wantedFieldsFiles[i]->getFileSize() / wantedFieldsWidth[i];
        if (Ntuples_loc != Ntuples){
            LOG(ERROR)<< "[GpuColScanToBlockPlugin: ] Columns do not have the same number of elements";
            throw runtime_error(string("[GpuColScanToBlockPlugin: ] Columns do not have the same number of elements"));
        }
    }

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty(context->getLLVMContext());
    else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty(context->getLLVMContext());
    else                          assert(false);

    context->setGlobalFunction();

    Function* F = context->getGlobalFunction();
    IRBuilder<>* Builder = context->getBuilder();

    tupleCnt = ConstantInt::get(size_type, Ntuples);
    tupleCnt->setName("N");

    AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(F, itemCtrVar, size_type);
    Builder->CreateStore(
                ConstantInt::get(size_type, 0),
                mem_itemCtr);

    NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;

    blockSize = ConstantInt::get(size_type, h_vector_size);

    return scan(producer);
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory GpuColScanToBlockPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr)   {
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
RawValueMemory GpuColScanToBlockPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type) {
    return mem_value;
}

RawValue GpuColScanToBlockPlugin::readCachedValue(CacheInfo info, const OperatorState& currState)   {
    return readCachedValue(info, currState.getBindings());
}

RawValue GpuColScanToBlockPlugin::readCachedValue(CacheInfo info, const map<RecordAttribute, RawValueMemory>& bindings) {
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

//RawValue GpuColScanToBlockPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type) {
//  IRBuilder<>* Builder = context->getBuilder();
//  RawValue value;
//  value.isNull = mem_value.isNull;
//  value.value = Builder->CreateLoad(mem_value.mem);
//  return value;
//}
RawValue GpuColScanToBlockPlugin::hashValue(RawValueMemory mem_value,
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
        LOG(ERROR)<< "[GpuColScanToBlockPlugin: ] String datatypes not supported yet";
        throw runtime_error(string("[GpuColScanToBlockPlugin: ] String datatypes not supported yet"));
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
    LOG(ERROR) << "[GpuColScanToBlockPlugin: ] Cannot contain collections";
    throw runtime_error(string("[GpuColScanToBlockPlugin: ] Cannot contain collections"));
    case RECORD:
    LOG(ERROR) << "[GpuColScanToBlockPlugin: ] Cannot contain record-valued attributes";
    throw runtime_error(string("[GpuColScanToBlockPlugin: ] Cannot contain record-valued attributes"));
    default:
    LOG(ERROR) << "[GpuColScanToBlockPlugin: ] Unknown datatype";
    throw runtime_error(string("[GpuColScanToBlockPlugin: ] Unknown datatype"));
}
}

RawValue GpuColScanToBlockPlugin::hashValueEager(RawValue valWrapper,
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

void GpuColScanToBlockPlugin::finish()  {
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

Value* GpuColScanToBlockPlugin::getValueSize(RawValueMemory mem_value,
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

void GpuColScanToBlockPlugin::skipLLVM(RecordAttribute attName, Value* offset)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    IRBuilder<>* Builder = context->getBuilder();

    //Fetch values from symbol table
    AllocaInst* mem_pos;
    {
        map<string, AllocaInst*>::iterator it;
        string posVarStr = string(posVar);
        string currPosVar = posVarStr + "." + attName.getAttrName();
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }

    //Increment and store back
    Value* val_curr_pos = Builder->CreateLoad(mem_pos);
    Value* val_new_pos = Builder->CreateAdd(val_curr_pos,offset);
    Builder->CreateStore(val_new_pos,mem_pos);

}

void GpuColScanToBlockPlugin::nextEntry()   {

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
    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, blockSize);
    Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

/* Operates over int*! */
void GpuColScanToBlockPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int32Type = Type::getInt32Ty(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

    IRBuilder<>* Builder = context->getBuilder();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();


    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    //Fetch values from symbol table
    AllocaInst *mem_pos;
    {
        map<std::string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }
    Value *val_pos = Builder->CreateLoad(mem_pos);

    AllocaInst* buf;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
    Value *parsedInt = Builder->CreateLoad(bufShiftedPtr);

    AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
    Builder->CreateStore(parsedInt,mem_currResult);
    LOG(INFO) << "[BINARYCOL - READ INT: ] Read Successful";

    RawValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_currResult;
    mem_valWrapper.isNull = context->createFalse();
    variables[attName] = mem_valWrapper;

#ifdef DEBUGBINCOL
//      vector<Value*> ArgsV;
//      ArgsV.clear();
//      ArgsV.push_back(parsedInt);
//      Function* debugInt = context->getFunction("printi");
//      Builder->CreateCall(debugInt, ArgsV, "printi");
#endif
}

/* Operates over char*! */
void GpuColScanToBlockPlugin::readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

    IRBuilder<>* Builder = context->getBuilder();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    //Fetch values from symbol table
    AllocaInst *mem_pos;
    {
        map<std::string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }
    Value *val_pos = Builder->CreateLoad(mem_pos);

    AllocaInst* buf;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
    Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int64);
    Value *parsedInt = Builder->CreateLoad(mem_result);

    AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int64Type);
    Builder->CreateStore(parsedInt,mem_currResult);
    LOG(INFO) << "[BINARYCOL - READ INT64: ] Read Successful";

    RawValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_currResult;
    mem_valWrapper.isNull = context->createFalse();
    variables[attName] = mem_valWrapper;
}

/* Operates over char*! */
Value* GpuColScanToBlockPlugin::readAsInt64LLVM(RecordAttribute attName)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

    IRBuilder<>* Builder = context->getBuilder();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    //Fetch values from symbol table
    AllocaInst *mem_pos;
    {
        map<std::string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }
    Value *val_pos = Builder->CreateLoad(mem_pos);

    AllocaInst* buf;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
    Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int64);
    Value *parsedInt64 = Builder->CreateLoad(mem_result);

    return parsedInt64;
}


/*
 * FIXME Needs to be aware of dictionary (?).
 * Probably readValue() is the appropriate place for this.
 * I think forwarding the dict. code (int32) is sufficient here.
 */
void GpuColScanToBlockPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    readAsIntLLVM(attName, variables);
}

void GpuColScanToBlockPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* int1Type = Type::getInt1Ty(llvmContext);
    PointerType* ptrType_bool = PointerType::get(int1Type, 0);

    IRBuilder<>* Builder = context->getBuilder();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    //Fetch values from symbol table
    AllocaInst *mem_pos;
    {
        map<std::string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }
    Value *val_pos = Builder->CreateLoad(mem_pos);

    AllocaInst* buf;
    {
        map<std::string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
    Value *parsedInt = Builder->CreateLoad(bufShiftedPtr);

    AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int1Type);
    Builder->CreateStore(parsedInt,currResult);
    LOG(INFO) << "[BINARYCOL - READ BOOL: ] Read Successful";

    RawValueMemory mem_valWrapper;
    mem_valWrapper.mem = currResult;
    mem_valWrapper.isNull = context->createFalse();
    variables[attName] = mem_valWrapper;
}

void GpuColScanToBlockPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* doubleType = Type::getDoubleTy(llvmContext);
    PointerType* ptrType_double = PointerType::get(doubleType, 0);

    IRBuilder<>* Builder = context->getBuilder();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    //Fetch values from symbol table
    AllocaInst *mem_pos;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }
    Value *val_pos = Builder->CreateLoad(mem_pos);

    AllocaInst* buf;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
    Value *parsedFloat = Builder->CreateLoad(bufShiftedPtr);

    AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
    Builder->CreateStore(parsedFloat,currResult);
    LOG(INFO) << "[BINARYCOL - READ FLOAT: ] Read Successful";

    RawValueMemory mem_valWrapper;
    mem_valWrapper.mem = currResult;
    mem_valWrapper.isNull = context->createFalse();
    variables[attName] = mem_valWrapper;
}

void GpuColScanToBlockPlugin::prepareArray(RecordAttribute attName) {
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//  Type* floatPtrType = Type::getFloatPtrTy(llvmContext);
    Type* doublePtrType = Type::getDoublePtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    Type* int32PtrType = Type::getInt32PtrTy(llvmContext);
    Type* int64PtrType = Type::getInt64PtrTy(llvmContext);
    Type* int8PtrType = Type::getInt8PtrTy(llvmContext);

    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

    string posVarStr = string(posVar);
    string currPosVar = posVarStr + "." + attName.getAttrName();
    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attName.getAttrName();

    /* Code equivalent to skip(size_t) */
    Value* val_offset = context->createInt64(sizeof(size_t));
    AllocaInst* mem_pos;
    {
        map<string, AllocaInst*>::iterator it;
        string posVarStr = string(posVar);
        string currPosVar = posVarStr + "." + attName.getAttrName();
        it = NamedValuesBinaryCol.find(currPosVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currPosVar);
        }
        mem_pos = it->second;
    }

    //Increment and store back
    Value* val_curr_pos = Builder->CreateLoad(mem_pos);
    Value* val_new_pos = Builder->CreateAdd(val_curr_pos,val_offset);
    /* Not storing this 'offset' - we want the cast buffer to
     * conceptually start from 0 */
    //  Builder->CreateStore(val_new_pos,mem_pos);

    /* Get relevant char* rawBuf */
    AllocaInst* buf;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find(currBufVar);
        if (it == NamedValuesBinaryCol.end()) {
            throw runtime_error(string("Unknown variable name: ") + currBufVar);
        }
        buf = it->second;
    }
    Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_new_pos);

    /* Cast to appropriate form */
    typeID id = attName.getOriginalType()->getTypeID();
    switch (id) {
    case BOOL: {
        //No need to do sth - char* and int8* are interchangeable
        break;
    }
    case FLOAT: {
        AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
                string("mem_bufPtr"), doublePtrType);
        Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, doublePtrType);
        Builder->CreateStore(val_bufPtr, mem_bufPtr);
        NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
        break;
    }
    case INT: {
        AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
                string("mem_bufPtr"), int32PtrType);
        Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, int32PtrType);
        Builder->CreateStore(val_bufPtr, mem_bufPtr);
        NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
        break;
    }
    case INT64: {
            AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
                    string("mem_bufPtr"), int64PtrType);
            Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, int64PtrType);
            Builder->CreateStore(val_bufPtr, mem_bufPtr);
            NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
            break;
        }
    case STRING: {
        /* String representation comprises the code and the dictionary
         * Codes are (will be) int32, so can again treat like int32* */
        AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
                string("mem_bufPtr"), int32PtrType);
        Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, int32PtrType);
        Builder->CreateStore(val_bufPtr, mem_bufPtr);
        NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
        break;
    }
    case RECORD:
    case LIST:
    case BAG:
    case SET:
    default: {
        string error_msg = string("[Binary Col PG: ] Unsupported Record");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
    }
}

void GpuColScanToBlockPlugin::scan(const RawOperator& producer)
{
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);

    //Container for the variable bindings
    map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();

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
    Value *lhs  = Builder->CreateLoad(mem_itemCtr, "i");
    Value *rhs  = tupleCnt;
    
    Value *cond = Builder->CreateICmpSLT(lhs, rhs);

    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    //Get the 'oid' of each record and pass it along.
    //More general/lazy plugins will only perform this action,
    //instead of eagerly 'converting' fields
    //FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix,activeLoop,this->getOIDType()); //FIXME: OID type for blocks ?

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    (*variableBindings)[tupleIdentifier] = mem_posWrapper;

    // Type * kernel_params_type = ArrayType::get(charPtrType, wantedFields.size() + 2); //input + N + state

    // Value * kernel_params      = UndefValue::get(kernel_params_type);
    // Value * kernel_params_addr = context->CreateEntryBlockAlloca(F, "gpu_params", kernel_params_type);

    //Actual Work (Loop through attributes etc.)
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr        (*(wantedFields[i]));
        RecordAttribute block_attr  (attr, true);

        // size_t offset = 0;

        /* Read wanted field.
         * Reminder: Primitive, scalar types have the (raw) buffer
         * already cast to appr. type*/
        if (!attr.getOriginalType()->isPrimitive()){
            LOG(ERROR)<< "[BINARY COL PLUGIN: ] Only primitives are currently supported";
            throw runtime_error(string("[BINARY COL PLUGIN: ] Only primitives are currently supported"));
        }

        const PrimitiveType * t = dynamic_cast<const PrimitiveType *>(attr.getOriginalType());

        Type * ptr_t = PointerType::get(t->getLLVMType(context->getLLVMContext()), 0);

        Value * val_bufPtr = ConstantInt::get(llvmContext, APInt(64, ((uint64_t) wantedFieldsFiles[i]->getData())));
        
        Value * arg        = Builder->CreateIntToPtr(val_bufPtr, ptr_t);
        arg->setName(attr.getAttrName() + "_ptr");


        string bufVarStr = string(bufVar);
        string currBufVar = bufVarStr + "." + attr.getAttrName() + "_ptr";

        Value *parsed = Builder->CreateGEP(arg, lhs);

        AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(F, currBufVar, ptr_t);
        Builder->CreateStore(parsed, mem_currResult);

        // kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateBitCast(mem_currResult, charPtrType), i);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_currResult;
        mem_valWrapper.isNull = context->createFalse();
        (*variableBindings)[block_attr] = mem_valWrapper;
    }

    AllocaInst * blockN_ptr = context->CreateEntryBlockAlloca(F, "blockN", tupleCnt->getType());
    
    Value * remaining  = Builder->CreateSub(tupleCnt, lhs);
    Value * blockN     = Builder->CreateSelect(Builder->CreateICmpSLT(blockSize, remaining), blockSize, remaining);
    Builder->CreateStore(blockN, blockN_ptr);
    
    RecordAttribute tupCnt = RecordAttribute(fnamePrefix,"activeCnt",this->getOIDType()); //FIXME: OID type for blocks ?

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    (*variableBindings)[tupCnt] = mem_cntWrapper;


    OperatorState* state = new OperatorState(producer, *variableBindings);
    producer.getParent()->consume(context,*state);

    // Builder->CreateStore(blockN, blockN_ptr);

    // kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateBitCast(blockN_ptr, charPtrType), wantedFields.size()    );

    // Value * subState   = Builder->CreateLoad(context->getSubStateVar(), "subState");

    // kernel_params = Builder->CreateInsertValue(kernel_params, subState, wantedFields.size() + 1);

    // Builder->CreateStore(kernel_params, kernel_params_addr);

    // Function * launchk = context->getFunction("launch_kernel");

    // Type  * ptr_t = PointerType::get(charPtrType, 0);

    // Value * entryPtr = ConstantInt::get(llvmContext, APInt(64, ((uint64_t) entry_point)));
    
    // Value * entry    = Builder->CreateIntToPtr(entryPtr, charPtrType);

    // vector<Value *> kernel_args{entry, Builder->CreateBitCast(kernel_params_addr, PointerType::get(charPtrType, 0))};
    
    // Builder->CreateCall(launchk, kernel_args);

    // Start insertion in IncBB.
    Builder->SetInsertPoint(IncBB);
    nextEntry();
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(LoopBB);

                                                                                                            // //Triggering parent
                                                                                                            // OperatorState* state = new OperatorState(producer, *variableBindings);
                                                                                                            // RawOperator* const opParent = producer.getParent();
                                                                                                            // opParent->consume(context,*state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(CondBB);


    Builder->SetInsertPoint(context->getEndingBlock());

    Builder->CreateRetVoid();

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(AfterBB);
}



