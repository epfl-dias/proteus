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

#include "plugins/scan-to-blocks-sm-plugin.hpp"

ScanToBlockSMPlugin::ScanToBlockSMPlugin(GpuRawContext* const context, string fnamePrefix, RecordType rec, vector<RecordAttribute*>& whichFields)
    : fnamePrefix(fnamePrefix), rec(rec), wantedFields(whichFields), context(context),
      posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr"),
      isCached(false), val_size(NULL) {
    if (wantedFields.size() == 0) {
        string error_msg{"[ScanToBlockSMPlugin: ] Invalid number of fields"};
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }

    // std::vector<Type *> parts_array;
    for (const auto &in: wantedFields){
        string fileName = fnamePrefix + "." + in->getAttrName();
        
        wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
        //FIXME: consider if address space should be global memory rather than generic
        // Type * t = PointerType::get(((const PrimitiveType *) tin)->getLLVMType(context->getLLVMContext()), /* address space */ 0);

        // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

        if (in->getOriginalType()->getTypeID() == DSTRING){
            //fetch the dictionary
            void * dict = StorageManager::getDictionaryOf(fileName);
            ((DStringType *) (in->getOriginalType()))->setDictionary(dict);
        }
    }

    Nparts = wantedFieldsFiles[0].size();
    for (size_t i = 1 ; i < wantedFields.size() ; ++i){
        size_t Nparts_loc = wantedFieldsFiles[i].size();
        if (Nparts_loc != Nparts){
            string error_msg{"[ScanToBlockSMPlugin: ] Columns do not have the same number of partitions"};
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
        }
    }

    for (const auto &in: wantedFields){
        RecordAttribute bin(*in, true);
        parts_array.emplace_back(ArrayType::get(bin.getLLVMType(context->getLLVMContext()), Nparts));
    }

    parts_arrays_type = StructType::get(context->getLLVMContext(), parts_array);
}

ScanToBlockSMPlugin::ScanToBlockSMPlugin(GpuRawContext* const context, string fnamePrefix, RecordType rec)
    : fnamePrefix(fnamePrefix), rec(rec), context(context),
      posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr"),
      isCached(false), val_size(NULL) {
    Nparts = 0;
}

ScanToBlockSMPlugin::~ScanToBlockSMPlugin() {std::cout << "freeing plugin..." << std::endl;}

void ScanToBlockSMPlugin::init()    {
}

void ScanToBlockSMPlugin::generate(const RawOperator &producer) {
    return scan(producer);
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory ScanToBlockSMPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr)   {
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
         for (it = binProjections.begin(); it != binProjections.end();
                 it++) {
             RecordAttribute attr = it->first;
             cout << attr.getRelationName() << "_" << attr.getAttrName()
                     << endl;
         }
            cout << "How many bindings? " << binProjections.size();
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        mem_projection = it->second;
    }
    return mem_projection;
}

/* FIXME Differentiate between operations that need the code and the ones needing the materialized string */
RawValueMemory ScanToBlockSMPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type) {
    return mem_value;
}

RawValue ScanToBlockSMPlugin::readCachedValue(CacheInfo info, const OperatorState& currState)   {
    return readCachedValue(info, currState.getBindings());
}

RawValue ScanToBlockSMPlugin::readCachedValue(CacheInfo info, const map<RecordAttribute, RawValueMemory>& bindings) {
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

//RawValue ScanToBlockSMPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type) {
//  IRBuilder<>* Builder = context->getBuilder();
//  RawValue value;
//  value.isNull = mem_value.isNull;
//  value.value = Builder->CreateLoad(mem_value.mem);
//  return value;
//}
RawValue ScanToBlockSMPlugin::hashValue(RawValueMemory mem_value,
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
        LOG(ERROR)<< "[ScanToBlockSMPlugin: ] String datatypes not supported yet";
        throw runtime_error(string("[ScanToBlockSMPlugin: ] String datatypes not supported yet"));
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
    LOG(ERROR) << "[ScanToBlockSMPlugin: ] Cannot contain collections";
    throw runtime_error(string("[ScanToBlockSMPlugin: ] Cannot contain collections"));
    case RECORD:
    LOG(ERROR) << "[ScanToBlockSMPlugin: ] Cannot contain record-valued attributes";
    throw runtime_error(string("[ScanToBlockSMPlugin: ] Cannot contain record-valued attributes"));
    default:
    LOG(ERROR) << "[ScanToBlockSMPlugin: ] Unknown datatype";
    throw runtime_error(string("[ScanToBlockSMPlugin: ] Unknown datatype"));
}
}

RawValue ScanToBlockSMPlugin::hashValueEager(RawValue valWrapper,
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

void ScanToBlockSMPlugin::finish()  {
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

Value* ScanToBlockSMPlugin::getValueSize(RawValueMemory mem_value,
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

void ScanToBlockSMPlugin::skipLLVM(RecordAttribute attName, Value* offset)
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

void ScanToBlockSMPlugin::nextEntry()   {

    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

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

    //Necessary because it's the itemCtr that affects the scan loop
    AllocaInst* part_i_ptr;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find("part_i_ptr");
        if (it == NamedValuesBinaryCol.end())
        {
            throw runtime_error(string("Unknown variable name: part_i_ptr"));
        }
        part_i_ptr = it->second;
    }

    //Necessary because it's the itemCtr that affects the scan loop
    AllocaInst* block_i_ptr;
    {
        map<string, AllocaInst*>::iterator it;
        it = NamedValuesBinaryCol.find("block_i_ptr");
        if (it == NamedValuesBinaryCol.end())
        {
            throw runtime_error(string("Unknown variable name: block_i_ptr"));
        }
        block_i_ptr = it->second;
    }
    //Increment and store back

    BasicBlock *wrapBB  = BasicBlock::Create(llvmContext, "incWrap", F);
    BasicBlock *stepBB  = BasicBlock::Create(llvmContext, "incStep", F);
    BasicBlock *afterBB = BasicBlock::Create(llvmContext, "incAfter", F);

    Value * part_i = Builder->CreateLoad(part_i_ptr, "part_i");
    
    IntegerType * size_type = (IntegerType *) part_i->getType();

    Value * part_N = ConstantInt::get(size_type, Nparts - 1);


    Value * cond   = Builder->CreateICmpULT(part_i, part_N);
    Builder->CreateCondBr(cond, stepBB, wrapBB);

    
    Builder->SetInsertPoint(stepBB);
    Builder->CreateStore(
        Builder->CreateAdd(
            part_i,
            ConstantInt::get(size_type, 1)
        ),
        part_i_ptr
    );

    Builder->CreateBr(afterBB);


    Builder->SetInsertPoint(wrapBB);

    Builder->CreateStore(
        ConstantInt::get(size_type, 0),
        part_i_ptr
    );

    Value * block_i = Builder->CreateLoad(block_i_ptr, "block_i");
    Builder->CreateStore(
        Builder->CreateAdd(
            block_i,
            blockSize
        ),
        block_i_ptr
    );

    Builder->CreateBr(afterBB);


    Builder->SetInsertPoint(afterBB);

    // itemCtr = block_i_ptr * Nparts + part_i_ptr * blockSize
    Value * itemCtr =   Builder->CreateAdd(
                            Builder->CreateMul(
                                Builder->CreateLoad(block_i_ptr),
                                ConstantInt::get(size_type, Nparts)
                            ),
                            Builder->CreateMul(
                                Builder->CreateLoad(part_i_ptr),
                                blockSize
                            )
                        );

    Builder->CreateStore(itemCtr, mem_itemCtr);
}

/* Operates over int*! */
void ScanToBlockSMPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
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
void ScanToBlockSMPlugin::readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
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
Value* ScanToBlockSMPlugin::readAsInt64LLVM(RecordAttribute attName)
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
void ScanToBlockSMPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
    readAsIntLLVM(attName, variables);
}

void ScanToBlockSMPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
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

void ScanToBlockSMPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
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

void ScanToBlockSMPlugin::prepareArray(RecordAttribute attName) {
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

void ScanToBlockSMPlugin::scan(const RawOperator& producer){
    LLVMContext& llvmContext = context->getLLVMContext();

    context->setGlobalFunction();

    Function* F = context->getGlobalFunction();
    IRBuilder<>* Builder = context->getBuilder();


    //Prepare
    IntegerType * size_type;
    if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty  (llvmContext);
    else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty  (llvmContext);
    else                          assert(false);

    IntegerType* int64Type                  = Type::getInt64Ty  (llvmContext);
    Type* charPtrType                       = Type::getInt8PtrTy(llvmContext);

    //Container for the variable bindings
    map<RecordAttribute, RawValueMemory> variableBindings;

    //Get the ENTRY BLOCK
    context->setCurrentEntryBlock(Builder->GetInsertBlock());

    Value * file_parts_ptr = context->CreateEntryBlockAlloca(F, "file_parts_ptr", parts_arrays_type);

    std::vector<Value *> parts_ptrs;

    // std::vector<Constant *> file_parts_init;
    for (size_t i = 0 ; i < wantedFieldsFiles.size() ; ++i){
        std::vector<Constant *> part_ptrs;

        Type * col_type     = wantedFields[i]->getLLVMType(llvmContext);
        Type * col_ptr_type = PointerType::getUnqual(col_type);

        for (const auto &t: wantedFieldsFiles[i]){
            Constant * constaddr = ConstantInt::get(int64Type, (int64_t) (t.data));
            Constant * constptr  = ConstantExpr::getIntToPtr(constaddr, col_ptr_type);

            part_ptrs.emplace_back(constptr);
        }

        // file_parts_init.emplace_back(
        Value * const_parts = ConstantArray::get(
                                ArrayType::get(
                                    RecordAttribute{*(wantedFields[i]), true}.
                                        getLLVMType(context->getLLVMContext()),
                                    Nparts
                                ),
                                part_ptrs
                            );

        parts_ptrs.emplace_back(
            context->CreateEntryBlockAlloca(
                F,
                wantedFields[i]->getAttrName() + "_parts_ptr",
                const_parts->getType()
            )
        );

        Builder->CreateStore(const_parts, parts_ptrs.back());
        // );

    }

    // Constant * file_parts = ConstantStruct::get(parts_arrays_type, file_parts_init);
    // Builder->CreateStore(file_parts, file_parts_ptr);


    ArrayType * arr_type    = ArrayType::get(int64Type, Nparts);
    Value     * N_parts_ptr = context->CreateEntryBlockAlloca(F, "N_parts_ptr"   , arr_type);
    std::vector<Constant *> N_parts_init;

    size_t max_pack_size = 0;
    for (const auto &t: wantedFieldsFiles[0]){
        size_t pack_N = t.size/context->getSizeOf(wantedFields[0]->getLLVMType(llvmContext));
        N_parts_init.push_back(context->createInt64(pack_N));
        max_pack_size = std::max(pack_N, max_pack_size);
    }

    Builder->CreateStore(
                            ConstantArray::get(arr_type, N_parts_init),
                            N_parts_ptr
                        );

    ConstantInt * zero_idx = ConstantInt::get(size_type, 0);

    AllocaInst * part_i_ptr  = context->CreateEntryBlockAlloca(F, "part_i_ptr", size_type);
    Builder->CreateStore(zero_idx, part_i_ptr);
    NamedValuesBinaryCol["part_i_ptr" ] = part_i_ptr;

    AllocaInst * block_i_ptr = context->CreateEntryBlockAlloca(F, "block_i_ptr", size_type);
    Builder->CreateStore(zero_idx,  block_i_ptr);
    NamedValuesBinaryCol["block_i_ptr"] =  block_i_ptr;

    AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(F, itemCtrVar, size_type);
    Builder->CreateStore(zero_idx, mem_itemCtr);
    NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;



    blockSize = ConstantInt::get(size_type, h_vector_size);

    BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

    // // Start insertion in CondBB.
    // Builder->SetInsertPoint(CondBB);

    // Make the new basic block for the loop header (BODY), inserting after current block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);
    BasicBlock *MainBB = BasicBlock::Create(llvmContext, "scanMain", F);

    // Make the new basic block for the increment, inserting after current block.
    BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
    context->setEndingBlock(AfterBB);

    Builder->SetInsertPoint(CondBB);
    
    // /**
    //  * Equivalent:
    //  * while(block_i < max(partsize))
    //  */

    Value * block_i  = Builder->CreateLoad(block_i_ptr, "block_i");

    Value * maxPackCnt = ConstantInt::get(size_type, max_pack_size);
    maxPackCnt->setName("maxPackCnt");

    Value *cond = Builder->CreateICmpULT(block_i, maxPackCnt);

    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    Value * part_i_loc  = Builder->CreateLoad(part_i_ptr , "part_i" );
    Value * block_i_loc = Builder->CreateLoad(block_i_ptr, "block_i");

    Value * tupleCnt = Builder->CreateLoad(
                            Builder->CreateInBoundsGEP(
                                N_parts_ptr, 
                                std::vector<Value *>{
                                    context->createInt64(0),
                                    part_i_loc
                                }
                            )
                        );

    Value * part_unfinished = Builder->CreateICmpULT(block_i_loc, tupleCnt);

    Builder->CreateCondBr(part_unfinished, MainBB, IncBB);

    Builder->SetInsertPoint(MainBB);

    //Get the 'oid' of each record and pass it along.
    //More general/lazy plugins will only perform this action,
    //instead of eagerly 'converting' fields
    //FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix,activeLoop,this->getOIDType()); //FIXME: OID type for blocks ?

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    variableBindings[tupleIdentifier] = mem_posWrapper;

    //Actual Work (Loop through attributes etc.)
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr        (*(wantedFields[i]));
        RecordAttribute block_attr  (attr, true);

        Type * ptr_t = PointerType::get(attr.getLLVMType(context->getLLVMContext()), 0);

        Value * base_ptr   = Builder->CreateInBoundsGEP(parts_ptrs[i], std::vector<Value *>{context->createInt64(0), part_i_loc});
        Value * val_bufPtr = Builder->CreateInBoundsGEP(Builder->CreateLoad(base_ptr), block_i_loc);
        val_bufPtr->setName(attr.getAttrName() + "_ptr");

        string bufVarStr = string(bufVar);
        string currBufVar = bufVarStr + "." + attr.getAttrName() + "_ptr";

        AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(F, currBufVar, ptr_t);
        Builder->CreateStore(val_bufPtr, mem_currResult);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_currResult;
        mem_valWrapper.isNull = context->createFalse();
        variableBindings[block_attr] = mem_valWrapper;
    }

    AllocaInst * blockN_ptr = context->CreateEntryBlockAlloca(F, "blockN", tupleCnt->getType());
    
    Value * remaining  = Builder->CreateSub(tupleCnt, block_i_loc);
    Value * blockN     = Builder->CreateSelect(Builder->CreateICmpULT(blockSize, remaining), blockSize, remaining);
    Builder->CreateStore(blockN, blockN_ptr);

    RecordAttribute tupCnt = RecordAttribute(fnamePrefix,"activeCnt",this->getOIDType()); //FIXME: OID type for blocks ?

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    variableBindings[tupCnt] = mem_cntWrapper;

    // // Start insertion in IncBB.
    Builder->SetInsertPoint(IncBB);
    nextEntry();
    
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(MainBB);
    
    OperatorState state{producer, variableBindings};
    producer.getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);

    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(CondBB);
    // Builder->CreateRetVoid();

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(context->getEndingBlock());
    // Builder->SetInsertPoint(AfterBB);
}



