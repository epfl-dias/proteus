/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#include "olap-pipeline.hpp"

using namespace llvm;
using std::vector;

static void registerOlapFunctions(PipelineGen *pipelineGen) {
  Module *TheModule = pipelineGen->getModule();
  LLVMContext &ctx = TheModule->getContext();
  assert(TheModule != nullptr);

  Type *int1_bool_type = Type::getInt1Ty(ctx);
  Type *int8_type = Type::getInt8Ty(ctx);
  Type *int32_type = Type::getInt32Ty(ctx);
  Type *int64_type = Type::getInt64Ty(ctx);
  // not 100% portable, but we only run on 64bit architectures
  Type *uintptr_type = Type::getInt64Ty(ctx);
  Type *void_type = Type::getVoidTy(ctx);
  Type *double_type = Type::getDoubleTy(ctx);
  StructType *strObjType = Context::CreateStringStruct(ctx);
  PointerType *void_ptr_type = PointerType::get(int8_type, 0);
  PointerType *char_ptr_type = PointerType::get(int8_type, 0);
  PointerType *int32_ptr_type = PointerType::get(int32_type, 0);

  /**
   * Args of functions computing flush
   */
  vector<Type *> ArgsFlushInt;
  ArgsFlushInt.insert(ArgsFlushInt.begin(), char_ptr_type);
  ArgsFlushInt.insert(ArgsFlushInt.begin(), int32_type);

  vector<Type *> ArgsFlushDString;
  ArgsFlushDString.insert(ArgsFlushDString.begin(), char_ptr_type);
  ArgsFlushDString.insert(ArgsFlushDString.begin(), char_ptr_type);
  ArgsFlushDString.insert(ArgsFlushDString.begin(), int32_type);

  vector<Type *> ArgsFlushInt64;
  ArgsFlushInt64.insert(ArgsFlushInt64.begin(), char_ptr_type);
  ArgsFlushInt64.insert(ArgsFlushInt64.begin(), int64_type);

  vector<Type *> ArgsFlushDate;
  ArgsFlushDate.insert(ArgsFlushDate.begin(), char_ptr_type);
  ArgsFlushDate.insert(ArgsFlushDate.begin(), int64_type);

  vector<Type *> ArgsFlushDouble;
  ArgsFlushDouble.insert(ArgsFlushDouble.begin(), char_ptr_type);
  ArgsFlushDouble.insert(ArgsFlushDouble.begin(), double_type);

  vector<Type *> ArgsFlushStringC;
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), char_ptr_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), int64_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), int64_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), char_ptr_type);

  vector<Type *> ArgsFlushStringCv2;
  ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(), char_ptr_type);
  ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(), char_ptr_type);

  vector<Type *> ArgsFlushStringObj;
  ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(), char_ptr_type);
  ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(), strObjType);

  vector<Type *> ArgsFlushBoolean;
  ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(), char_ptr_type);
  ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(), int1_bool_type);

  vector<Type *> ArgsFlushPtr;
  ArgsFlushPtr.insert(ArgsFlushPtr.begin(), char_ptr_type);
  ArgsFlushPtr.insert(ArgsFlushPtr.begin(), uintptr_type);

  vector<Type *> ArgsFlushStartEnd;
  ArgsFlushStartEnd.insert(ArgsFlushStartEnd.begin(), char_ptr_type);

  vector<Type *> ArgsFlushChar;
  ArgsFlushChar.insert(ArgsFlushChar.begin(), char_ptr_type);
  ArgsFlushChar.insert(ArgsFlushChar.begin(), int8_type);

  vector<Type *> ArgsFlushDelim;
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), char_ptr_type);
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), int8_type);
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), int64_type);

  vector<Type *> ArgsFlushOutput;
  ArgsFlushOutput.insert(ArgsFlushOutput.begin(), char_ptr_type);

  FunctionType *FTflushInt = FunctionType::get(void_type, ArgsFlushInt, false);
  FunctionType *FTflushDString =
      FunctionType::get(void_type, ArgsFlushDString, false);
  FunctionType *FTflushInt64 =
      FunctionType::get(void_type, ArgsFlushInt64, false);
  FunctionType *FTflushDate =
      FunctionType::get(void_type, ArgsFlushDate, false);
  FunctionType *FTflushDouble =
      FunctionType::get(void_type, ArgsFlushDouble, false);
  FunctionType *FTflushStringC =
      FunctionType::get(void_type, ArgsFlushStringC, false);
  FunctionType *FTflushStringCv2 =
      FunctionType::get(void_type, ArgsFlushStringCv2, false);
  FunctionType *FTflushStringObj =
      FunctionType::get(void_type, ArgsFlushStringObj, false);
  FunctionType *FTflushBoolean =
      FunctionType::get(void_type, ArgsFlushBoolean, false);
  FunctionType *FTflushPtr = FunctionType::get(void_type, ArgsFlushPtr, false);
  FunctionType *FTflushStartEnd =
      FunctionType::get(void_type, ArgsFlushStartEnd, false);
  FunctionType *FTflushChar =
      FunctionType::get(void_type, ArgsFlushChar, false);
  FunctionType *FTflushDelim =
      FunctionType::get(void_type, ArgsFlushDelim, false);
  FunctionType *FTflushOutput =
      FunctionType::get(void_type, ArgsFlushOutput, false);

  /**
   * Flushing
   */
  Function *flushInt_ = Function::Create(FTflushInt, Function::ExternalLinkage,
                                         "flushInt", TheModule);
  Function *flushDString_ = Function::Create(
      FTflushDString, Function::ExternalLinkage, "flushDString", TheModule);
  Function *flushInt64_ = Function::Create(
      FTflushInt64, Function::ExternalLinkage, "flushInt64", TheModule);
  Function *flushDate_ = Function::Create(
      FTflushDate, Function::ExternalLinkage, "flushDate", TheModule);
  Function *flushDouble_ = Function::Create(
      FTflushDouble, Function::ExternalLinkage, "flushDouble", TheModule);
  Function *flushStringC_ = Function::Create(
      FTflushStringC, Function::ExternalLinkage, "flushStringC", TheModule);
  Function *flushStringCv2_ =
      Function::Create(FTflushStringCv2, Function::ExternalLinkage,
                       "flushStringReady", TheModule);
  Function *flushStringObj_ =
      Function::Create(FTflushStringObj, Function::ExternalLinkage,
                       "flushStringObject", TheModule);
  Function *flushBoolean_ = Function::Create(
      FTflushBoolean, Function::ExternalLinkage, "flushBoolean", TheModule);
  Function *flushPtr_ = Function::Create(FTflushPtr, Function::ExternalLinkage,
                                         "flushPtr", TheModule);
  Function *flushObjectStart_ =
      Function::Create(FTflushStartEnd, Function::ExternalLinkage,
                       "flushObjectStart", TheModule);
  Function *flushArrayStart_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushArrayStart", TheModule);
  Function *flushObjectEnd_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushObjectEnd", TheModule);
  Function *flushArrayEnd_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushArrayEnd", TheModule);
  Function *flushChar_ = Function::Create(
      FTflushChar, Function::ExternalLinkage, "flushChar", TheModule);
  Function *flushDelim_ = Function::Create(
      FTflushDelim, Function::ExternalLinkage, "flushDelim", TheModule);
  Function *flushOutput_ = Function::Create(
      FTflushOutput, Function::ExternalLinkage, "flushOutput", TheModule);

  /**
   * HASHTABLES FOR JOINS / AGGREGATIONS
   */
  // Last type is needed to capture file size. Tentative
  Type *ht_int_types[] = {int32_type, int32_type, void_ptr_type, int32_type};
  FunctionType *FTintHT = FunctionType::get(void_type, ht_int_types, false);
  Function *insertIntKeyToHT_ = Function::Create(
      FTintHT, Function::ExternalLinkage, "insertIntKeyToHT", TheModule);

  Type *ht_types[] = {char_ptr_type, int64_type, void_ptr_type, int32_type};
  FunctionType *FT_HT = FunctionType::get(void_type, ht_types, false);
  Function *insertToHT_ = Function::Create(FT_HT, Function::ExternalLinkage,
                                           "insertToHT", TheModule);

  Type *ht_int_probe_types[] = {int32_type, int32_type, int32_type};
  PointerType *void_ptr_ptr_type = PointerType::get(void_ptr_type, 0);
  FunctionType *FTint_probeHT =
      FunctionType::get(void_ptr_ptr_type, ht_int_probe_types, false);
  Function *probeIntHT_ = Function::Create(
      FTint_probeHT, Function::ExternalLinkage, "probeIntHT", TheModule);
  probeIntHT_->addFnAttr(llvm::Attribute::AlwaysInline);

  Type *ht_probe_types[] = {char_ptr_type, int64_type};
  FunctionType *FT_probeHT =
      FunctionType::get(void_ptr_ptr_type, ht_probe_types, false);
  Function *probeHT_ = Function::Create(FT_probeHT, Function::ExternalLinkage,
                                        "probeHT", TheModule);
  probeHT_->addFnAttr(llvm::Attribute::AlwaysInline);

  Type *ht_get_metadata_types[] = {char_ptr_type};
  StructType *metadataType = Context::getHashtableMetadataType(ctx);
  PointerType *metadataArrayType = PointerType::get(metadataType, 0);
  FunctionType *FTget_metadata_HT =
      FunctionType::get(metadataArrayType, ht_get_metadata_types, false);
  Function *getMetadataHT_ = Function::Create(
      FTget_metadata_HT, Function::ExternalLinkage, "getMetadataHT", TheModule);

  /**
   * Radix
   */
  /* What the type of HT buckets is */
  vector<Type *> htBucketMembers;
  // int *bucket;
  htBucketMembers.push_back(int32_ptr_type);
  // int *next;
  htBucketMembers.push_back(int32_ptr_type);
  // uint32_t mask;
  htBucketMembers.push_back(int32_type);
  // int count;
  htBucketMembers.push_back(int32_type);
  StructType *htBucketType = StructType::get(ctx, htBucketMembers);
  PointerType *htBucketPtrType = PointerType::get(htBucketType, 0);

  /* JOIN!!! */
  /* What the type of HT entries is */
  /* (int32, void*) */
  vector<Type *> htEntryMembers;
  htEntryMembers.push_back(int32_type);
  htEntryMembers.push_back(int64_type);
  StructType *htEntryType = StructType::get(ctx, htEntryMembers);
  PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

  Type *radix_partition_types[] = {int64_type, htEntryPtrType};
  FunctionType *FTradix_partition =
      FunctionType::get(int32_ptr_type, radix_partition_types, false);
  Function *radix_partition =
      Function::Create(FTradix_partition, Function::ExternalLinkage,
                       "partitionHTLLVM", TheModule);

  Type *bucket_chaining_join_prepare_types[] = {htEntryPtrType, int32_type,
                                                htBucketPtrType};
  FunctionType *FTbucket_chaining_join_prepare =
      FunctionType::get(void_type, bucket_chaining_join_prepare_types, false);
  Function *bucket_chaining_join_prepare = Function::Create(
      FTbucket_chaining_join_prepare, Function::ExternalLinkage,
      "bucket_chaining_join_prepareLLVM", TheModule);

  /* AGGR! */
  /* What the type of HT entries is */
  /* (int64, void*) */
  vector<Type *> htAggEntryMembers;
  htAggEntryMembers.push_back(int64_type);
  htAggEntryMembers.push_back(int64_type);
  StructType *htAggEntryType = StructType::get(ctx, htAggEntryMembers);
  PointerType *htAggEntryPtrType = PointerType::get(htAggEntryType, 0);
  Type *radix_partition_agg_types[] = {int64_type, htAggEntryPtrType};
  FunctionType *FTradix_partition_agg =
      FunctionType::get(int32_ptr_type, radix_partition_agg_types, false);
  Function *radix_partition_agg =
      Function::Create(FTradix_partition_agg, Function::ExternalLinkage,
                       "partitionAggHTLLVM", TheModule);

  Type *bucket_chaining_agg_prepare_types[] = {htAggEntryPtrType, int32_type,
                                               htBucketPtrType};
  FunctionType *FTbucket_chaining_agg_prepare =
      FunctionType::get(void_type, bucket_chaining_agg_prepare_types, false);
  Function *bucket_chaining_agg_prepare =
      Function::Create(FTbucket_chaining_agg_prepare, Function::ExternalLinkage,
                       "bucket_chaining_agg_prepareLLVM", TheModule);
  /**
   * End of Radix
   */

  pipelineGen->registerFunction("insertInt", insertIntKeyToHT_);
  pipelineGen->registerFunction("probeInt", probeIntHT_);
  pipelineGen->registerFunction("insertHT", insertToHT_);
  pipelineGen->registerFunction("probeHT", probeHT_);
  pipelineGen->registerFunction("getMetadataHT", getMetadataHT_);

  pipelineGen->registerFunction("flushInt", flushInt_);
  pipelineGen->registerFunction("flushDString", flushDString_);
  pipelineGen->registerFunction("flushInt64", flushInt64_);
  pipelineGen->registerFunction("flushDate", flushDate_);
  pipelineGen->registerFunction("flushDouble", flushDouble_);
  pipelineGen->registerFunction("flushStringC", flushStringC_);
  pipelineGen->registerFunction("flushStringCv2", flushStringCv2_);
  pipelineGen->registerFunction("flushStringObj", flushStringObj_);
  pipelineGen->registerFunction("flushBoolean", flushBoolean_);
  pipelineGen->registerFunction("flushPtr", flushPtr_);
  pipelineGen->registerFunction("flushChar", flushChar_);
  pipelineGen->registerFunction("flushDelim", flushDelim_);
  pipelineGen->registerFunction("flushOutput", flushOutput_);

  pipelineGen->registerFunction("flushObjectStart", flushObjectStart_);
  pipelineGen->registerFunction("flushArrayStart", flushArrayStart_);
  pipelineGen->registerFunction("flushObjectEnd", flushObjectEnd_);
  pipelineGen->registerFunction("flushArrayEnd", flushArrayEnd_);
  pipelineGen->registerFunction("flushArrayEnd", flushArrayEnd_);

  pipelineGen->registerFunction("partitionHT", radix_partition);
  pipelineGen->registerFunction("bucketChainingPrepare",
                                bucket_chaining_join_prepare);
  pipelineGen->registerFunction("partitionAggHT", radix_partition_agg);
  pipelineGen->registerFunction("bucketChainingAggPrepare",
                                bucket_chaining_agg_prepare);
}

void OlapCpuPipelineGenFactory::registerFunctions(PipelineGen *pipelineGen) {
  registerOlapFunctions(pipelineGen);

  auto *llvmModule = pipelineGen->getModule();
  Type *void_type = Type::getVoidTy(llvmModule->getContext());
  Type *int32PtrType = Type::getInt32PtrTy(llvmModule->getContext());
  Type *charPtrType = Type::getInt8PtrTy(llvmModule->getContext());

  FunctionType *step_mmc_mem_move_broadcast_device =
      FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
  Function *fstep_mmc_mem_move_broadcast_device = Function::Create(
      step_mmc_mem_move_broadcast_device, Function::ExternalLinkage,
      "step_mmc_mem_move_broadcast_device", llvmModule);
  pipelineGen->registerFunction("step_mmc_mem_move_broadcast_device",
                                fstep_mmc_mem_move_broadcast_device);

  FunctionType *getClusterCounts = FunctionType::get(
      int32PtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetClusterCounts =
      Function::Create(getClusterCounts, Function::ExternalLinkage,
                       "getClusterCounts", llvmModule);
  pipelineGen->registerFunction("getClusterCounts", fgetClusterCounts);

  FunctionType *getRelationMem = FunctionType::get(
      charPtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetRelationMem = Function::Create(
      getRelationMem, Function::ExternalLinkage, "getRelationMem", llvmModule);
  pipelineGen->registerFunction("getRelationMem", fgetRelationMem);

  FunctionType *getHTMemKV = FunctionType::get(
      charPtrType, std::vector<Type *>{charPtrType, charPtrType}, false);
  Function *fgetHTMemKV = Function::Create(
      getHTMemKV, Function::ExternalLinkage, "getHTMemKV", llvmModule);
  pipelineGen->registerFunction("getHTMemKV", fgetHTMemKV);

  FunctionType *registerClusterCounts = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, int32PtrType, charPtrType},
      false);
  Function *fregisterClusterCounts =
      Function::Create(registerClusterCounts, Function::ExternalLinkage,
                       "registerClusterCounts", llvmModule);
  pipelineGen->registerFunction("registerClusterCounts",
                                fregisterClusterCounts);

  FunctionType *registerRelationMem = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, charPtrType},
      false);
  Function *fregisterRelationMem =
      Function::Create(registerRelationMem, Function::ExternalLinkage,
                       "registerRelationMem", llvmModule);
  pipelineGen->registerFunction("registerRelationMem", fregisterRelationMem);

  FunctionType *registerHTMemKV = FunctionType::get(
      void_type, std::vector<Type *>{charPtrType, charPtrType, charPtrType},
      false);
  Function *fregisterHTMemKV =
      Function::Create(registerHTMemKV, Function::ExternalLinkage,
                       "registerHTMemKV", llvmModule);
  pipelineGen->registerFunction("registerHTMemKV", fregisterHTMemKV);
}

PipelineGen *OlapCpuPipelineGenFactory::create(Context *context,
                                               std::string pipName,
                                               PipelineGen *copyStateFrom) {
  auto *pipelineGen =
      CpuPipelineGenFactory::create(context, pipName, copyStateFrom);
  OlapCpuPipelineGenFactory::registerFunctions(pipelineGen);
  return pipelineGen;
}

void OlapGpuPipelineGenFactory::registerFunctions(PipelineGen *pipelineGen) {
  registerOlapFunctions(pipelineGen);
}

PipelineGen *OlapGpuPipelineGenFactory::create(Context *context,
                                               std::string pipName,
                                               PipelineGen *copyStateFrom) {
  auto *pipelineGen =
      GpuPipelineGenFactory::create(context, pipName, copyStateFrom);

  // Register olap functions for the main module
  OlapGpuPipelineGenFactory::registerFunctions(pipelineGen);

  // Register olap function for the olap module
  auto *gpuPipelineGen = dynamic_cast<GpuPipelineGen *>(pipelineGen);
  gpuPipelineGen->enableWrapperModule();
  OlapGpuPipelineGenFactory::registerFunctions(pipelineGen);
  gpuPipelineGen->disableWrapperModule();

  return pipelineGen;
}
