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

#include "plugins/json-jsmn-plugin.hpp"

//Definitely not enough as a solution
#define MAXTOKENS 1000

namespace jsmn	{

#define TOKEN_PRINT(t) \
	printf("start: %d, end: %d, type: %d, size: %d\n", \
			(t).start, (t).end, (t).type, (t).size)

#define TOKEN_STRING(js, t, s) \
	(strncmp(js+(t).start, s, (t).end - (t).start) == 0 \
	 && strlen(s) == (t).end - (t).start)


JSONPlugin::JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema)
	: context(context),
	  fname(fname),
	  schema(schema),
	  var_buf("bufPtr"),
	  var_tokenPtr("tokens"),
	  var_tokenOffset("tokenOffset")	{

	//Memory mapping etc
	LOG(INFO) << "[JSONPlugin - jsmn: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;
	fd = open(name_c, O_RDONLY);
	if (fd == -1) {
		throw runtime_error(string("json.open"));
	}
	buf = (const char*) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
	if (buf == MAP_FAILED) {
		throw runtime_error(string("json.mmap"));
	}

	//Retrieving schema - not needed yet
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.registerFileJSON(fname,schema);

	//Tokenizing
	int error_code;
	jsmn_parser p;

	//Populating our json 'positional index'
	tokens = new jsmntok_t[MAXTOKENS];
	if(tokens == NULL)	{
		throw runtime_error(string("new() of tokens failed"));
	}
	for(int i = 0; i < MAXTOKENS; i++)	{
		tokens[i].start = 0;
		tokens[i].end = 0;
		tokens[i].size = 0;
	}

	jsmn_init(&p);
	error_code = jsmn_parse(&p, buf, strlen(buf), tokens, MAXTOKENS);
	if(error_code < 0)	{
		string msg = "Json (JSMN) plugin failure: ";
		LOG(ERROR) << msg << error_code;
		throw runtime_error(msg);
	}

	//Preparing structures and variables for codegen part
	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<>* Builder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);

	//Buffer holding the JSON document
	AllocaInst *mem_buf = context->CreateEntryBlockAlloca(F,std::string("charPtr"),charPtrType);
	Value* val_buf_i64 = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* val_buf = Builder->CreateIntToPtr(val_buf_i64,charPtrType);
	Builder->CreateStore(val_buf,mem_buf);
	NamedValuesJSON[var_buf] = mem_buf;

	//The array of tokens
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
	AllocaInst *mem_tokenPtr = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),ptr_jsmnStructType);
	Value* ptrVal = ConstantInt::get(llvmContext, APInt(64,((uint64_t)tokens)));
	//i8*
	Value* unshiftedPtr = Builder->CreateIntToPtr(ptrVal,ptr_jsmnStructType);
	Builder->CreateStore(unshiftedPtr,mem_tokenPtr);
	NamedValuesJSON[var_tokenPtr] = mem_tokenPtr;

	AllocaInst *mem_tokenOffset = context->CreateEntryBlockAlloca(F,std::string(var_tokenOffset),int64Type);
	//0 represents the entire JSON document
	Value* offsetVal = Builder->getInt64(1);
	Builder->CreateStore(offsetVal,mem_tokenOffset);
	NamedValuesJSON[var_tokenOffset] = mem_tokenOffset;
}

/*
 * Generic INTERPRETED Implementation: Simply identifies "tuples"
 * Assumption: Handles data in the form [ obj, obj, ..., obj ]
 */
void JSONPlugin::scanObjectsInterpreted(list<string> path, list<ExpressionType*> types)	{
	//Token[0] contains info about entire document
	for (int i = 1; tokens[i].start != 0; )	{
		//We want token i to be one of the 'outermost' objects
		int curr = i;

		//Work done for every 'tuple'
		//TOKEN_PRINT(tokens[i]);
		int neededToken = readPathInterpreted(i, path);
		//Actually, must return (nil) in this case!!!
		if(neededToken <= 0)	{
			printf("(nil)\n");
		}
		readValueInterpreted(neededToken, types.back());

		//'skipToEnd'
		while(tokens[i].start < tokens[curr].end && tokens[i].start != 0)	{
			i++;
		}
	}
}

void JSONPlugin::scanObjects(const RawOperator& producer, Function* debug)	{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();

	//Get the entry block
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Container for the variable bindings
	std::map<std::string, AllocaInst*>* variableBindings = new std::map<std::string, AllocaInst*>();

	/**
	 * Loop through results (if any)
	 * for (int i = 1; tokens[i].start != 0; )
	 */
	BasicBlock *jsonScanCond, *jsonScanBody, *jsonScanInc, *jsonScanEnd;
	context->CreateForLoop("jsonScanCond", "jsonScanBody", "jsonScanInc","jsonScanEnd",
							&jsonScanCond, &jsonScanBody, &jsonScanInc,	&jsonScanEnd);

	/**
	 * Entry Block: Simply jumping to condition part
	 */
	Builder->CreateBr(jsonScanCond);

	/**
	 * Condition: tokens[i].start != 0
	 */
	Builder->SetInsertPoint(jsonScanCond);

	//Prepare left-hand side
	AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];
	AllocaInst* mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
	Value* val_offset = Builder->CreateLoad(mem_tokenOffset);

	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
	AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_offset);
	Builder->CreateStore(token_i,mem_tokens_shifted);
	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_start = context->getStructElem(mem_tokens_shifted,1);

	#ifdef DEBUG
//		std::vector<Value*> ArgsV;
//		ArgsV.clear();
//		ArgsV.push_back(token_i_start);
//		Function* debugInt = context->getFunction("printi");
//		Builder->CreateCall(debugInt, ArgsV);
	#endif

	//Prepare right-hand side
	Value* rhs = context->createInt32(0);

	ICmpInst* endCond = new ICmpInst(*jsonScanCond, ICmpInst::ICMP_NE,
			token_i_start, rhs, "cmpJSMNEnd");

	BranchInst::Create(jsonScanBody, jsonScanEnd, endCond, jsonScanCond);

	/**
	 *	BODY
	 */
	Builder->SetInsertPoint(jsonScanBody);

	//	//This will be the insertion point for the rest of the code
	//	//Triggering parent
	//	OperatorState* state = new OperatorState(producer, *variableBindings);
	//	RawOperator* const opParent = producer.getParent();
	//	opParent->consume(context,*state);
	readPath(val_offset,"b");

	skipToEnd();

	Builder->CreateBr(jsonScanInc);
	/**
	 * INC:
	 * Nothing to do - skipToEnd() takes care of inc
	 */
	Builder->SetInsertPoint(jsonScanInc);
	Builder->CreateBr(jsonScanCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(jsonScanEnd);
	LOG(INFO) << "[Scan - JSON: ] End of scanObjects()";

}

/**
 *  while(tokens[i].start < tokens[curr].end && tokens[i].start != 0)	i++;
 */
void JSONPlugin::skipToEnd()	{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];
	AllocaInst* mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
	Value* val_offset = Builder->CreateLoad(mem_tokenOffset);

	Value* val_curr = Builder->CreateLoad(mem_tokenOffset);
	AllocaInst* mem_tokens_curr_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_curr = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_curr);
	Builder->CreateStore(token_curr,mem_tokens_curr_shifted);
	Value* token_curr_end = context->getStructElem(mem_tokens_curr_shifted,2);

	/**
	 * LOOP BLOCKS
	 */
	BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
	context->CreateForLoop("tokenSkipCond", "tokenSkipBody", "tokenSkipInc","tokenSkipEnd",
							&tokenSkipCond, &tokenSkipBody, &tokenSkipInc,	&tokenSkipEnd);

	/**
	 * Entry Block: Simply jumping to condition part
	 */
	Builder->CreateBr(tokenSkipCond);

	/**
	 * Condition: tokens[i].start < tokens[curr].end && tokens[i].start != 0
	 */
	Builder->SetInsertPoint(tokenSkipCond);
	val_offset = Builder->CreateLoad(mem_tokenOffset);
	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_offset);
	Builder->CreateStore(token_i,mem_tokens_i_shifted);

	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_start = context->getStructElem(mem_tokens_i_shifted,1);
	Value* rhs = context->createInt32(0);

	Value *endCond1 = Builder->CreateICmpSLT(token_i_start,token_curr_end);
	Value *endCond2 = Builder->CreateICmpNE(token_i_start,rhs);
	Value *endCond  = Builder->CreateAnd(endCond1,endCond2);

	BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

	/**
	 * BODY:
	 * i++
	 */
	Builder->SetInsertPoint(tokenSkipBody);

	val_offset = Builder->CreateLoad(mem_tokenOffset);
	Value *val_step = Builder->getInt64(1);
	//CastInst* token_i_start64 = new SExtInst(token_i_start, int64Type, "i_64", tokenSkipBody);
	//cout<<int64_conv->getType()->getTypeID()<< " vs " << val_step->getType()->getTypeID();

	//Builder->CreateCall(debugInt, ArgsV, "printi");
	Value *token_i_inc = Builder->CreateAdd(val_offset, val_step, "i_inc");
	Builder->CreateStore(token_i_inc, mem_tokenOffset);

#ifdef DEBUG
	//std::vector<Value*> ArgsV;
	//ArgsV.clear();
	//ArgsV.push_back(token_i_start);
	//Function* debugInt = context->getFunction("printi");
	//Builder->CreateCall(debugInt, ArgsV);
#endif

	Builder->CreateBr(tokenSkipInc);

	/**
	 * INC:
	 * Nothing to do
	 * (in principle, job done in body could be done here)
	 */
	Builder->SetInsertPoint(tokenSkipInc);
	Builder->CreateBr(tokenSkipCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(tokenSkipEnd);
	LOG(INFO) << "[Scan - JSON: ] End of skiptoEnd()";
}

int JSONPlugin::readPathInterpreted(int parentToken, list<string> path)	{
	//Only objects are relevant to path expressions
	if(tokens[parentToken].type != JSMN_OBJECT)	{
		string msg = string("[JSON Plugin - jsmn: ]: Path traversal is only applicable to objects");
		LOG(ERROR) << msg;
		throw runtime_error(msg);
	}
	if(path.size() == 0)	{
		string error_msg = "[JSONPlugin - jsmn: ] Path length cannot be 0";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	string key = path.front();
	for(int i = parentToken + 1; tokens[i].end <= tokens[parentToken].end; i+=2)	{
		if(TOKEN_STRING(buf,tokens[i],key.c_str()))	{
			//next one is the one I need
			printf("Found ");
			TOKEN_PRINT(tokens[i+1]);
			if(path.size() == 1)	{
				return i+1;
			}	else	{
				path.pop_front();
				return readPathInterpreted(i+1 , path);
			}
		}
	}
	return -1;
}

void JSONPlugin::readPath(Value* parentTokenNo, char* path)	{
	//Only objects are relevant to path expressions
	//These types of validation should be applied as high as possible
	//Still, probably unavoidable here
	//	if(tokens[parentTokenNo].type != JSMN_OBJECT)	{
	//		string msg = string("[JSON Plugin - jsmn: ]: Path traversal is only applicable to objects");
	//		LOG(ERROR) << msg;
	//		throw runtime_error(msg);
	//	}

	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];

	AllocaInst* mem_tokens_parent_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_parent = context->getArrayElem(mem_tokens, ptr_jsmnStructType, parentTokenNo);
	Builder->CreateStore(token_parent,mem_tokens_parent_shifted);
	Value* token_parent_end = context->getStructElem(mem_tokens_parent_shifted,2);
	/**
	 * LOOP BLOCKS
	 */
	BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
	context->CreateForLoop("path_tokenSkipCond", "path_tokenSkipBody", "path_tokenSkipInc","path_tokenSkipEnd",
							&tokenSkipCond, &tokenSkipBody, &tokenSkipInc,	&tokenSkipEnd);

	/**
	 * Entry Block:
	 */

	Value *val_1 = Builder->getInt64(1);
	Value *val_i = Builder->CreateAdd(parentTokenNo, val_1);
	AllocaInst* mem_i = context->CreateEntryBlockAlloca(F,std::string("tmp_i"),int64Type);
	Builder->CreateStore(val_i,mem_i);
	Builder->CreateBr(tokenSkipCond);

	/**
	 * tokens[i].end <= tokens[parentToken].end
	 */

	Builder->SetInsertPoint(tokenSkipCond);
	val_i = Builder->CreateLoad(mem_i);
	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_i);
	Builder->CreateStore(token_i,mem_tokens_i_shifted);

	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_end = context->getStructElem(mem_tokens_i_shifted,2);
	Value *endCond = Builder->CreateICmpSLE(token_i_end,token_parent_end);
	BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

	/**
	* BODY:
	*/
	Builder->SetInsertPoint(tokenSkipBody);

	/**
	 * IF-ELSE inside body:
	 * if(TOKEN_STRING(buf,tokens[i],key.c_str()))
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifTokenEq", "elseTokenEq",
								&ifBlock, &elseBlock,tokenSkipInc);

	Value* token_i_start = context->getStructElem(mem_tokens_i_shifted,1);
	Value* globalStr = context->CreateGlobalString(path);
	Value* buf = Builder->CreateLoad(NamedValuesJSON[var_buf]);
	//Preparing custom 'strcmp'
	std::vector<Value*> argsV;
	argsV.push_back(buf);
	argsV.push_back(token_i_start);
	argsV.push_back(token_i_end);
	argsV.push_back(globalStr);
	Function* tokenCmp = context->getFunction("compareTokenString");
	Value* tokenEq = Builder->CreateCall(tokenCmp, argsV);
	Value* rhs = context->createInt32(1);
	Value *cond  = Builder->CreateICmpEQ(tokenEq,rhs);
	Builder->CreateCondBr(cond,ifBlock,elseBlock);

	/**
	 * IF BLOCK
	 * TOKEN_PRINT(tokens[i+1]);
	 */
	Builder->SetInsertPoint(ifBlock);
#ifdef DEBUG
	Value* val_i_1 = Builder->CreateAdd(val_i , val_1);
	AllocaInst* mem_tokens_i_1_shifted = context->CreateEntryBlockAlloca(F,std::string(var_tokenPtr),context->CreateJSMNStruct());
	Value* token_i_1 = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_i_1);
	Builder->CreateStore(token_i_1,mem_tokens_i_1_shifted);
	Value* token_i_1_start = context->getStructElem(mem_tokens_i_1_shifted,1);

	argsV.clear();
	argsV.push_back(token_i_1_start);
	Function* debugInt = context->getFunction("printi");
	Builder->CreateCall(debugInt, argsV);
#endif
	Builder->CreateBr(tokenSkipEnd);

	/**
	 * ELSE BLOCK
	 */
	Builder->SetInsertPoint(elseBlock);
	Builder->CreateBr(tokenSkipInc);

	/**
	 * (Back to LOOP)
	 * INC:
	 * i += 2
	 */
	Builder->SetInsertPoint(tokenSkipInc);
	val_i = Builder->CreateLoad(mem_i);
	Value* val_2 = Builder->getInt64(2);
	Value* val_i_2 = Builder->CreateAdd(val_i , val_2);
	Builder->CreateStore(val_i_2 , mem_i);

	//	argsV.clear();
	//	argsV.push_back(val_i_2);
	//	Function* debugInt64 = context->getFunction("printi64");
	//	Builder->CreateCall(debugInt64, argsV);

	token_i = context->getArrayElem(mem_tokens, ptr_jsmnStructType, val_i_2);
	Builder->CreateStore(token_i,mem_tokens_i_shifted);

	Builder->CreateBr(tokenSkipCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(tokenSkipEnd);
	LOG(INFO) << "[Scan - JSON: ] End of readPath()";
}


void JSONPlugin::readValueInterpreted(int tokenNo, const ExpressionType* type) {
	string error_msg;
	TOKEN_PRINT(tokens[tokenNo]);
	switch(type->getTypeID())	{
	case RECORD:
		//object
	case LIST:
		//array
		printf("Passing object along\n");
		break;
	case SET:
		error_msg = string("[JSON Plugin - jsmn: ]: SET datatype cannot occur");
		LOG(ERROR) << error_msg;
		throw runtime_error(string(error_msg));
	case BAG:
		error_msg = string("[JSON Plugin - jsmn: ]: BAG datatype cannot occur");
		LOG(ERROR) << error_msg;
		throw runtime_error(string(error_msg));
	case BOOL:
		if(TOKEN_STRING(buf,tokens[tokenNo],"true"))	{
			printf("Value: True!\n");
		} else if(TOKEN_STRING(buf,tokens[tokenNo],"true")) {
			printf("Value: False!\n");
		} else	{
			error_msg = string("[JSON Plugin - jsmn: ]: Error when parsing boolean");
			LOG(ERROR) << error_msg;
			throw runtime_error(string(error_msg));
		}
	case STRING:
		printf("Passing object along\n");
		break;
	case FLOAT:
		printf("Double value read: %f\n",atof(buf + tokens[tokenNo].start));
		break;
	case INT:
		//Must be careful with trailing whitespaces for some item
		//printf("Int value read: %d\n",atois(buf + tokens[tokenNo].start, tokens[tokenNo].end - tokens[tokenNo].start));
		printf("Int value read: %d\n",atoi(buf + tokens[tokenNo].start));
		break;
	default:
		error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
		LOG(ERROR)<< error_msg;
		throw runtime_error(string(error_msg));
	}

}

void JSONPlugin::generate(const RawOperator& producer) {
	return scanObjects(producer, context->getGlobalFunction());
}

void JSONPlugin::finish() {
	close(fd);
	munmap((void*) buf,fsize);
}

JSONPlugin::~JSONPlugin() {
	delete tokens;
}

}

//
////Only objects are relevant to path expressions after all
//int JSONPlugin::locateObject(int parentToken, string key)	{
//	for(int i = parentToken; tokens[i].end <= tokens[parentToken].end; i++)	{
//		if(tokens[i].type == JSMN_OBJECT)	{
//			printf("Object indeed\n");
//			jsmntok_t objName = tokens[i+1];
//			if(TOKEN_STRING(buf,objName,key.c_str()))	{
//				//next one is the one I need
//				printf("Found???");
//				TOKEN_PRINT(tokens[i+2]);
//				return i+2;
//			}
//		}
//	}
//	//Not found
//	return -1;
//}
//
//void JSONPlugin::loop(string key)	{
//
//	//Equivalent of scan csv rows
//	int i = 1;
//	for (i = 1; tokens[i].start != 0; )	{
//
//		//We want token i to be one of the 'outermost' objects
//		int curr = i;
//		printf("Which token? %d\n",i);
//		TOKEN_PRINT(tokens[i]);
//
//		//'Pipelined' function here
//		locateObject(i, key);
//
//		//Equivalent of 'skipToEnd'
//		while(tokens[i].start < tokens[curr].end && tokens[i].start != 0)	{
//			i++;
//			//printf("Which token and pos? %d %d\n",i,tokens[i].start);
//		}
//	}
////	printf("Which token? %d\n",i);
////	TOKEN_PRINT(tokens[i]);
//
//
//
////	for (int i = 1; tokens[i].end < tokens[0].end; i++) {
////			if (tokens[i].type == JSMN_STRING || tokens[i].type == JSMN_PRIMITIVE) {
////				printf("%.*s\n", tokens[i].end - tokens[i].start, buf + tokens[i].start);
////			} else if (tokens[i].type == JSMN_ARRAY) {
////				printf("[%d elems]\n", tokens[i].size);
////			} else if (tokens[i].type == JSMN_OBJECT) {
////				printf("{%d elems}\n", tokens[i].size);
////			} else {
////				TOKEN_PRINT(tokens[i]);
////			}
////		}
//
//}

//int JSMNPlugin::calculateTokensPerItem(const ExpressionType& expr)	{
//	string error_msg;
//	int parentSize = tokens[0].size;
//	switch (schema->getTypeID()) {
//		case BOOL:
//		case STRING:
//		case FLOAT:
//		case INT:
//		case RECORD:
//		case BAG:
//			error_msg = string(
//					"Outermost element cannot be of type " + schema->getType());
//			LOG(ERROR)<< error_msg;
//			throw runtime_error(string(error_msg));
//		case SET:
//			//object
//		case LIST:
//			//array
//			return calculateTokens_(((SetType)expr).getNestedType());
//	}
//}
//
////Generic
//int JSMNPlugin::calculateTokens_(const ExpressionType& expr)	{
//	string error_msg;
//		int parentSize = tokens[0].size;
//		switch (schema->getTypeID()) {
//			case BOOL: return 1;
//			case STRING: return 1;
//			case FLOAT: return 1;
//			case INT: return 1;
//			case RECORD:
//				int sum = 0;
//				for(list<RecordAttribute*>::iterator it = ((RecordType)expr).getArgs().begin();
//					it != ((RecordType)expr).getArgs().end();
//					it++)	{
//					sum += 1; //The key of the attribute
//					sum += calculateTokens_(*(*it)->getOriginalType());
//				}
//				return sum;
//			case LIST:
//				//array
//
//			case BAG:
//			case SET:
//				error_msg = string("JSON inner elements cannot be of type " + schema->getType());
//				LOG(ERROR)<< error_msg;
//				throw runtime_error(string(error_msg));
//			}
//}





//JSONPlugin::JSONPlugin(RawContext* const context, string& fname,
//		vector<RecordAttribute*>* fieldsToSelect,
//		vector<RecordAttribute*>* fieldsToProject) :
//		attsToProject(fieldsToSelect), attsToSelect(fieldsToProject), context(context), fname(fname) {
//
//	helper = new JSONHelper(fname, fieldsToSelect, fieldsToProject);
//	RawCatalog& catalog = RawCatalog::getInstance();
//	catalog.setJSONHelper(fname,helper);
//}
//
//void JSONPlugin::init() {}
//
//void JSONPlugin::scanJSON(const RawOperator& producer, Function* debug) {
//
//	//Prepare
//	LLVMContext& llvmContext = context->getLLVMContext();
//	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//	Type* int64Type = Type::getInt64Ty(llvmContext);
//	IRBuilder<>* Builder = context->getBuilder();
//
//	//Get the entry block
//	Function *TheFunction = Builder->GetInsertBlock()->getParent();
//
//	//Container for the variable bindings
//	std::map<std::string, AllocaInst*>* variableBindings = new std::map<std::string, AllocaInst*>();
//
//	//Loop through results (if any)
//	BasicBlock *jsonScanCond, *jsonScanBody, *jsonScanInc, *jsonScanEnd;
//	context->CreateForLoop("jsonScanCond", "jsonScanBody", "jsonScanInc","jsonScanEnd",
//							&jsonScanCond, &jsonScanBody, &jsonScanInc,	&jsonScanEnd);
//
//	JSONHelper* helper = this->helper;
//		int attsNo = helper->getAttsNo();
//	int nameSize = (helper->getFileNameStr()).size() + 1;
//	char* filename_noconst = (char *) alloca(nameSize);
//	memcpy(filename_noconst, helper->getFileName(), nameSize);
//	Value* filenameLLVM = context->CreateGlobalString(filename_noconst);
//	std::vector<Value*> ArgsV;
//
//	//Flushing code in Entry Block
//	Builder->CreateBr(jsonScanCond);
//
//	//Condition! --> while (!(cursor == json_semi_index::cursor()))
//	//Prepare eof function arguments
//	Builder->SetInsertPoint(jsonScanCond);
//	ArgsV.clear();
//	ArgsV.push_back(filenameLLVM);
//	Function* endCond = context->getFunction("eofJSON");
//	Value* endCheck = Builder->CreateCall(endCond, ArgsV/*,"eofJSON"*/);
//	Value* falseLLVM = context->createFalse();
//	ICmpInst* eof_cmp = new ICmpInst(*jsonScanCond, ICmpInst::ICMP_EQ, endCheck,
//			falseLLVM, "cmpJSONEnd");
//	BranchInst::Create(jsonScanBody, jsonScanEnd, eof_cmp, jsonScanCond);
//
//	//BODY
//	Builder->SetInsertPoint(jsonScanBody);
//	std::set<RecordAttribute *, bool (*)(RecordAttribute *, RecordAttribute *)>* allAtts =
//			helper->getAllAtts();
//
//	int i = 0;
//	for (std::set<RecordAttribute*>::iterator it = allAtts->begin(); it != allAtts->end(); it++) {
//		ArgsV.clear();
//		ExpressionType* type = (*it)->getOriginalType();
//		Value* currAttrNo = context->createInt32(i++);
//		ArgsV.push_back(filenameLLVM);
//		ArgsV.push_back(currAttrNo);
//		string currAttrName = (*it)->getName();
//
//		if ((*it)->isProjected() && !(type->isPrimitive())) {
//			Function* jsonObjectScan = context->getFunction("getJSONPositions");
//			LOG(INFO) << "[Scan - JSON: ] Non-primitive datatype " << currAttrName
//					  << " requested from " << helper->getFileNameStr();
//
//			//Preparing JSON struct
//			Value* positionStructs = Builder->CreateCall(jsonObjectScan, ArgsV);
//			Type* jsonStructType = context->CreateJSONPosStruct();
//			AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,
//					"currJSONResult", jsonStructType);
//			Builder->CreateStore(positionStructs, Alloca);
//			(*variableBindings)[currAttrName] = Alloca;
//		} else {
//			Value* result;
//			Type* resultType;
//			Function* jsonPrimitiveScan;
//			LOG(INFO) << "[Scan - JSON: ] Primitive datatype " << currAttrName
//					  << " requested from " << helper->getFileNameStr();
//			switch (type->getTypeID()) {
//			case BOOL:
//				throw runtime_error(string("Not Supported Yet: ") + (*it)->getType());
//			case STRING:
//				jsonPrimitiveScan = context->getFunction("getJSONPositions");
//				resultType = context->CreateJSONPosStruct();
//				break;
//			case FLOAT:
//				jsonPrimitiveScan = context->getFunction("getJSONDouble");
//				resultType = Type::getDoubleTy(llvmContext);
//				break;
//			case INT:
//				jsonPrimitiveScan = context->getFunction("getJSONInt");
//				resultType = Type::getInt32Ty(llvmContext);
//				break;
//			case COLLECTION:
//				throw runtime_error(string("Only primitive types should qualify here: ")
//								           + (*it)->getType());
//			case RECORD:
//				throw runtime_error(string("Only primitive types should qualify here: ")
//								           + (*it)->getType());
//			default:
//				throw runtime_error(string("Unknown type: ") + (*it)->getType());
//			}
//
//			result = Builder->CreateCall(jsonPrimitiveScan, ArgsV);
//			AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,"currJSONResult", resultType);
//			Builder->CreateStore(result, Alloca);
//			(*variableBindings)[currAttrName] = Alloca;
//		}
//	}
//
//	Builder->CreateBr(jsonScanInc);
//	Builder->SetInsertPoint(jsonScanInc);
//
//
//	//Triggering parent
//	OperatorState* state = new OperatorState(producer, *variableBindings);
//	RawOperator* const opParent = producer.getParent();
//	opParent->consume(context,*state);
//
//	// Block for.inc (scan_inc)
//	Builder->CreateBr(jsonScanCond);
//
//	// Block for.end (label_for_end)
//	Builder->SetInsertPoint(jsonScanEnd);
//	LOG(INFO) << "[Scan - JSON: ] End of scan code @ plugin";
//}
//
