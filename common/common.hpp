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

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <string>
#include <stdexcept>

//LLVM Includes
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/PassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"

//#JSON
#define JSMN_STRICT
#include "jsmn/jsmn.h"

//Handling conflict between Mongo and Google logger
#include "mongodb/db/jsobj.h"
#include "mongodb/db/json.h"
#undef LOG
//Used to remove all logging messages at compile time and not affect performance
//Must be placed before glog include
//#define GOOGLE_STRIP_LOG 1
#include <glog/logging.h>

#define DEBUG

#define KB 1024
#define MB (1024*KB)

using std::cout;
using std::runtime_error;
using std::string;
using std::endl;
using std::pair;
using std::cout;
using std::multimap;
using std::list;

using namespace llvm;

double diff(struct timespec st, struct timespec end);

void fatal(const char *err);

void exception(const char *err);


#endif /* COMMON_HPP_ */
