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

#ifndef PLUGINS_LLVM_HPP_
#define PLUGINS_LLVM_HPP_

#include "common/common.hpp"
#include "util/raw-context.hpp"
#include "operators/operators.hpp"
#include "operators/operator-state.hpp"

//Used by all plugins
//static const std::string activeTuple = "activeTuple";
static const string activeLoop = "activeTuple";

/**
 * In principle, every readPath() method should deal with a record.
 * For some formats/plugins, however, (e.g. CSV) projection pushdown makes significant
 * difference in performance.
 */
typedef struct Bindings	{
	const OperatorState* state;
	const Value* record;
} Bindings;

/**********************************/
/*  The abstract part of plug-ins */
/**********************************/
class Plugin {
public:
	virtual 			~Plugin() 														{ LOG(INFO) << "[PLUGIN: ] Collapsing plug-in"; }
	virtual 			string& getName() 												= 0;
	virtual void 		init() 															= 0;
	virtual void 		finish() 														= 0;
	virtual void 		generate(const RawOperator& producer) 							= 0;
//	virtual AllocaInst* readPath(const OperatorState& state, char* pathVar) 			= 0;
	virtual AllocaInst* readPath(Bindings wrappedBindings, const char* pathVar)			= 0;
	virtual AllocaInst*	readValue(AllocaInst* mem_value, const ExpressionType* type) 	= 0;
};
#endif /* PLUGINS_LLVM_HPP_ */
