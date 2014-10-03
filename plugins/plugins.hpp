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
#include "util/raw-catalog.hpp"
#include "operators/operators.hpp"
#include "operators/operator-state.hpp"



/**********************************/
/*  The abstract part of plug-ins */
/**********************************/
class Plugin {
public:
	virtual ~Plugin() { LOG(INFO) << "Collapsing plug-in"; }
	virtual string& getName() = 0;
	virtual void init() = 0;
	virtual void finish() = 0;
	virtual void generate(const RawOperator& producer) = 0;
};
#endif /* PLUGINS_LLVM_HPP_ */
