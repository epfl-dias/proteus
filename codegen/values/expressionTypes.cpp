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

#include "values/expressionTypes.hpp"

bool recordComparator (RecordAttribute* x, RecordAttribute* y) {
	return (x->getAttrNo() < y->getAttrNo());
}

Value * RecordType::projectArg(Value * record, RecordAttribute * attr, IRBuilder<>* const Builder) const{
	if (!(record->getType()->isStructTy())) return NULL;
	if (!(((StructType *) record->getType())->isLayoutIdentical((StructType *) getLLVMType(record->getContext())))) return NULL;
	int index = getIndex(attr);
	if (index < 0) return NULL;
	return Builder->CreateExtractValue(record, index);
}

int RecordType::getIndex(RecordAttribute * x) const{
	int index = 0;
	for (const auto &attr: args){
		if (x->getAttrName() == attr->getAttrName()) return index;
		++index;
	}
	return -1;
}