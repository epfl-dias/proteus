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

#include "common/common.hpp"
#include "operators/scan.hpp"
#include "operators/root.hpp"
#include "plugins/json-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"

#include <boost/foreach.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "jsoncpp/json/json.h"

#include "common/common.hpp"

#include "succinct/util.hpp"
#include "succinct/mapper.hpp"

#include "semi_index/json_semi_index.hpp"
#include "semi_index/path_parser.hpp"
#include "semi_index/zrandom.hpp"

using json::path::path_element_t;
using json::path::path_t;
using json::path::path_list_t;
using succinct::util::fast_getline;
using semi_index::json_semi_index;

//Used for comparison with pure JSON scan
//delicious.json file not included in uploaded repo due to size concerns
int main()
{
	RawContext context = RawContext("Benchmark - JSON parser");

	//SCAN1
	string filename = string("delicious.json");
	PrimitiveType* intType = new IntType();//PrimitiveType(Int);
	PrimitiveType* stringType = new StringType();//PrimitiveType(Int);
	string field1 = string("title");
	RecordAttribute* attr1 = new RecordAttribute(1,field1,stringType);
	RecordAttribute* attr2 = new RecordAttribute(2,string("author"),stringType);

	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

	semi_index::JSONPlugin* pg = new semi_index::JSONPlugin(&context, filename, &whichFields,&whichFields);
	Scan scan = Scan(&context, *pg);

	//ROOT
	Root rootOp = Root(&scan);
	scan.setParent(&rootOp);
	rootOp.produce();

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	context.prepareFunction(context.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n",diff(t0, t1));


	//Close all open files
	pg->finish();
}



