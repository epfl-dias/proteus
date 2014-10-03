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

#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <boost/foreach.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "semi_index/json_semi_index.hpp"
#include "semi_index/path_parser.hpp"
#include "semi_index/zrandom.hpp"
#include "jsoncpp/json/json.h"
#include "succinct/util.hpp"
#include "succinct/mapper.hpp"

#include "common/common.hpp"
#include "values/expressionTypes.hpp"

using json::path::path_element_t;
using json::path::path_t;
using json::path::path_list_t;
using succinct::util::fast_getline;
using semi_index::json_semi_index;

class JSONHelper {
public:
	JSONHelper(const string& file, vector<RecordAttribute*>* fieldsToSelect, vector<RecordAttribute*>* fieldsToProject);

	json_semi_index& getIndex()															{ return *index; }
	json_semi_index::cursor* getCursor()												{ return cursor; }
	void setCursor(json_semi_index::cursor* newCursor)									{ cursor = newCursor; }
	json_semi_index::cursor* getNextCursor()											{ return nextCursor; }
	void setNextCursor(json_semi_index::cursor* newCursor)								{ nextCursor = newCursor; }
	path_list_t& getAtts()																{ return *atts; }
	vector<RecordAttribute*> getSelectAtts()											{ return *attsToSelect; }
	vector<RecordAttribute*> getProjectAtts()											{ return *attsToProject; }
	std::set<RecordAttribute*,bool(*)(RecordAttribute*,RecordAttribute*)>* getAllAtts() { return allAtts; }
	int getAttsNo()																		{ return attsNo; }
	const char* getRawBuf() 															{ return buf; }
	const char* getFileName()															{ return fileName.c_str(); }
	string getFileNameStr()																{ return fileName; }

	void si_save(const char* index_file);

private:
	string fileName;
	string indexName;
	const char *buf;
	json_semi_index* index;

	//Using two cursors as a workaround;
	//Cursor must be advanced once per 'tuple', so this job had to take place in the eof() function
	json_semi_index::cursor* cursor;
	json_semi_index::cursor* nextCursor;

	vector<RecordAttribute*>* attsToSelect;
	vector<RecordAttribute*>* attsToProject;
	std::set<RecordAttribute*,bool(*)(RecordAttribute*,RecordAttribute*)>* allAtts;
	path_list_t* atts;
	int attsNo;

};

typedef struct JSONObject	{
	size_t pos;//turn to char* at the appropriate time
	size_t end;
} JSONObject;

#endif /* HELPERS_HPP_ */
