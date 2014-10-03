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

#include "plugins/helpers.hpp"

JSONHelper::JSONHelper(const string& file,
		vector<RecordAttribute*>* attsToSelect,
		vector<RecordAttribute*>* attsToProject) :	attsToSelect(attsToSelect),
													attsToProject(attsToProject),
													fileName(file)
{
	//Using custom comparator
	bool(*fn_pt)(RecordAttribute*,RecordAttribute*) = recordComparator;
	std::set<RecordAttribute*,bool(*)(RecordAttribute*,RecordAttribute*)>* whichFields =
			new std::set<RecordAttribute*,bool(*)(RecordAttribute*,RecordAttribute*)>(fn_pt);

	for(std::vector<RecordAttribute*>::iterator it = attsToSelect->begin();
		it != attsToSelect->end();
		it++)	{
		whichFields->insert(*it);
	}
	for(std::vector<RecordAttribute*>::iterator it = attsToProject->begin();
		it != attsToSelect->end();
		it++)	{
		(*it)->setProjected();
		whichFields->insert(*it);
	}

	LOG(INFO) << "[Helper Construction: ] " << file;
	stringstream ss;
	bool first = true;
	for(std::set<RecordAttribute*>::iterator it = whichFields->begin(); it != whichFields->end(); it++)	{
		if(!first)	{
			ss<<",";
		}
		first=false;
		RecordAttribute* rec = *it;
		ss<<rec->getName();
	}
	allAtts = whichFields;

	const char* searchTerms = ss.str().c_str();
	atts = new path_list_t();
	*atts = json::path::parse(searchTerms);
	attsNo = atts->size();
	//LOG(INFO) << "[HELPER TERMS: ] "<<ss.str();
	indexName = fileName+".si";

	struct stat buffer;
	//Check for existence; if it does not exist, build JSON semi_index now
	const char* indexNameC = indexName.c_str();
	if(stat (indexNameC, &buffer) != 0)	{
		std::cout<<string("[Building JSON index!:] ")<<indexNameC<<std::endl;
		si_save(indexNameC);
	}

	//Map file
	boost::iostreams::mapped_file_source* json_map = new boost::iostreams::mapped_file_source(fileName);
	buf = (*json_map).data();
	boost::iostreams::mapped_file_source* m = new boost::iostreams::mapped_file_source(indexName);
	index = new json_semi_index();
	succinct::mapper::map(*index, *m, succinct::mapper::map_flags::warmup);

	//'Next' cursor will be advanced in a different pace by eof() checker
	cursor = new json_semi_index::cursor();
	*cursor = index->get_cursor();
	nextCursor = cursor;

	LOG(INFO) << "[Helper Created: ] "<<file;
}

void JSONHelper::si_save(const char* index_file) {
	using succinct::util::lines;
	FILE* fileInput = fopen(fileName.c_str(),"r");
	if(fileInput == NULL)	{
		throw runtime_error(string("Failed to open json file"));
	}
	json_semi_index index(lines(fileInput));
	succinct::mapper::size_tree_of(index)->dump();
	succinct::mapper::freeze(index, index_file);
}
