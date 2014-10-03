#ifndef RAW_CATALOG_HPP_
#define RAW_CATALOG_HPP_

#include "common/common.hpp"
#include "plugins/helpers.hpp"

using namespace llvm;

class RawCatalog
{
public:

	static RawCatalog& getInstance()
	{
		static RawCatalog instance;
		return instance;
	}

	//XXX is there a more elegant way to group hashtables together?
	multimap<int,void*>* getIntHashTable(string tableName) {
		std::map<std::string, multimap<int,void*>*>::iterator it;
		it = intHTs.find(tableName);
		if (it == intHTs.end()) {
			LOG(INFO) << "Creating HT for table "<<tableName;
			intHTs[tableName] = new multimap<int,void*>();
			return intHTs[tableName];
		}	else	{
			LOG(INFO) << "HT found";
			return intHTs[tableName];
		}
	}

	Type* getType(string tableName)	{
		std::map<std::string, int>::iterator it;
		it = resultTypes.find(tableName);
		if (it == resultTypes.end()) {
			LOG(ERROR) << "Could not locate type of table "<<tableName;
			throw runtime_error(string("Could not locate type of table ")+tableName);
		}	else	{
			int idx = resultTypes[tableName];
			return tableTypes[idx];
		}
	}

	int getTypeIndex(string tableName)	{
		std::map<std::string, int>::iterator it;
		it = resultTypes.find(tableName);
		if (it == resultTypes.end()) {
			LOG(ERROR) << "Info in catalog does not exist for table "<<tableName;
			throw runtime_error(string("Info in catalog does not exist for table ")+tableName);
		}	else	{
			int idx = resultTypes[tableName];
			return idx;
		}
	}

	Type* getTypeInternal(int idx)	{
		return tableTypes[idx];
	}

	void insertTableInfo(string tableName, Type* type)	{
		int idx = resultTypes[tableName];
		if(idx == 0)	{
			idx = getUniqueId();
			resultTypes[tableName] = idx;
			tableTypes[idx] = type;
		}
		else	{
			throw runtime_error(string("Info in catalog already exists for table ")+tableName);
		}
	}

	int getUniqueId()	{
		uniqueTableId++;
		return uniqueTableId;
	}

	BasicBlock* getJoinInsertionPoint()	{
		return joinInsertionPoint;
	}

	void setJoinInsertionPoint(BasicBlock* bb)	{
		joinInsertionPoint = bb;
	}

	JSONHelper* getJSONHelper(char* fileName)	{
		std::map<std::string, JSONHelper*>::iterator it;
		it = jsonFiles.find(fileName);
		if (it == jsonFiles.end()) {
			throw runtime_error(string("Catalog contains no information about JSON file ")+fileName);
		}
		return it->second;
	}

	void setJSONHelper(string fileName, JSONHelper* helper)	{
		if( jsonFiles[fileName] != NULL)
		{
			throw runtime_error(string("Catalog already contains a helper class for ")+fileName);
		}
		else
		{
			jsonFiles[fileName] = helper;
		}
	}

	void clear();

private:
	std::map<string,multimap<int,void*>*> intHTs;

	std::map<string,JSONHelper*> jsonFiles;
	/**
	 * Reason for this: The hashtables we populate (intHTs etc) are created a priori.
	 * Therefore, the 'values' need to be void* to accommodate any kind of 'tuples'
	 * XXX [Actually, is there a more flexible way to handle this? Templates or sth??]
	 */
	std::map<string,int> resultTypes;

	int uniqueTableId;
	int maxTables;

	BasicBlock* joinInsertionPoint;

	//Position 0 not used, so that we can use it to perform containment checks when using tableTypes
	Type** tableTypes;
	//Is maxTables enough????
	RawCatalog()	: uniqueTableId(1) , maxTables(1000), joinInsertionPoint(NULL) {
		tableTypes = new Type*[maxTables];
	}
	~RawCatalog()
	{

	}

	//Not implementing; RawCatalog is a singleton
	RawCatalog(RawCatalog const&);              // Don't Implement.
	void operator=(RawCatalog const&); // Don't implement
};


#endif /* RAW_CATALOG_HPP_ */
