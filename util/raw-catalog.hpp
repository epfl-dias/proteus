#ifndef RAW_CATALOG_HPP_
#define RAW_CATALOG_HPP_

#include "common/common.hpp"
#include "plugins/plugins.hpp"
#include "values/expressionTypes.hpp"

#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using rapidjson::Writer;
using rapidjson::StringBuffer;

//Forward Declaration
class Plugin;

using namespace llvm;

//FIXME no need to be a singleton (I think)
class RawCatalog
{
public:

	static RawCatalog& getInstance()
	{
		static RawCatalog instance;
		return instance;
	}

	//TODO REPLACE ONCE REMOVED FROM JOIN
	//TODO NEED a more elegant way to group hashtables together - array?
	multimap<int,void*>* getIntHashTable(string tableName) {
		map<string, multimap<int,void*>*>::iterator it;
		it = intHTs.find(tableName);
		if (it == intHTs.end()) {
#ifdef DEBUG
			LOG(INFO) << "Creating HT for table "<< tableName;
#endif
			intHTs[tableName] = new multimap<int,void*>();
			return intHTs[tableName];
		}	else	{
#ifdef DEBUG
			LOG(INFO) << "HT found for " << tableName;
#endif
			return intHTs[tableName];
		}
	}

	//TODO NEED a more elegant way to group hashtables together - array?
	multimap<size_t,void*>* getHashTable(string tableName) {
		map<string, multimap<size_t,void*>*>::iterator it;
		it = HTs.find(tableName);
		if (it == HTs.end()) {
			LOG(INFO) << "Creating HT for table "<<tableName;
			HTs[tableName] = new multimap<size_t,void*>();
			return HTs[tableName];
		}	else	{
			LOG(INFO) << "HT found";
			return HTs[tableName];
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
			LOG(ERROR) << "Info in catalog does not exist for table " << tableName;
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
			/**
			 * Not an error any more.
			 * Reason: Some operators (e.g. outerNull) duplicate code after them
			 * => Some utility functions end up being called twice
			 */
			LOG(INFO) << "Info in catalog already exists for table " << tableName;
//			throw runtime_error(string("Info in catalog already exists for table ")+tableName);
		}
	}

	int getUniqueId()	{
		uniqueTableId++;
		return uniqueTableId;
	}

	void registerFileJSON(string fileName, ExpressionType* type) {
		map<string, ExpressionType*>::iterator it = jsonTypeCatalog.find(fileName);
		if(it == jsonTypeCatalog.end())	{
			LOG(WARNING) << "Catalog already contains the type of " << fileName;
		}
		jsonTypeCatalog[fileName] = type;
	}

	ExpressionType* getTypeJSON(string fileName) {
		map<string, ExpressionType*>::iterator it = jsonTypeCatalog.find(fileName);
		if(it == jsonTypeCatalog.end())	{
			LOG(ERROR) << "Catalog does not contain the type of " << fileName;
			throw runtime_error(string("Catalog does not contain the type of ")+fileName);
		}
		return it->second;
	}

	void registerPlugin(string fileName, Plugin* pg)	{
		map<string, Plugin*>::iterator it = plugins.find(fileName);
		if(it == plugins.end())	{
			LOG(WARNING) << "Catalog already contains the plugin of " << fileName;
		}
		plugins[fileName] = pg;
	}

	Plugin* getPlugin(string fileName)	{
		map<string, Plugin*>::iterator it = plugins.find(fileName);
		if(it == plugins.end())	{
			string error_msg = string("Catalog does not contain the plugin of ") + fileName;
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}
		return it->second;
	}

	map<int,Value*>* getReduceHT()	{
		if(reduceSetHT == NULL)	{
			string error_msg = string("[Catalog: ] HT to be used in Reduce not initialized");
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}
		return reduceSetHT;
	}

	void setReduceHT(map<int,Value*> *reduceSetHT)	{ this->reduceSetHT = reduceSetHT; }

//	Writer<StringBuffer> getJSONFlusher(string fileName)
//	{
//		map<string, Writer<StringBuffer>>::iterator it;
//		it = jsonFlushers.find(fileName);
//		if (it == jsonFlushers.end())
//		{
//			LOG(INFO) << "Creating Writer/Flusher, flushing to " << fileName;
//			StringBuffer s;
//			Writer<StringBuffer> writer(s);
//			(this->jsonFlushers)[fileName] = writer;
//			return writer;
//		}
//		else
//		{
//			return jsonFlushers[fileName];
//		}
//	}

	stringstream* getSerializer(string fileName)
	{
		map<string, stringstream*>::iterator it;
		it = serializers.find(fileName);
		if (it == serializers.end())
		{
			LOG(INFO) << "Creating Serializer, flushing to " << fileName;
			stringstream* strBuffer = new stringstream();
			(this->serializers)[fileName] = strBuffer;
			return strBuffer;
		}
		else
		{
			return serializers[fileName];
		}
	}

	void clear();

private:
	map<string,Plugin*> 					plugins;
	map<string,multimap<int,void*>*> 		intHTs;
	map<string,multimap<size_t,void*>*> 	HTs;
	map<string, ExpressionType*> 			jsonTypeCatalog;
//	map<string, Writer<StringBuffer>> 		jsonFlushers;
	map<string, stringstream*> 				serializers;
	//Initialized by Reduce() if accumulator type is set
	map<int,Value*> 						*reduceSetHT;

	/**
	 * Reason for this: The hashtables we populate (intHTs etc) are created a priori.
	 * Therefore, the 'values' need to be void* to accommodate any kind of 'tuples'
	 * XXX [Actually, is there a more flexible way to handle this? Templates or sth??]
	 */
	map<string,int> resultTypes;

	int uniqueTableId;
	int maxTables;

	BasicBlock* joinInsertionPoint;

	//Position 0 not used, so that we can use it to perform containment checks when using tableTypes
	Type** tableTypes;
	//Is maxTables enough????
	RawCatalog()	: uniqueTableId(1) , maxTables(1000), joinInsertionPoint(NULL), reduceSetHT(NULL) {
		tableTypes = new Type*[maxTables];
	}
	~RawCatalog() {
		if(reduceSetHT != NULL)	{
			delete reduceSetHT;
		}
	}

	//Not implementing; RawCatalog is a singleton
	RawCatalog(RawCatalog const&);     // Don't Implement.
	void operator=(RawCatalog const&); // Don't implement
};


#endif /* RAW_CATALOG_HPP_ */
