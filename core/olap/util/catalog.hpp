/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#ifndef CATALOG_HPP_
#define CATALOG_HPP_

#include "common/common.hpp"
#include "plugins/plugins.hpp"
#include "values/expressionTypes.hpp"

// Forward Declaration
class Plugin;

// using namespace llvm;

// FIXME no need to be a singleton (I think)
class Catalog {
 public:
  static Catalog &getInstance();

  // TODO REPLACE ONCE REMOVED FROM JOIN
  // TODO NEED a more elegant way to group hashtables together - array?
  int getIntHashTableID(string tableName) {
    map<string, int>::iterator it;
    it = htIdentifiers.find(tableName);

    if (it != htIdentifiers.end()) {
      cout << "HT for label " << tableName << " found" << endl;
      return it->second;
    } else {
      cout << "NEW HT for label " << tableName << endl;
      int newIdentifier = intHashtables.size();
      multimap<int, void *> *newHT = new multimap<int, void *>();
      intHashtables.push_back(newHT);
      htIdentifiers[tableName] = newIdentifier;
      return newIdentifier;
    }
  }

  multimap<int, void *> *getIntHashTable(int tableID) {
    return intHashtables[tableID];
  }

  // TODO NEED a more elegant way to group hashtables together - array?
  multimap<size_t, void *> *getHashTable(string tableName) {
    map<string, multimap<size_t, void *> *>::iterator it;
    it = HTs.find(tableName);
    if (it == HTs.end()) {
      LOG(INFO) << "Creating HT for table " << tableName;
      HTs[tableName] = new multimap<size_t, void *>();
      return HTs[tableName];
    } else {
      LOG(INFO) << "HT found";
      return it->second;
    }
  }

  llvm::Type *getType(string tableName) {
    map<string, int>::iterator it;
    it = resultTypes.find(tableName);
    if (it == resultTypes.end()) {
      LOG(ERROR) << "Could not locate type of table " << tableName;
      throw runtime_error(string("Could not locate type of table ") +
                          tableName);
    } else {
      int idx = resultTypes[tableName];
      return tableTypes[idx];
    }
  }

  int getTypeIndex(string tableName) {
    map<string, int>::iterator it;
    it = resultTypes.find(tableName);
    if (it == resultTypes.end()) {
      LOG(ERROR) << "Info in catalog does not exist for table " << tableName;
      throw runtime_error(string("Info in catalog does not exist for table ") +
                          tableName);
    } else {
      int idx = resultTypes[tableName];
      return idx;
    }
  }

  llvm::Type *getTypeInternal(int idx) { return tableTypes[idx]; }

  void insertTableInfo(string tableName, llvm::Type *type) {
    int idx = resultTypes[tableName];
    if (idx == 0) {
      idx = getUniqueId();
      resultTypes[tableName] = idx;
      tableTypes[idx] = type;
    } else {
      /**
       * Not an error any more.
       * Reason: Some operators (e.g. outerNull) duplicate code after them
       * => Some utility functions end up being called twice
       */
      LOG(INFO) << "Info in catalog already exists for table " << tableName;
      // throw runtime_error(string("Info in catalog already exists for table
      // ")+tableName);
    }
  }

  int getUniqueId() {
    uniqueTableId++;
    return uniqueTableId;
  }

  void registerFileJSON(string fileName, ExpressionType *type) {
    auto it = jsonTypeCatalog.find(fileName);
    if (it != jsonTypeCatalog.end()) {
      LOG(WARNING) << "Catalog already contains the type of " << fileName;
    }
    jsonTypeCatalog[fileName] = type;
  }

  ExpressionType *getTypeJSON(string fileName) {
    auto it = jsonTypeCatalog.find(fileName);
    if (it == jsonTypeCatalog.end()) {
      LOG(ERROR) << "Catalog does not contain the type of " << fileName;
      throw runtime_error(string("Catalog does not contain the type of ") +
                          fileName);
    }
    return it->second;
  }

  void registerPlugin(string fileName, Plugin *pg) {
    auto it = this->plugins.find(fileName);
    if (it != this->plugins.end()) {
      LOG(WARNING) << "Catalog already contains the plugin of " << fileName;
    }
    this->plugins[fileName] = pg;
  }

  Plugin *getPlugin(string fileName) {
    auto it = this->plugins.find(fileName);
    if (it == this->plugins.end()) {
      string error_msg =
          string("Catalog does not contain the plugin of ") + fileName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    return it->second;
  }

  map<int, llvm::Value *> *getReduceHT() {
    if (reduceSetHT == nullptr) {
      string error_msg =
          string("[Catalog: ] HT to be used in Reduce not initialized");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    return reduceSetHT;
  }

  void setReduceHT(map<int, llvm::Value *> *reduceSetHT) {
    this->reduceSetHT = reduceSetHT;
  }

  //    Writer<StringBuffer> getJSONFlusher(string fileName)
  //    {
  //        map<string, Writer<StringBuffer>>::iterator it;
  //        it = jsonFlushers.find(fileName);
  //        if (it == jsonFlushers.end())
  //        {
  //            LOG(INFO) << "Creating Writer/Flusher, flushing to " <<
  //            fileName; StringBuffer s; Writer<StringBuffer> writer(s);
  //            (this->jsonFlushers)[fileName] = writer;
  //            return writer;
  //        }
  //        else
  //        {
  //            return jsonFlushers[fileName];
  //        }
  //    }

  stringstream &getSerializer(string fileName) {
    map<string, stringstream *>::iterator it;
    it = serializers.find(fileName);
    if (it == serializers.end()) {
      LOG(INFO) << "Creating Serializer, flushing to " << fileName;
      stringstream *strBuffer = new stringstream();
      (this->serializers)[fileName] = strBuffer;
      return *strBuffer;
    } else {
      return *serializers[fileName];
    }
  }

  void clear() {
    htIdentifiers.clear();
    // NOTE: Clear != Free()
    intHashtables.clear();
    jsonTypeCatalog.clear();
    resultTypes.clear();
    plugins.clear();
    fprintf(stderr, "Catalog cleared!\n");
  }

 private:
  map<string, Plugin *> plugins;
  map<string, int> htIdentifiers;

  vector<multimap<int, void *> *> intHashtables;
  map<string, multimap<size_t, void *> *> HTs;

  map<string, ExpressionType *> jsonTypeCatalog;
  //    map<string, Writer<StringBuffer>>         jsonFlushers;
  map<string, stringstream *> serializers;
  // Initialized by Reduce() if accumulator type is set
  map<int, llvm::Value *> *reduceSetHT;

  /**
   * Reason for this: The hashtables we populate (intHTs etc) are created a
   * priori. Therefore, the 'values' need to be void* to accommodate any kind of
   * 'tuples'
   * XXX [Actually, is there a more flexible way to handle this? Templates or
   * sth??]
   */
  map<string, int> resultTypes;

  int uniqueTableId;
  int maxTables;

  // Position 0 not used, so that we can use it to perform containment checks
  // when using tableTypes
  llvm::Type **tableTypes;
  // Is maxTables enough????
  Catalog() : uniqueTableId(1), maxTables(1000), reduceSetHT(nullptr) {
    tableTypes = new llvm::Type *[maxTables];
  }
  ~Catalog() {
    if (reduceSetHT != nullptr) {
      delete reduceSetHT;
    }
  }

  // Not implementing; Catalog is a singleton
  Catalog(Catalog const &);         // Don't Implement.
  void operator=(Catalog const &);  // Don't implement
};

#endif /* CATALOG_HPP_ */
