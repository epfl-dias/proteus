/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
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

#include "plugins/plugins.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class GpuColScanPlugin   : public Plugin {
public:

    /**
     * Plugin for scanning columns on the gpu side.
     *
     * Support for
     * -> integers
     * -> floats
     * -> chars & booleans
     *
     *
     * Conventions / Assumptions:
     * -> integers, floats, chars and booleans are stored as tight arrays
     * -> varchars are not yet supported
     * -> dates require more sophisticated serialization (boost?)
     */

    GpuColScanPlugin(GpuRawContext* const context, string fnamePrefix, RecordType rec, vector<RecordAttribute*>& whichFields, RawOperator* const child);
//  GpuColScanPlugin(GpuRawContext* const context, vector<RecordAttribute*>& whichFields, vector<CacheInfo> whichCaches);
    ~GpuColScanPlugin();
    virtual string& getName() { return fnamePrefix; }
    void init();
//  void initCached();
    void generate(const RawOperator& producer);
    void finish();
    virtual RawValueMemory readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr);
    virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);
    virtual RawValue readCachedValue(CacheInfo info, const OperatorState& currState);
//  {
//      string error_msg = "[GpuColScanPlugin: ] No caching support should be needed";
//      LOG(ERROR) << error_msg;
//      throw runtime_error(error_msg);
//  }
    virtual RawValue readCachedValue(CacheInfo info,
            const map<RecordAttribute, RawValueMemory>& bindings);

    virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);
    virtual RawValue hashValueEager(RawValue value, const ExpressionType* type);

    virtual RawValueMemory initCollectionUnnest(RawValue val_parentObject) {
        string error_msg = "[GpuColScanPlugin: ] Binary col. files do not contain collections";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }
    virtual RawValue collectionHasNext(RawValue val_parentObject,
            RawValueMemory mem_currentChild) {
        string error_msg = "[GpuColScanPlugin: ] Binary col. files do not contain collections";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }
    virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentChild) {
        string error_msg = "[GpuColScanPlugin: ] Binary col. files do not contain collections";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushTuple(RawValueMemory mem_value, llvm::Value* fileName) {
            string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
        }

    virtual void flushValue(RawValueMemory mem_value, const ExpressionType *type, llvm::Value* fileName)  {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushValueEager(RawValue value, const ExpressionType *type, llvm::Value* fileName) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushChunk(RawValueMemory mem_value, llvm::Value* fileName)  {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }

    virtual llvm::Value* getValueSize(RawValueMemory mem_value, const ExpressionType* type);

    virtual ExpressionType *getOIDType() {
        return new Int64Type();
    }

    virtual PluginType getPluginType() { return PGBINARY; }

    RawOperator * getChild(){ return child; }

    virtual void flushBeginList (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushBeginBag  (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushBeginSet  (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushEndList   (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushEndBag    (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushEndSet    (llvm::Value *fileName                    ) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushDelim     (llvm::Value *fileName                    , int depth) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    virtual void flushDelim     (llvm::Value *resultCtr, llvm::Value* fileName  , int depth) {
        string error_msg = "[GpuColScanPlugin: ] Flush not implemented yet";
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }
private:
    //Schema info provided
    RecordType rec;
    vector<RecordAttribute*>  wantedFields;
    vector<int>               wantedFieldsArg_id;
    int                       tupleCntArg_id;

    RawOperator* const        child;

    /* Used when we treat the col. files as internal caches! */
    bool isCached;
    vector<CacheInfo> whichCaches;

    string fnamePrefix;
    off_t *colFilesize; //Size of each column
    int *fd; //One per column
    char **buf;

    //Mapping attrNumber to
    //  -> file descriptor of its dictionary
    //  -> file size of its dictionary
    //  -> mapped input of its dictionary
    map<int,int> dictionaries;
    map<int,off_t> dictionaryFilesizes;
    //Note: char* buf can be cast to appropriate struct
    //Struct will probably look like { implicit_oid, len, char* }
    map<int,char*> dictionariesBuf;

    /**
     * Code-generation-related
     */
    //Used to store memory positions of offset, buf and filesize in the generated code
    map<string, AllocaInst*> NamedValuesBinaryCol;
    GpuRawContext* const context;
    //To be initialized by init(). Dictates # of loops
    llvm::Value *val_size;

    const char* posVar;     // = "offset";
    const char* bufVar;     // = "fileBuffer";
    const char* __attribute__((unused)) fsizeVar;   // = "fileSize";
    const char* __attribute__((unused)) sizeVar;    // = "size";
    const char* itemCtrVar; // = "itemCtr";

    void nextEntry();

    //Generates a for loop that performs the file scan
    void scan(const RawOperator& producer);
};
