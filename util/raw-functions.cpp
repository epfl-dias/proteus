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

#include "util/raw-functions.hpp"

//Remember to add these functions as extern in .hpp too!
extern "C" double putchari(int X) {
	putchar((char) X);
	return 0;
}

void printBoolean(bool in)	{
	if(in)	{
		printf("True\n");
	}	else	{
		printf("False\n");
	}
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
int printi(int X) {
#ifdef DEBUG
	printf("[printi:] Generated code called %d\n", X);
#else
	printf("%d\n", X);
#endif
	return 0;
}

int printShort(short X) {
	printf("[printShort:] Generated code called %d\n", X);
	return 0;
}

int printFloat(double X) {
#ifdef DEBUG
	printf("[printFloat:] Generated code called %f\n", X);
#else
	printf("%f\n", X);
#endif

	return 0;
}

int printi64(size_t X) {
	printf("[printi64:] Debugging int64, not size_t: %ld\n", X);

//	printf("[printi64:] Generated code called %lu\n", X);

	//This is the appropriate one...
//	printf("[printi64:] Generated code called %zu\n", X);
	//cout <<"[printi64:] Generated code called "<< X<< endl;
	return 0;
}

int printc(char* X) {
	printf("[printc:] Generated code -- char read: %c\n", X[0]);
	return 0;
}

void resetTime()	{
	stopwatch_t& timer = stopwatch_t::getInstance();
	timer.reset();
}

void calculateTime()	{
	stopwatch_t& timer = stopwatch_t::getInstance();
	double elapsedTime = timer.time_ms();
	printf("Operation took %f msec\n",elapsedTime);
}


//int s(const char* X) {
//	//printf("Generated code -- char read: %c\n", X[0]);
//	return atoi(X);
//}

void insertToHT(char* HTname, size_t key, void* value, int type_size) {
	RawCatalog& catalog = RawCatalog::getInstance();
	//still, one unneeded indirection..... is there a quicker way?
	multimap<size_t, void*>* HT = catalog.getHashTable(string(HTname));

	void* valMaterialized = malloc(type_size);
	memcpy(valMaterialized, value, type_size);

	HT->insert(pair<size_t, void*>(key, valMaterialized));

	//	HT->insert(pair<int,void*>(key,value));
	LOG(INFO) << "[Insert: ] Hash key " << key << " inserted successfully";

	LOG(INFO) << "[INSERT: ] There are " << HT->count(key)
			<< " elements with key " << key << ":";

}

void** probeHT(char* HTname, size_t key) {

	string name = string(HTname);
	RawCatalog& catalog = RawCatalog::getInstance();

	//same indirection here as above.
	multimap<size_t, void*>* HT = catalog.getHashTable(name);

	pair<multimap<size_t, void*>::iterator, multimap<size_t, void*>::iterator> results;
	results = HT->equal_range(key);

	void** bindings = 0;
	int count = HT->count(key);
	LOG(INFO) << "[PROBE:] There are " << HT->count(key)
			<< " elements with hash key " << key;
	if (count) {
		//+1 used to set last position to null and know when to terminate
		bindings = new void*[count + 1];
		bindings[count] = NULL;
	} else {
		bindings = new void*[1];
		bindings[0] = NULL;
		return bindings;
	}

	int curr = 0;
	for (multimap<size_t, void*>::iterator it = results.first;
			it != results.second; ++it) {
		bindings[curr] = it->second;
		curr++;
	}
	return bindings;
}

/**
 * TODO
 * Obviously extremely inefficient.
 * Once having replaced multimap for our own code,
 * we also need to gather this metadata at build time.
 *
 * Examples: Number of buckets (keys) / elements in each bucket
 */
HashtableBucketMetadata* getMetadataHT(char* HTname)	{
	string name = string(HTname);
	RawCatalog& catalog = RawCatalog::getInstance();

	//same indirection here as above.
	multimap<size_t, void*>* HT = catalog.getHashTable(name);

	vector<size_t> keys;
	for (multimap<size_t, void*>::iterator it = HT->begin(), end = HT->end();
			it != end; it = HT->upper_bound(it->first))
	{
		keys.push_back(it->first);
		//cout << it->first << ' ' << it->second << endl;
	}
	HashtableBucketMetadata *metadata = new HashtableBucketMetadata[keys.size() + 1];
	size_t pos = 0;
	for(vector<size_t>::iterator it = keys.begin(); it != keys.end(); it++ , pos++)	{
		metadata[pos].hashKey = *it;
		metadata[pos].bucketSize = HT->count(*it);
	}
	//XXX Silly stopping condition..
	metadata[pos].bucketSize = 0;
	return metadata;
}

/* Deprecated */
void insertIntKeyToHT(int htIdentifier, int key, void* value, int type_size) {
	RawCatalog& catalog = RawCatalog::getInstance();
	//still, one unneeded indirection..... is there a quicker way?
	multimap<int, void*>* HT = catalog.getIntHashTable(htIdentifier);

	void* valMaterialized = malloc(type_size);
	//FIXME obviously expensive, but probably cannot be helped
	memcpy(valMaterialized, value, type_size);

	HT->insert(pair<int, void*>(key, valMaterialized));
//	cout << "INSERTED KEY " << key << endl;

#ifdef DEBUG
//	LOG(INFO) << "[Insert: ] Integer key " << key << " inserted successfully";
//
//	LOG(INFO) << "[INSERT: ] There are " << HT->count(key)
//			<< " elements with key " << key << ":";
#endif

}

/* Deprecated */
void** probeIntHT(int htIdentifier, int key, int typeIndex) {

//	string name = string(HTname);
	RawCatalog& catalog = RawCatalog::getInstance();

	//same indirection here as above.
	multimap<int, void*>* HT = catalog.getIntHashTable(htIdentifier);

	pair<multimap<int, void*>::iterator, multimap<int, void*>::iterator> results;
	results = HT->equal_range(key);

	void** bindings = 0;
	int count = HT->count(key);

	if (count) {
		//+1 used to set last position to null and know when to terminate
		bindings = new void*[count + 1];
		bindings[count] = NULL;
	} else {
		bindings = new void*[1];
		bindings[0] = NULL;
		return bindings;
	}

	int curr = 0;
	for (multimap<int, void*>::iterator it = results.first;
			it != results.second; ++it) {
		bindings[curr] = it->second;
		curr++;
	}
#ifdef DEBUG
	LOG(INFO) << "[PROBE INT:] There are " << HT->count(key)
			<< " elements with key " << key;
#endif
	return bindings;
}

bool equalStringObjs(StringObject obj1, StringObject obj2)	{
//	cout << obj1.start << " with len " << obj1.len << endl;
//	cout << obj2.start << " with len " << obj2.len << endl;
	if(obj1.len != obj2.len)	{
		return false;
	}
	if(strncmp(obj1.start,obj2.start,obj1.len) != 0)	{
		return false;
	}
	return true;
}

bool equalStrings(char *str1, char *str2)	{
	return strcmp(str1,str2) == 0;
}

int compareTokenString(const char* buf, int start, int end, const char* candidate)	{
//	cout << "Candidate?? " << candidate << endl;
//	cout << "Buf?" << start << " " << end << endl;
	return (strncmp(buf + start, candidate, end - start) == 0 \
			&& strlen(candidate) == end - start);
}

int compareTokenString64(const char* buf, size_t start, size_t end, const char* candidate)	{
//	cout << "Start? " << start << endl;
//	cout << "End? " << end << endl;
//	cout << "Candidate?? " << candidate << endl;
//	char *deleteme = (char*) malloc(end - start +1);
//	memcpy(deleteme,buf+start,end-start);
//	deleteme[end-start] = '\0';
//	cout << "From file: " << deleteme << endl;
	return (strncmp(buf + start, candidate, end - start) == 0 \
			&& strlen(candidate) == end - start);
}

bool convertBoolean(const char* buf, int start, int end)	{
	if (compareTokenString(buf, start, end, "true") == 1
			|| compareTokenString(buf, start, end, "TRUE") == 1) {
		return true;
	} else if (compareTokenString(buf, start, end, "false") == 1
			|| compareTokenString(buf, start, end, "FALSE") == 1) {
		return false;
	} else {
		string error_msg = string("[convertBoolean: Error - unknown input]");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
}

bool convertBoolean64(const char* buf, size_t start, size_t end)	{
	if (compareTokenString64(buf, start, end, "true") == 1
			|| compareTokenString64(buf, start, end, "TRUE") == 1) {
		return true;
	} else if (compareTokenString64(buf, start, end, "false") == 1
			|| compareTokenString64(buf, start, end, "FALSE") == 1) {
		return false;
	} else {
		string error_msg = string("[convertBoolean64: Error - unknown input]");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
}

size_t hashInt(int toHash)	{
	boost::hash<int> hasher;
	return hasher(toHash);
}

size_t hashDouble(double toHash) {
	boost::hash<double> hasher;
	return hasher(toHash);
}

size_t hashString(string toHash)	{
	boost::hash<string> hasher;
	size_t result = hasher(toHash);
	return result;
}

//XXX Copy string? Or edit in place?
size_t hashStringC(char* toHash, size_t start, size_t end)	{
	char tmp = toHash[end];
	toHash[end] = '\0';
	boost::hash<string> hasher;
	size_t result = hasher(toHash + start);
	toHash[end] = tmp;
	return result;
}

size_t hashBoolean(bool toHash) {
	boost::hash<bool> hasher;
	return hasher(toHash);
}

size_t hashStringObject(StringObject obj)	{
	char tmp = obj.start[obj.len+1];
	obj.start[obj.len+1] = '\0';
	boost::hash<string> hasher;
	size_t result = hasher(obj.start);
	obj.start[obj.len+1] = tmp;
	return result;
}


//size_t combineHashes(size_t hash1, size_t hash2) {
//	 size_t seed = 0;
//	 boost::hash_combine(seed, hash1);
//	 boost::hash_combine(seed, hash2);
//	 return seed;
//}
//
//template <class T>
//inline void hash_combine_no_order(size_t& seed, const T& v)
//{
//    boost::hash<T> hasher;
//    seed ^= hasher(v);
//}
//
//size_t combineHashesNoOrder(size_t hash1, size_t hash2) {
//	 size_t seed = 0;
//	 hash_combine_no_order(seed, hash1);
//	 hash_combine_no_order(seed, hash2);
//	 return seed;
//}

size_t combineHashes(size_t hash1, size_t hash2) {
	 boost::hash_combine(hash1, hash2);
	 return hash1;
}

template <class T>
inline void hash_combine_no_order(size_t& seed, const T& v)
{
    boost::hash<T> hasher;
    seed ^= hasher(v);
}

size_t combineHashesNoOrder(size_t hash1, size_t hash2) {
	 hash_combine_no_order(hash1, hash2);
	 return hash1;
}

/**
 * Radix chunks of functionality
 */
int *partitionHTLLVM(size_t num_tuples, joins::tuple_t *inTuples)	{
	return partitionHT(num_tuples, inTuples);
}

void bucket_chaining_join_prepareLLVM(const joins::tuple_t * const tuplesR,
		int num_tuples, HT * ht) {
	bucket_chaining_join_prepare(tuplesR, num_tuples, ht);
}

void bucket_chaining_agg_prepareLLVM(const agg::tuple_t * const tuplesR,
		int num_tuples, HT * ht) {
	bucket_chaining_agg_prepare(tuplesR, num_tuples, ht);
}

int *partitionAggHTLLVM(size_t num_tuples, agg::tuple_t *inTuples)	{
	return partitionHT(num_tuples, inTuples);
}


/**
 * Flushing data.
 * Issue with standard flusher for now:
 * Cannot 'cheat' and pass along JSON serialized data
 * without having to first deserialize them
 */
//void flushInt(int toFlush, char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.Uint(toFlush);
//}
//
//void flushDouble(double toFlush, char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.Double(toFlush);
//}
//
//void flushBoolean(bool toFlush, char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.Bool(toFlush);
//}
//
//void flushStringC(char* toFlush, size_t start, size_t end, char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	char tmp = toFlush[end + 1 - start];
//	toFlush[end+1] = '\0';
//	w.String(toFlush);
//	toFlush[end+1] = tmp;
//}
//
///**
// * flushString: Not used atm
// * Careful: Cannot be used from static code!
// * It's going to be executed and flush to JSON file
// * before actual 'query' execution
// */
//void flushString(string toFlush, char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.String(toFlush.c_str());
//}
//
//void flushObjectStart(char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.StartObject();
//}
//
//void flushArrayStart(char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.StartArray();
//}
//
//void flushObjectEnd(char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.EndObject();
//}
//
//void flushArrayEnd(char* fileName)	{
//	RawCatalog& catalog = RawCatalog::getInstance();
//	string name = string(fileName);
//	Writer<StringBuffer> w = catalog.getJSONFlusher(name);
//	w.EndArray();
//}

void flushInt(int toFlush, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream *strBuffer = catalog.getSerializer(name);
	(*strBuffer) << toFlush;
}

void flushDouble(double toFlush, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream *strBuffer = catalog.getSerializer(name);
		(*strBuffer) << toFlush;
}

void flushBoolean(bool toFlush, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream *strBuffer = catalog.getSerializer(name);
		(*strBuffer) << toFlush;
}

//FIXME Bug here
void flushStringC(char* toFlush, size_t start, size_t end, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	char tmp = toFlush[end];
	toFlush[end] = '\0';
	(*strBuffer) << (toFlush + start);
	toFlush[end] = tmp;
}

void flushStringReady(char* toFlush, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "\"";
	(*strBuffer) << toFlush;
	(*strBuffer) << "\"";
}

void flushStringObject(StringObject obj, char* fileName) {
	char tmp = obj.start[obj.len + 1];
	obj.start[obj.len + 1] = '\0';

	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "\"";
	(*strBuffer) << obj.start;
	(*strBuffer) << "\"";

	obj.start[obj.len + 1] = tmp;

}

void flushObjectStart(char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "{";
}

void flushArrayStart(char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "}";
}

void flushObjectEnd(char* fileName)		{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "[";
}

void flushArrayEnd(char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << "]";
}

void flushChar(char whichChar, char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
	string name = string(fileName);
	stringstream* strBuffer = catalog.getSerializer(name);
	(*strBuffer) << whichChar;
}

void flushDelim(size_t resultCtr, char whichDelim, char* fileName) {
	RawCatalog& catalog = RawCatalog::getInstance();
	if (likely(resultCtr > 0)) {
		flushChar(whichDelim, fileName);
	}
}

void flushOutput(char* fileName)	{
	RawCatalog& catalog = RawCatalog::getInstance();
		string name = string(fileName);
		stringstream* strBuffer = catalog.getSerializer(name);
		ofstream outFile;
		outFile.open(fileName);
		outFile << strBuffer->rdbuf();
}

/**
 * Memory mgmt
 */

void* getMemoryChunk(size_t chunkSize)	{
	return allocateFromRegion(chunkSize);
}

void* increaseMemoryChunk(void* chunk, size_t chunkSize)	{
	return increaseRegion(chunk, chunkSize);
}

void releaseMemoryChunk(void* chunk)	{
	return freeRegion(chunk);
}

/**
 * Parsing
 */
/*
 * Return position of \n
 * Code from
 * https://www.klittlepage.com/2013/12/10/accelerated-fix-processing-via-avx2-vector-instructions/
 *
 * XXX Assumption: all lines of file end with \n
 */
/*
As we're looking for simple, single character needles (newlines) we can use bitmasking to
search in lieu of SSE 4.2 string comparison functions. This simple
implementation splits a 256 bit AVX register into eight 32-bit words. Whenever
a word is non-zero (any bits are set within the word) a linear scan identifies
the position of the matching character within the 32-bit word.
*/
//__attribute__((always_inline))
//inline
size_t newlineAVX(const char* const target, size_t targetLength) {
	char nl = '\n';
#ifdef	__AVX2__
//	cout << "AVX mode ON" << endl;
	__m256i eq = _mm256_set1_epi8(nl);
	size_t strIdx = 0;
	union {
		__m256i v;
		char c[32];
	}testVec;
	union {
		__m256i v;
		uint32_t l[8];
	}mask;

	if(targetLength >= 32) {
		for(; strIdx <= targetLength - 32; strIdx += 32) {
			testVec.v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
							target + strIdx));
			mask.v = _mm256_cmpeq_epi8(testVec.v, eq);
			for(int i = 0; i < 8; ++i) {
				if(0 != mask.l[i]) {
					for(int j = 0; j < 4; ++j) {
						char c = testVec.c[4 * i + j];
						if(nl == c) {
//							cout << "1. NL at pos (" << strIdx << "+" << 4*i+j << ")" << endl;
							return strIdx + 4 * i + j;
						}
					}
				}
			}
		}
	}

	for(; strIdx < targetLength; ++strIdx) {
		const char c = target[strIdx];
		if(nl == c) {
//			cout << "2. NL at pos " << strIdx << endl;
			return strIdx;
		}
	}

	string error_msg = string("No newline found");
	LOG(ERROR)<< error_msg;
#endif
#ifndef __AVX2__
	//cout << "Careful: Non-AVX parsing" << endl;
	int i = 0;
	while(target[i] != nl && i < targetLength)	{
		i++;
	}
//	if(i == targetLength && target[i] != nl)	{
//		string error_msg = string("No newline found");
//			LOG(ERROR)<< error_msg;
//	}
	//cout << "Newline / End of line at pos " << i << endl;
	return i;
#endif

}

//void parseLineJSON(char *buf, size_t start, size_t end, jsmntok_t** tokens, size_t line)	{
//
//	int error_code;
//	jsmn_parser p;
//
//	/* inputs/json/jsmnDeeper-flat.json : MAXTOKENS = 25 */
//
//	//Populating our json 'positional index'
//	jsmntok_t* tokenArray = (jsmntok_t*) calloc(MAXTOKENS,sizeof(jsmntok_t));
//	if(tokens == NULL)
//	{
//		throw runtime_error(string("new() of tokens failed"));
//	}
//
//	jsmn_init(&p);
//	char* bufShift = buf + start;
//	char eol = buf[end];
//	buf[end] = '\0';
////	printf("JSON Raw Input: %s\n",bufShift);
////	printf("Which line? %d\n",line);
//	error_code = jsmn_parse(&p, bufShift, end - start, tokenArray, MAXTOKENS);
//	buf[end] = eol;
//	if(error_code < 0)
//	{
//		string msg = "Json (JSMN) plugin failure: ";
//		LOG(ERROR) << msg << error_code;
//		throw runtime_error(msg);
//	}
////	else
////	{
////		cout << "How many tokens?? " << error_code << endl;
////	}
//	tokens[line] = tokenArray;
////	cout << "[parseLineJSON: ] " << tokenArray[0].start << " to " << tokenArray[0].end << endl;
//}

void parseLineJSON(char *buf, size_t start, size_t end, jsmntok_t** tokens, size_t line)	{

//	cout << "[parseLineJSON: ] Entry for line " << line << " from " << start << " to " << end << endl;
	int error_code;
	jsmn_parser p;

	/* inputs/json/jsmnDeeper-flat.json : MAXTOKENS = 25 */

	//Populating our json 'positional index'

	jsmn_init(&p);
	char* bufShift = buf + start;
	char eol = buf[end];
	buf[end] = '\0';
//	error_code = jsmn_parse(&p, bufShift, end - start, tokens[line], MAXTOKENS);
	//printf("Before %ld %ld %ld\n",tokens,tokens + line, tokens[line]);
	size_t tokensNo = MAXTOKENS;
	error_code = jsmn_parse(&p, bufShift, end - start, &(tokens[line]), tokensNo);
	//printf("After %ld %ld\n",tokens,tokens[line]);
	buf[end] = eol;
//	if(line > 0 && (line +1)% 10000000 == 0)
//	{
//		printf("Processing line no. %ld\n",line);
//	}
	if(error_code < 0)
	{
		string msg = "Json (JSMN) plugin failure: ";
		LOG(ERROR) << msg << error_code << " in line " << line;
		throw runtime_error(msg);
	}
//	else
//	{
//		cout << "How many tokens?? " << error_code << " in line " << line << endl;
//	}
//	cout << "[parseLineJSON - " << line << ": ] "
//	<< tokens[line][0].start
//	<< " to " << tokens[line][0].end << endl;
//	cout << "[parseLineJSON - exit] "<< endl;
}



//'Inline' -> shouldn't it be placed in .hpp?
inline int atoi1(const char *buf) {
	return  (buf[0] - '0');
}

inline int atoi2(const char *buf) {
	return  ((buf[0] - '0') * 10) + \
			(buf[1] - '0');
}

inline int atoi3(const char *buf) {
	return  ((buf[0] - '0') * 100) + \
			((buf[1] - '0') * 10) + \
			(buf[2] - '0');
}

inline int atoi4(const char *buf) {
	return  ((buf[0] - '0') * 1000) + \
			((buf[1] - '0') * 100) + \
			((buf[2] - '0') * 10) + \
			(buf[3] - '0');
}

inline int atoi5(const char *buf) {
	return  ((buf[0] - '0') * 10000) + \
			((buf[1] - '0') * 1000) + \
			((buf[2] - '0') * 100) + \
			((buf[3] - '0') * 10) + \
			(buf[4] - '0');
}

inline int atoi6(const char *buf) {
	return  ((buf[0] - '0') * 100000) + \
			((buf[1] - '0') * 10000) + \
			((buf[2] - '0') * 1000) + \
			((buf[3] - '0') * 100) + \
			((buf[4] - '0') * 10) + \
			(buf[5] - '0');
}

inline int atoi7(const char *buf) {
	return  ((buf[0] - '0') * 1000000) + \
			((buf[1] - '0') * 100000) + \
			((buf[2] - '0') * 10000) + \
			((buf[3] - '0') * 1000) + \
			((buf[4] - '0') * 100) + \
			((buf[5] - '0') * 10) + \
			(buf[6] - '0');
}

inline int atoi8(const char *buf) {
	return  ((buf[0] - '0') * 10000000) + \
			((buf[1] - '0') * 1000000) + \
			((buf[2] - '0') * 100000) + \
			((buf[3] - '0') * 10000) + \
			((buf[4] - '0') * 1000) + \
			((buf[5] - '0') * 100) + \
			((buf[6] - '0') * 10) + \
			(buf[7] - '0');
}

inline int atoi9(const char *buf) {
	return  ((buf[0] - '0') * 100000000) + \
			((buf[1] - '0') * 10000000) + \
			((buf[2] - '0') * 1000000) + \
			((buf[3] - '0') * 100000) + \
			((buf[4] - '0') * 10000) + \
			((buf[5] - '0') * 1000) + \
			((buf[6] - '0') * 100) + \
			((buf[7] - '0') * 10) + \
			(buf[8] - '0');
}

inline int atoi10(const char *buf) {
	return  ((buf[0] - '0') * 1000000000) + \
			((buf[1] - '0') * 100000000) + \
			((buf[2] - '0') * 10000000) + \
			((buf[3] - '0') * 1000000) + \
			((buf[4] - '0') * 100000) + \
			((buf[5] - '0') * 10000) + \
			((buf[6] - '0') * 1000) + \
			((buf[7] - '0') * 100) + \
			((buf[8] - '0') * 10) + \
			(buf[9] - '0');
}

int atois(const char *buf, int len) {
	switch (len) {
	case 1:
		return atoi1(buf);
	case 2:
		return atoi2(buf);
	case 3:
		return atoi3(buf);
	case 4:
		return atoi4(buf);
	case 5:
		return atoi5(buf);
	case 6:
		return atoi6(buf);
	case 7:
		return atoi7(buf);
	case 8:
		return atoi8(buf);
	case 9:
		return atoi9(buf);
	case 10:
		return atoi10(buf);
	default:
		LOG(ERROR) << "[ATOIS: ] Invalid Size " << len;
		throw runtime_error(string("[ATOIS: ] Invalid Size "));
	}
}

void registerFunctions(RawContext& context)	{
	LLVMContext& ctx = context.getLLVMContext();
	Module* const TheModule = context.getModule();

	Type* int1_bool_type = Type::getInt1Ty(ctx);
	Type* int8_type = Type::getInt8Ty(ctx);
	Type* int16_type = Type::getInt16Ty(ctx);
	Type* int32_type = Type::getInt32Ty(ctx);
	Type* int64_type = Type::getInt64Ty(ctx);
	Type* void_type = Type::getVoidTy(ctx);
	Type* double_type = Type::getDoubleTy(ctx);
	StructType* strObjType = context.CreateStringStruct();
	PointerType* void_ptr_type = PointerType::get(int8_type, 0);
	PointerType* char_ptr_type = PointerType::get(int8_type, 0);
	PointerType* int32_ptr_type = PointerType::get(int32_type, 0);

	vector<Type*> Ints8Ptr(1,Type::getInt8PtrTy(ctx));
	vector<Type*> Ints8(1,int8_type);
	vector<Type*> Ints1(1,int1_bool_type);
	vector<Type*> Ints(1,int32_type);
	vector<Type*> Ints64(1,int64_type);
	vector<Type*> Floats(1,double_type);
	vector<Type*> Shorts(1,int16_type);

	vector<Type*> ArgsCmpTokens;
	ArgsCmpTokens.insert(ArgsCmpTokens.begin(),char_ptr_type);
	ArgsCmpTokens.insert(ArgsCmpTokens.begin(),int32_type);
	ArgsCmpTokens.insert(ArgsCmpTokens.begin(),int32_type);
	ArgsCmpTokens.insert(ArgsCmpTokens.begin(),char_ptr_type);

	vector<Type*> ArgsCmpTokens64;
	ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);
	ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
	ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
	ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);

	vector<Type*> ArgsConvBoolean;
	ArgsConvBoolean.insert(ArgsConvBoolean.begin(),int32_type);
	ArgsConvBoolean.insert(ArgsConvBoolean.begin(),int32_type);
	ArgsConvBoolean.insert(ArgsConvBoolean.begin(),char_ptr_type);

	vector<Type*> ArgsConvBoolean64;
	ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),int64_type);
	ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),int64_type);
	ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),char_ptr_type);

	vector<Type*> ArgsAtois;
	ArgsAtois.insert(ArgsAtois.begin(),int32_type);
	ArgsAtois.insert(ArgsAtois.begin(),char_ptr_type);

	vector<Type*> ArgsStringObjCmp;
	ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(),strObjType);
	ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(),strObjType);

	vector<Type*> ArgsStringCmp;
	ArgsStringCmp.insert(ArgsStringCmp.begin(), char_ptr_type);
	ArgsStringCmp.insert(ArgsStringCmp.begin(),char_ptr_type);

	/**
	 * Args of functions computing hash
	 */
	vector<Type*> ArgsHashInt;
	ArgsHashInt.insert(ArgsHashInt.begin(),int32_type);

	vector<Type*> ArgsHashDouble;
	ArgsHashDouble.insert(ArgsHashDouble.begin(),double_type);

	vector<Type*> ArgsHashStringC;
	ArgsHashStringC.insert(ArgsHashStringC.begin(),int64_type);
	ArgsHashStringC.insert(ArgsHashStringC.begin(),int64_type);
	ArgsHashStringC.insert(ArgsHashStringC.begin(),char_ptr_type);

	vector<Type*> ArgsHashStringObj;
	ArgsHashStringObj.insert(ArgsHashStringObj.begin(),strObjType);

	vector<Type*> ArgsHashBoolean;
	ArgsHashBoolean.insert(ArgsHashBoolean.begin(),int1_bool_type);

	vector<Type*> ArgsHashCombine;
	ArgsHashCombine.insert(ArgsHashCombine.begin(),int64_type);
	ArgsHashCombine.insert(ArgsHashCombine.begin(),int64_type);

	/**
	 * Args of functions computing flush
	 */
	vector<Type*> ArgsFlushInt;
	ArgsFlushInt.insert(ArgsFlushInt.begin(),char_ptr_type);
	ArgsFlushInt.insert(ArgsFlushInt.begin(),int32_type);

	vector<Type*> ArgsFlushDouble;
	ArgsFlushDouble.insert(ArgsFlushDouble.begin(),char_ptr_type);
	ArgsFlushDouble.insert(ArgsFlushDouble.begin(),double_type);

	vector<Type*> ArgsFlushStringC;
	ArgsFlushStringC.insert(ArgsFlushStringC.begin(),char_ptr_type);
	ArgsFlushStringC.insert(ArgsFlushStringC.begin(),int64_type);
	ArgsFlushStringC.insert(ArgsFlushStringC.begin(),int64_type);
	ArgsFlushStringC.insert(ArgsFlushStringC.begin(),char_ptr_type);

	vector<Type*> ArgsFlushStringCv2;
	ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(),char_ptr_type);
	ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(),char_ptr_type);

	vector<Type*> ArgsFlushStringObj;
	ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(),char_ptr_type);
	ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(),strObjType);

	vector<Type*> ArgsFlushBoolean;
	ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(),int1_bool_type);
	ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(),char_ptr_type);

	vector<Type*> ArgsFlushStartEnd;
	ArgsFlushStartEnd.insert(ArgsFlushStartEnd.begin(),char_ptr_type);

	vector<Type*> ArgsFlushChar;
	ArgsFlushChar.insert(ArgsFlushChar.begin(),char_ptr_type);
	ArgsFlushChar.insert(ArgsFlushChar.begin(),int8_type);

	vector<Type*> ArgsFlushDelim;
	ArgsFlushDelim.insert(ArgsFlushDelim.begin(),char_ptr_type);
	ArgsFlushDelim.insert(ArgsFlushDelim.begin(),int8_type);
	ArgsFlushDelim.insert(ArgsFlushDelim.begin(),int64_type);

	vector<Type*> ArgsFlushOutput;
	ArgsFlushOutput.insert(ArgsFlushOutput.begin(),char_ptr_type);

	vector<Type*> ArgsMemoryChunk;
	ArgsMemoryChunk.insert(ArgsMemoryChunk.begin(),int64_type);
	vector<Type*> ArgsIncrMemoryChunk;
	ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(),int64_type);
	ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(),void_ptr_type);
	vector<Type*> ArgsRelMemoryChunk;
	ArgsRelMemoryChunk.insert(ArgsRelMemoryChunk.begin(),void_ptr_type);

	/**
	 * Args of timing functions
	 */
	//Empty on purpose
	vector<Type*> ArgsTiming;


	FunctionType *FTint 				  =	FunctionType::get(Type::getInt32Ty(ctx), Ints, false);
	FunctionType *FTint64 				  = FunctionType::get(Type::getInt32Ty(ctx), Ints64, false);
	FunctionType *FTcharPtr 			  = FunctionType::get(Type::getInt32Ty(ctx), Ints8Ptr, false);
	FunctionType *FTatois 				  = FunctionType::get(int32_type, ArgsAtois, false);
	FunctionType *FTatof 				  = FunctionType::get(double_type, Ints8Ptr, false);
	FunctionType *FTprintFloat_ 		  = FunctionType::get(int32_type, Floats, false);
	FunctionType *FTprintShort_ 		  = FunctionType::get(int16_type, Shorts, false);
	FunctionType *FTcompareTokenString_   = FunctionType::get(int32_type, ArgsCmpTokens, false);
	FunctionType *FTcompareTokenString64_ = FunctionType::get(int32_type, ArgsCmpTokens64, false);
	FunctionType *FTconvertBoolean_ 	  = FunctionType::get(int1_bool_type, ArgsConvBoolean, false);
	FunctionType *FTconvertBoolean64_ 	  = FunctionType::get(int1_bool_type, ArgsConvBoolean64, false);
	FunctionType *FTprintBoolean_ 		  = FunctionType::get(void_type, Ints1, false);
	FunctionType *FTcompareStringObjs 	  = FunctionType::get(int1_bool_type, ArgsStringObjCmp, false);
	FunctionType *FTcompareString	  	  = FunctionType::get(int1_bool_type, ArgsStringCmp, false);
	FunctionType *FThashInt 			  = FunctionType::get(int64_type, ArgsHashInt, false);
	FunctionType *FThashDouble 			  = FunctionType::get(int64_type, ArgsHashDouble, false);
	FunctionType *FThashStringC 		  = FunctionType::get(int64_type, ArgsHashStringC, false);
	FunctionType *FThashStringObj 		  = FunctionType::get(int64_type, ArgsHashStringObj, false);
	FunctionType *FThashBoolean 		  = FunctionType::get(int64_type, ArgsHashBoolean, false);
	FunctionType *FThashCombine 		  = FunctionType::get(int64_type, ArgsHashCombine, false);
	FunctionType *FTflushInt 			  = FunctionType::get(void_type, ArgsFlushInt, false);
	FunctionType *FTflushDouble 		  = FunctionType::get(void_type, ArgsFlushDouble, false);
	FunctionType *FTflushStringC 		  = FunctionType::get(void_type, ArgsFlushStringC, false);
	FunctionType *FTflushStringCv2 		  = FunctionType::get(void_type, ArgsFlushStringCv2, false);
	FunctionType *FTflushStringObj 		  = FunctionType::get(void_type, ArgsFlushStringObj, false);
	FunctionType *FTflushBoolean 		  = FunctionType::get(void_type, ArgsFlushBoolean, false);
	FunctionType *FTflushStartEnd 		  = FunctionType::get(void_type, ArgsFlushStartEnd, false);
	FunctionType *FTflushChar 			  =	FunctionType::get(void_type, ArgsFlushChar, false);
	FunctionType *FTflushDelim 			  =	FunctionType::get(void_type, ArgsFlushDelim, false);
	FunctionType *FTflushOutput 		  =	FunctionType::get(void_type, ArgsFlushOutput, false);

	FunctionType *FTmemoryChunk 		  = FunctionType::get(void_ptr_type, ArgsMemoryChunk, false);
	FunctionType *FTincrMemoryChunk 	  =	FunctionType::get(void_ptr_type, ArgsIncrMemoryChunk, false);
	FunctionType *FTreleaseMemoryChunk 	  = FunctionType::get(void_type, ArgsRelMemoryChunk, false);

	FunctionType *FTtiming 			   	  = FunctionType::get(void_type, ArgsTiming, false);

	Function *printi_ 		= Function::Create(FTint, Function::ExternalLinkage,"printi", TheModule);
	Function *printi64_ 	= Function::Create(FTint64, Function::ExternalLinkage,"printi64", TheModule);
	Function *printc_ 		= Function::Create(FTcharPtr, Function::ExternalLinkage,"printc", TheModule);
	Function *printFloat_ 	= Function::Create(FTprintFloat_, Function::ExternalLinkage, "printFloat", TheModule);
	Function *printShort_ 	= Function::Create(FTprintShort_, Function::ExternalLinkage, "printShort", TheModule);
	Function *printBoolean_ = Function::Create(FTprintBoolean_, Function::ExternalLinkage, "printBoolean", TheModule);

	Function *atoi_ 	= Function::Create(FTcharPtr, Function::ExternalLinkage,"atoi", TheModule);
	Function *atois_ 	= Function::Create(FTatois, Function::ExternalLinkage,"atois", TheModule);
	atois_->addFnAttr(llvm::Attribute::AlwaysInline);
	Function *atof_ 	= Function::Create(FTatof, Function::ExternalLinkage,"atof", TheModule);

	Function *compareTokenString_	= Function::Create(FTcompareTokenString_,
			Function::ExternalLinkage, "compareTokenString", TheModule);
	compareTokenString_->addFnAttr(llvm::Attribute::AlwaysInline);
	Function *compareTokenString64_	= Function::Create(FTcompareTokenString64_,
				Function::ExternalLinkage, "compareTokenString64", TheModule);
	Function *stringObjEquality 		= Function::Create(FTcompareStringObjs,
			Function::ExternalLinkage, "equalStringObjs", TheModule);
	stringObjEquality->addFnAttr(llvm::Attribute::AlwaysInline);
	Function *stringEquality = Function::Create(FTcompareString,
			Function::ExternalLinkage, "equalStrings", TheModule);
	stringEquality->addFnAttr(llvm::Attribute::AlwaysInline);

	Function *convertBoolean_	= Function::Create(FTconvertBoolean_,
				Function::ExternalLinkage, "convertBoolean", TheModule);
	convertBoolean_->addFnAttr(llvm::Attribute::AlwaysInline);
	Function *convertBoolean64_ = Function::Create(FTconvertBoolean64_,
					Function::ExternalLinkage, "convertBoolean64", TheModule);
	convertBoolean64_->addFnAttr(llvm::Attribute::AlwaysInline);

	/**
	 * Hashing
	 */
	Function *hashInt_ = Function::Create(FThashInt, Function::ExternalLinkage,
			"hashInt", TheModule);
	Function *hashDouble_ = Function::Create(FThashDouble,
			Function::ExternalLinkage, "hashDouble", TheModule);
	Function *hashStringC_ = Function::Create(FThashStringC,
			Function::ExternalLinkage, "hashStringC", TheModule);
	Function *hashStringObj_ = Function::Create(FThashStringObj,
			Function::ExternalLinkage, "hashStringObject", TheModule);
	Function *hashBoolean_ = Function::Create(FThashBoolean,
			Function::ExternalLinkage, "hashBoolean", TheModule);
	Function *hashCombine_ = Function::Create(FThashCombine,
			Function::ExternalLinkage, "combineHashes", TheModule);
	Function *hashCombineNoOrder_ = Function::Create(FThashCombine,
			Function::ExternalLinkage, "combineHashesNoOrder", TheModule);

	/**
	 * Debug (TMP)
	 */
	vector<Type*> ArgsDebug;
	ArgsDebug.insert(ArgsDebug.begin(),void_ptr_type);
	FunctionType *FTdebug = FunctionType::get(void_type, ArgsDebug, false);
	Function *debug_ = Function::Create(FTdebug, Function::ExternalLinkage,
				"debug", TheModule);

	/**
	* Flushing
	*/
	Function *flushInt_ = Function::Create(FTflushInt,
			Function::ExternalLinkage, "flushInt", TheModule);
	Function *flushDouble_ = Function::Create(FTflushDouble,
			Function::ExternalLinkage, "flushDouble", TheModule);
	Function *flushStringC_ = Function::Create(FTflushStringC,
			Function::ExternalLinkage, "flushStringC", TheModule);
	Function *flushStringCv2_ = Function::Create(FTflushStringCv2,
				Function::ExternalLinkage, "flushStringReady", TheModule);
	Function *flushStringObj_ = Function::Create(FTflushStringObj,
					Function::ExternalLinkage, "flushStringObject", TheModule);
	Function *flushBoolean_ = Function::Create(FTflushBoolean,
			Function::ExternalLinkage, "flushBoolean", TheModule);
	Function *flushObjectStart_ = Function::Create(FTflushStartEnd,
				Function::ExternalLinkage, "flushObjectStart", TheModule);
	Function *flushArrayStart_ = Function::Create(FTflushStartEnd,
				Function::ExternalLinkage, "flushArrayStart", TheModule);
	Function *flushObjectEnd_ = Function::Create(FTflushStartEnd,
				Function::ExternalLinkage, "flushObjectEnd", TheModule);
	Function *flushArrayEnd_ = Function::Create(FTflushStartEnd,
				Function::ExternalLinkage, "flushArrayEnd", TheModule);
	Function *flushChar_ = Function::Create(FTflushChar,
					Function::ExternalLinkage, "flushChar", TheModule);
	Function *flushDelim_ = Function::Create(FTflushDelim,
						Function::ExternalLinkage, "flushDelim", TheModule);
	Function *flushOutput_ = Function::Create(FTflushOutput,
						Function::ExternalLinkage, "flushOutput", TheModule);

	/* Memory Management */
	Function *getMemoryChunk_ = Function::Create(FTmemoryChunk,
				Function::ExternalLinkage, "getMemoryChunk", TheModule);
	Function *increaseMemoryChunk_ = Function::Create(FTincrMemoryChunk,
					Function::ExternalLinkage, "increaseMemoryChunk", TheModule);
	Function *releaseMemoryChunk_ = Function::Create(FTreleaseMemoryChunk,
						Function::ExternalLinkage, "releaseMemoryChunk", TheModule);

	/* Timing */
	Function *resetTime_ = Function::Create(FTtiming, Function::ExternalLinkage,
			"resetTime", TheModule);
	Function *calculateTime_ = Function::Create(FTtiming,
			Function::ExternalLinkage, "calculateTime", TheModule);

	//Memcpy - not used (yet)
	Type* types[] = { void_ptr_type, void_ptr_type, Type::getInt32Ty(ctx) };
	Function* memcpy_ = Intrinsic::getDeclaration(TheModule, Intrinsic::memcpy, types);
	if (memcpy_ == NULL) {
		throw runtime_error(string("Could not find memcpy intrinsic"));
	}

	/**
	 * HASHTABLES FOR JOINS / AGGREGATIONS
	 */
	//Last type is needed to capture file size. Tentative
	Type* ht_int_types[] = { int32_type, int32_type, void_ptr_type, int32_type };
	FunctionType *FTintHT = FunctionType::get(void_type, ht_int_types, false);
	Function* insertIntKeyToHT_ = Function::Create(FTintHT, Function::ExternalLinkage, "insertIntKeyToHT", TheModule);

	Type* ht_types[] = { char_ptr_type, int64_type, void_ptr_type, int32_type };
	FunctionType *FT_HT = FunctionType::get(void_type, ht_types, false);
	Function* insertToHT_ = Function::Create(FT_HT, Function::ExternalLinkage, "insertToHT", TheModule);

	Type* ht_int_probe_types[] = { int32_type, int32_type, int32_type };
	PointerType* void_ptr_ptr_type = context.getPointerType(void_ptr_type);
	FunctionType *FTint_probeHT = FunctionType::get(void_ptr_ptr_type, ht_int_probe_types, false);
	Function* probeIntHT_ = Function::Create(FTint_probeHT,	Function::ExternalLinkage, "probeIntHT", TheModule);
	probeIntHT_->addFnAttr(llvm::Attribute::AlwaysInline);

	Type* ht_probe_types[] = { char_ptr_type, int64_type };
	FunctionType *FT_probeHT = FunctionType::get(void_ptr_ptr_type, ht_probe_types, false);
	Function* probeHT_ = Function::Create(FT_probeHT,	Function::ExternalLinkage, "probeHT", TheModule);
	probeHT_->addFnAttr(llvm::Attribute::AlwaysInline);

	Type* ht_get_metadata_types[] = { char_ptr_type };
	StructType *metadataType = context.getHashtableMetadataType();
	PointerType *metadataArrayType = PointerType::get(metadataType,0);
	FunctionType *FTget_metadata_HT = FunctionType::get(metadataArrayType,
			ht_get_metadata_types, false);
	Function* getMetadataHT_ = Function::Create(FTget_metadata_HT,
			Function::ExternalLinkage, "getMetadataHT", TheModule);

	/**
	 * Radix
	 */
	/* What the type of HT buckets is */
	vector<Type*> htBucketMembers;
	//int *bucket;
	htBucketMembers.push_back(int32_ptr_type);
	//int *next;
	htBucketMembers.push_back(int32_ptr_type);
	//uint32_t mask;
	htBucketMembers.push_back(int32_type);
	//int count;
	htBucketMembers.push_back(int32_type);
	StructType *htBucketType = StructType::get(ctx, htBucketMembers);
	PointerType *htBucketPtrType = PointerType::get(htBucketType, 0);

	/* JOIN!!! */
	/* What the type of HT entries is */
	/* (int32, void*) */
	vector<Type*> htEntryMembers;
	htEntryMembers.push_back(int32_type);
	htEntryMembers.push_back(int64_type);
	StructType *htEntryType = StructType::get(ctx,htEntryMembers);
	PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

	Type* radix_partition_types[] = { int64_type, htEntryPtrType };
	FunctionType *FTradix_partition = FunctionType::get(int32_ptr_type, radix_partition_types, false);
	Function *radix_partition = Function::Create(FTradix_partition,
							Function::ExternalLinkage, "partitionHTLLVM", TheModule);

	Type* bucket_chaining_join_prepare_types[] = { htEntryPtrType, int32_type,
			htBucketPtrType };
	FunctionType *FTbucket_chaining_join_prepare = FunctionType::get(void_type,
			bucket_chaining_join_prepare_types, false);
	Function *bucket_chaining_join_prepare = Function::Create(
			FTbucket_chaining_join_prepare, Function::ExternalLinkage,
			"bucket_chaining_join_prepareLLVM", TheModule);

	/* AGGR! */
	/* What the type of HT entries is */
	/* (int64, void*) */
	vector<Type*> htAggEntryMembers;
	htAggEntryMembers.push_back(int64_type);
	htAggEntryMembers.push_back(int64_type);
	StructType *htAggEntryType = StructType::get(ctx,htAggEntryMembers);
		PointerType *htAggEntryPtrType = PointerType::get(htAggEntryType, 0);
	Type* radix_partition_agg_types[] = { int64_type, htAggEntryPtrType };
	FunctionType *FTradix_partition_agg = FunctionType::get(int32_ptr_type,
			radix_partition_agg_types, false);
	Function *radix_partition_agg = Function::Create(FTradix_partition_agg,
			Function::ExternalLinkage, "partitionAggHTLLVM", TheModule);

	Type* bucket_chaining_agg_prepare_types[] = { htAggEntryPtrType, int32_type,
			htBucketPtrType };
	FunctionType *FTbucket_chaining_agg_prepare = FunctionType::get(void_type,
			bucket_chaining_agg_prepare_types, false);
	Function *bucket_chaining_agg_prepare = Function::Create(
			FTbucket_chaining_agg_prepare, Function::ExternalLinkage,
			"bucket_chaining_agg_prepareLLVM", TheModule);
	/**
	 * End of Radix
	 */


	/**
	 * Parsing
	 */
	Type* newline_types[] = { char_ptr_type , int64_type };
	FunctionType *FT_newline = FunctionType::get(int64_type, newline_types, false);
	Function *newline = Function::Create(FT_newline, Function::ExternalLinkage,
			"newlineAVX", TheModule);
	/* Does not make a difference... */
	newline->addFnAttr(llvm::Attribute::AlwaysInline);

//	vector<Type*> tokenMembers;
//	tokenMembers.push_back(int32_type);
//	tokenMembers.push_back(int32_type);
//	tokenMembers.push_back(int32_type);
//	tokenMembers.push_back(int32_type);
//	StructType *tokenType = StructType::get(ctx,tokenMembers);
	StructType *tokenType = context.CreateJSMNStruct();


	PointerType *tokenPtrType = PointerType::get(tokenType, 0);
	PointerType *token2DPtrType = PointerType::get(tokenPtrType, 0);
	Type* parse_line_json_types[] = { char_ptr_type, int64_type, int64_type,
			token2DPtrType, int64_type };
	FunctionType *FT_parse_line_json =
			FunctionType::get(void_type, parse_line_json_types, false);
	Function *parse_line_json = Function::Create(FT_parse_line_json,
			Function::ExternalLinkage, "parseLineJSON", TheModule);


	context.registerFunction("printi", printi_);
	context.registerFunction("printi64", printi64_);
	context.registerFunction("printFloat", printFloat_);
	context.registerFunction("printShort", printShort_);
	context.registerFunction("printBoolean", printBoolean_);
	context.registerFunction("printc", printc_);

	context.registerFunction("atoi", atoi_);
	context.registerFunction("atois", atois_);
	context.registerFunction("atof", atof_);

	context.registerFunction("insertInt", insertIntKeyToHT_);
	context.registerFunction("probeInt", probeIntHT_);
	context.registerFunction("insertHT", insertToHT_);
	context.registerFunction("probeHT", probeHT_);
	context.registerFunction("getMetadataHT", getMetadataHT_);

	context.registerFunction("compareTokenString", compareTokenString_);
	context.registerFunction("compareTokenString64", compareTokenString64_);
	context.registerFunction("convertBoolean", convertBoolean_);
	context.registerFunction("convertBoolean64", convertBoolean64_);
	context.registerFunction("equalStringObjs", stringObjEquality);
	context.registerFunction("equalStrings", stringEquality);

	context.registerFunction("hashInt", hashInt_);
	context.registerFunction("hashDouble", hashDouble_);
	context.registerFunction("hashStringC", hashStringC_);
	context.registerFunction("hashStringObject", hashStringObj_);
	context.registerFunction("hashBoolean", hashBoolean_);
	context.registerFunction("combineHashes", hashCombine_);
	context.registerFunction("combineHashesNoOrder", hashCombineNoOrder_);

	context.registerFunction("flushInt", flushInt_);
	context.registerFunction("flushDouble", flushDouble_);
	context.registerFunction("flushStringC", flushStringC_);
	context.registerFunction("flushStringCv2", flushStringCv2_);
	context.registerFunction("flushStringObj", flushStringObj_);
	context.registerFunction("flushBoolean", flushBoolean_);
	context.registerFunction("flushChar", flushChar_);
	context.registerFunction("flushDelim", flushDelim_);
	context.registerFunction("flushOutput", flushOutput_);

	context.registerFunction("flushObjectStart", flushObjectStart_);
	context.registerFunction("flushArrayStart", flushArrayStart_);
	context.registerFunction("flushObjectEnd", flushObjectEnd_);
	context.registerFunction("flushArrayEnd", flushArrayEnd_);
	context.registerFunction("flushArrayEnd", flushArrayEnd_);

	context.registerFunction("getMemoryChunk", getMemoryChunk_);
	context.registerFunction("increaseMemoryChunk", increaseMemoryChunk_);
	context.registerFunction("releaseMemoryChunk", releaseMemoryChunk_);
	context.registerFunction("memcpy", memcpy_);

	context.registerFunction("resetTime", resetTime_);
	context.registerFunction("calculateTime", calculateTime_);

	context.registerFunction("partitionHT",radix_partition);
	context.registerFunction("bucketChainingPrepare",bucket_chaining_join_prepare);
	context.registerFunction("partitionAggHT",radix_partition_agg);
	context.registerFunction("bucketChainingAggPrepare",bucket_chaining_agg_prepare);

	context.registerFunction("newline",newline);
	context.registerFunction("parseLineJSON",parse_line_json);
}
