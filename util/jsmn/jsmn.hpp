//#ifndef __JSMN_HPP_
//#define __JSMN_HPP_
//
//#include "common/common.hpp"
//
//#ifdef __cplusplus
//extern "C" {
//#endif
//
//#define DEBUGJSMN
//
///**
// * JSON type identifier. Basic types are:
// * 	o Object
// * 	o Array
// * 	o String
// * 	o Other primitive: number, boolean (true/false) or null
// */
//typedef enum {
//	JSMN_PRIMITIVE = 0,
//	JSMN_OBJECT = 1,
//	JSMN_ARRAY = 2,
//	JSMN_STRING = 3
//} jsmntype_t;
//
//typedef enum {
//	JSMN_OK = 0,
//	/* Not enough tokens were provided */
//	JSMN_ERROR_NOMEM = -1,
//	/* Invalid character inside JSON string */
//	JSMN_ERROR_INVAL = -2,
//	/* The string is not a full JSON packet, more bytes expected */
//	JSMN_ERROR_PART = -3,
//} jsmnerr_t;
//
///**
// * JSON token description.
// * @param		type	type (object, array, string etc.)
// * @param		start	start position in JSON data string
// * @param		end		end position in JSON data string
// */
//#define JSON_TIGHT
//
///**
// * XXX DATASET-SPECIFIC ATM
// */
///* Used to accommodate symantec workload (and flush its pm) */
//#define JSON_SYMANTEC
//#define JSON_SYMANTEC_WIDE
///* Used to accommodate very wide TPC-H pre-computed join
// * i.e. ordersLineitem.json */
////#define JSON_TPCH_WIDE
//
//
//#if !defined(JSON_TPCH_WIDE) && !defined(JSON_SYMANTEC_WIDE)
///* Only flush out equi-width json pm */
//#define JSON_FLUSH
//#endif
//
//#ifndef JSON_TIGHT
//typedef struct {
//	jsmntype_t type;
//	int start;
//	int end;
//	int size;
//#ifdef JSMN_PARENT_LINKS
//	int parent;
//#endif
//} jsmntok_t;
//#endif
//
///* NOTE: Changes affect RawContext::CreateJSMNStruct() */
//#ifdef JSON_TIGHT
//typedef struct {
//	char type;
//	short start;
//	short end;
//	char size;
//} jsmntok_t;
//
///* Super tight cases
// * NOT applicable to TPC-H (tuple length ~ 350) */
////typedef struct {
////	char type;
////	char start;
////	char end;
////	char size;
////} jsmntok_t;
//#endif
//
//#ifdef JSON_TIGHT
//
////Sufficient for lineitem.json
////#define MAXTOKENS 50
//#ifdef JSON_TPCH_WIDE
//#undef MAXTOKENS
//////Used for ordersLineitem
//#define MAXTOKENS 52
////Used to test reallocs locally
////#define MAXTOKENS 17
//#endif /* JSON_TPCH_WIDE */
//
//#ifndef JSON_TPCH_WIDE
////33 is Exactly enough for lineitem.json (2 x #fields + 1 for the obj.)
////Wider ones will break
//#undef MAXTOKENS
//#define MAXTOKENS 300 //33
//#endif /* NOT_JSON_TPCH_WIDE */
//
//#endif /* JSON_TIGHT */
//
//#ifdef JSON_TPCH_WIDE
//#undef JSON_SYMANTEC
//#endif
//
//#ifdef JSON_SYMANTEC
//#undef MAXTOKENS
//#define MAXTOKENS 80 //53
//#endif /* JSON_SYMANTEC */
//
///* Default, conservative case */
//#ifndef JSON_TIGHT
//#undef MAXTOKENS
//#define MAXTOKENS 1000
//#endif
//
///**
// * JSON parser. Contains an array of token blocks available. Also stores
// * the string being parsed now and current position in that string
// */
//typedef struct {
//	unsigned int pos; /* offset in the JSON string */
//	unsigned int toknext; /* next token to allocate */
//	int toksuper; /* superior token node, e.g parent object or array */
//} jsmn_parser;
//
///**
// * Create JSON parser over an array of tokens
// */
//void jsmn_init(jsmn_parser *parser);
//
///**
// * Run JSON parser. It parses a JSON data string into and array of tokens, each describing
// * a single JSON object.
// */
//jsmnerr_t jsmn_parse(jsmn_parser *parser, const char *js, size_t len,
//		jsmntok_t **tokens, size_t num_tokens);
//
//#ifdef __cplusplus
//}
//#endif
//
//#endif /* __JSMN_H_ */
//
